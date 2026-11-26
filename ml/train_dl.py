"""Unified Binary Classification Training, Baseline Comparison, Champion Improvement, and ONNX Export.

Problem (single unified formulation):
  Session-prefix Purchase Intent Prediction (Binary Classification):
    Given observed prefix S_k = (e_1, ..., e_k) up to the last L = 50 events prior to checkout/purchase,
    predict calibrated probability P(session ends with purchase | S_k) in (0, 1).

Pipeline:
  Stage 1 — Fair Baseline & Candidate Comparison (identical train/val/test samples):
    - Classical Baselines : Markov Chain (1st-order + backoff), HistGradientBoosting (32 tabular features)
    - Deep Learning Suite : GRU, Hybrid (GRU + Wide&Deep), Causal TCN, Bi-LSTM + Additive Attention,
                            MLP-Mixer, Causal Transformer
    All evaluated on VAL and TEST (ROC-AUC, PR-AUC, Log-Loss, Brier Score, Best F1, Params, ONNX CPU Latency).

  Stage 2 — Champion Selection, Improvement & Ablation Study:
    - Selects the best baseline balancing accuracy & real-time ONNX latency (< 1 ms at batch=1).
    - Develops & evaluates `hybrid_v2` (GRU + Shop-Conditioned Query Attention + ONNX-native 44-dim Wide
      Prefix Features + Shop-Conditioned FiLM + Warmup-Cosine + Per-Shop Platt Calibration).
    - Runs full ablation suite (both GRU ablations and `hybrid_v2` component ablations).

  Stage 3 — ONNX Export & Parity Verification (G4):
    - Exports Champion -> models/intent_v1.onnx and Challenger -> models/intent_transformer_full.onnx
    - Verifies PyTorch vs ONNX parity (< 1e-4 on 1,000 test samples across batch=1 and batch=1000)
    - Writes models/model_card.json, models/feature_spec.json, data/prefix/_dl_ablation.json,
      and data/prefix/_model_benchmark.json.

Run:
  python ml/train_dl.py --prefix data/prefix --out-dir models --report data/prefix/_dl_ablation.json
"""
import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from ml import features as FT  # noqa: E402
from ml.baselines import (  # noqa: E402
    BUCKET_NAMES,
    GBMPurchaseBaseline,
    MarkovPurchaseBaseline,
    evaluate_predictions,
    extract_tabular_and_markov,
)
from ml.models import BASELINE_DL_ARCHS, IntentModel  # noqa: E402


# ---------------------------------------------------------------- Unified dataset loader
class PrefixSequenceDataset:
    """Stores pre-transformed step arrays per session in contiguous flat arrays, plus selected (sess_idx, k)
    samples. Optionally carries matched tabular/Markov arrays extracted from the exact same (sess_idx, k) pairs.
    """

    def __init__(self, cat_flat, num_flat, offsets, sess_shop_idx, sess_shop_name, sess_label,
                 sample_sess, sample_k, max_len=FT.MAX_LEN, tabular=None):
        self.cat_flat = cat_flat
        self.num_flat = num_flat
        self.offsets = offsets
        self.sess_shop_idx = sess_shop_idx
        self.sess_shop_name = sess_shop_name
        self.sess_label = sess_label
        self.sample_sess = sample_sess
        self.sample_k = sample_k
        self.max_len = int(max_len)
        self.tabular = tabular

        self.y = self.sess_label[self.sample_sess]
        self.k = self.sample_k
        self.shop_idx = self.sess_shop_idx[self.sample_sess]
        self.shop_id = self.sess_shop_name[self.sample_sess]
        self._x_cat = None
        self._x_num = None
        self._lengths = None

    def __len__(self):
        return len(self.sample_sess)

    def _ensure_materialized(self):
        if self._x_cat is not None:
            return
        n = len(self.sample_sess)
        L = self.max_len
        x_cat = np.zeros((n, L, FT.N_CAT), dtype=np.int64)
        x_num = np.zeros((n, L, FT.N_NUM), dtype=np.float32)
        lengths = np.empty(n, dtype=np.int64)
        s_arr = self.sample_sess
        k_arr = self.sample_k
        offs = self.offsets[s_arr]
        for i in range(n):
            k = int(k_arr[i])
            base = int(offs[i])
            lo = k - L if k > L else 0
            m = k - lo
            x_cat[i, L - m:] = self.cat_flat[base + lo: base + k]
            x_num[i, L - m:] = self.num_flat[base + lo: base + k]
            lengths[i] = m
        self._x_cat = x_cat
        self._x_num = x_num
        self._lengths = lengths

    def get_batch(self, indices):
        self._ensure_materialized()
        return (self._x_cat[indices], self._x_num[indices], self._lengths[indices],
                self.shop_idx[indices], self.y[indices].astype(np.float32))


def load_sequence_splits(prefix_dir, spec, max_sessions_per_split=12000, max_k_per_session=6,
                         seed=42, include_tabular=False):
    """Load sessions from data/prefix/sessions and build PrefixSequenceDataset for train, val, test."""
    base = os.path.join(prefix_dir, "sessions")
    shop_dirs = sorted(d for d in os.listdir(base) if d.startswith("shop_id="))
    rng = np.random.default_rng(seed)
    max_len = int(spec["max_len"])

    import pandas as pd
    splits_parts = defaultdict(list)
    for sd in shop_dirs:
        shop_name = sd.split("=", 1)[1]
        s_path = os.path.join(base, sd)
        # Keep natural multi-shop ratio: 3x sessions for REES46 vs synthetic
        cap = (max_sessions_per_split * 3 if shop_name == "rees46_cosmetics" else max_sessions_per_split)
        for spd in sorted(d for d in os.listdir(s_path) if d.startswith("split=")):
            split_name = spd.split("=", 1)[1]
            part_dir = os.path.join(s_path, spd)
            pfiles = sorted(os.path.join(part_dir, f) for f in os.listdir(part_dir) if f.endswith(".parquet"))
            dfs = []
            rows_so_far = 0
            target_read = (cap * 2) if cap else 10**9
            for pf in pfiles:
                sub = pq.read_table(pf).to_pandas()
                dfs.append(sub)
                rows_so_far += len(sub)
                if rows_so_far >= target_read:
                    break
            df = pd.concat(dfs, ignore_index=True)
            df["shop_id"] = shop_name
            df["split"] = split_name
            if cap and len(df) > cap:
                idx = rng.choice(len(df), size=cap, replace=False)
                df = df.iloc[np.sort(idx)].reset_index(drop=True)
            splits_parts[split_name].append(df)

    datasets = {}
    for split_name, parts in splits_parts.items():
        sdf = pd.concat(parts, ignore_index=True)
        steps_col = sdf["steps"].tolist()
        n_sess = len(steps_col)
        sess_label = sdf["label"].to_numpy(dtype=np.int32)
        sess_shop_idx = sdf["shop_idx"].to_numpy(dtype=np.int64)
        sess_shop_name = sdf["shop_id"].to_numpy(dtype=object)

        offsets = np.zeros(n_sess + 1, dtype=np.int64)
        ks_list = []
        total_steps = 0
        total_samples = 0
        for i, st in enumerate(steps_col):
            n = len(st)
            offsets[i] = total_steps
            total_steps += n
            if n == 0:
                ks_list.append([])
                continue
            if max_k_per_session and n > max_k_per_session:
                cand = sorted(set([1, min(2, n), n] + rng.choice(range(1, n + 1), size=max_k_per_session,
                                                                 replace=False).tolist()))
            else:
                cand = list(range(1, n + 1))
            ks_list.append(cand)
            total_samples += len(cand)
        offsets[n_sess] = total_steps

        cat_flat = np.zeros((total_steps, FT.N_CAT), dtype=np.int64)
        num_flat = np.zeros((total_steps, FT.N_NUM), dtype=np.float32)
        sample_sess = np.zeros(total_samples, dtype=np.int32)
        sample_k = np.zeros(total_samples, dtype=np.int32)

        p_step = 0
        p_samp = 0
        for i, st in enumerate(steps_col):
            for step in st:
                t_idx = int(step["type_idx"])
                c_idx = int(step["cat_idx"])
                b_idx = int(step["brand_idx"])
                price = float(step["price"])
                qty = int(step["qty"])
                dt_ms = int(step["dt_prev_ms"])
                tf_ms = int(step["t_first_ms"])
                same = int(step["same_prod"])
                n_view = int(step["n_view"])
                n_add = int(step["n_add"])
                n_rem = int(step["n_remove"])
                cart_u = int(step["cart_units"])
                pos = int(step["pos"])

                cat_flat[p_step] = (t_idx, c_idx, b_idx)
                num_flat[p_step] = (
                    math.log1p(max(price, 0.0)),
                    math.log1p(max(qty, 0)),
                    math.log1p(dt_ms / 1000.0),
                    math.log1p(tf_ms / 1000.0),
                    float(same),
                    math.log1p(n_view),
                    math.log1p(n_add),
                    math.log1p(n_rem),
                    math.copysign(math.log1p(abs(cart_u)), cart_u) if cart_u else 0.0,
                    math.log1p(pos),
                )
                p_step += 1

            for k in ks_list[i]:
                sample_sess[p_samp] = i
                sample_k[p_samp] = k
                p_samp += 1

        tabular = None
        if include_tabular:
            tabular = extract_tabular_and_markov(steps_col, sess_label, sess_shop_idx, sess_shop_name, ks_list)

        datasets[split_name] = PrefixSequenceDataset(
            cat_flat, num_flat, offsets, sess_shop_idx, sess_shop_name, sess_label,
            sample_sess, sample_k, max_len=max_len, tabular=tabular,
        )
    return datasets


# ---------------------------------------------------------------- Training, calibration & inference
@torch.no_grad()
def predict_logits_and_probs(model, ds, batch_size=2048, device="cpu"):
    model.eval()
    n = len(ds)
    logits = np.empty(n, dtype=np.float32)
    probs = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        xc, xn, ln, sh, _ = ds.get_batch(idx)
        lg = model.forward_logits(
            torch.from_numpy(xc).to(device),
            torch.from_numpy(xn).to(device),
            torch.from_numpy(ln).to(device),
            torch.from_numpy(sh).to(device),
        )
        pr = torch.sigmoid(lg)
        logits[start:start + len(idx)] = lg.cpu().numpy()
        probs[start:start + len(idx)] = pr.cpu().numpy()
    return logits, probs


def fit_platt_calibration(raw_logits, y_true, shop_idx):
    """Fit per-shop Platt scaling (a, b) with a >= 0.1 on VAL to minimize NLL (LogLoss)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    raw_logits = np.asarray(raw_logits, dtype=np.float64)
    shop_idx = np.asarray(shop_idx, dtype=np.int64)
    params = {}
    for s in np.unique(shop_idx):
        m = (shop_idx == s)
        z = raw_logits[m]
        y = y_true[m]
        if len(y) < 20 or y.sum() == 0 or y.sum() == len(y):
            params[int(s)] = (1.0, 0.0)
            continue

        def nll(ab):
            a, b = ab
            u = np.clip(a * z + b, -30.0, 30.0)
            loss = np.maximum(u, 0.0) - u * y + np.log1p(np.exp(-np.abs(u)))
            return float(loss.mean()) + 1e-4 * ((a - 1.0) ** 2)

        res = minimize(nll, x0=[1.0, 0.0], bounds=[(0.1, 5.0), (-10.0, 10.0)], method="L-BFGS-B")
        params[int(s)] = (round(float(res.x[0]), 6), round(float(res.x[1]), 6))
    return params


def _build_lr_scheduler(optimizer, total_steps, warmup_frac=0.1, min_lr_ratio=0.1):
    warmup_steps = max(int(total_steps * warmup_frac), 1) if total_steps > 10 else 0

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(spec, train_ds, val_ds, arch="gru", d_model=32, n_layers=1, n_heads=4, dropout=0.1,
                use_cat_brand=True, use_num=True, use_wide=True, use_attn=True, use_film=True,
                use_pos_weight=True, calibrate_on_val=True, label_smoothing=0.0, swa_top2=False,
                epochs=3, batch_size=1536, lr=2.5e-3, weight_decay=1e-4, warmup_frac=0.0,
                seed=42, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = IntentModel(
        spec, arch=arch, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
        use_cat_brand=use_cat_brand, use_num=use_num, use_wide=use_wide, use_attn=use_attn, use_film=use_film,
    ).to(device)

    pos_rate = float(train_ds.y.mean())
    pw_val = math.sqrt((1.0 - pos_rate) / max(pos_rate, 1e-4)) if use_pos_weight else 1.0
    pos_weight = torch.tensor(pw_val, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(train_ds)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    scheduler = _build_lr_scheduler(optimizer, max(epochs * steps_per_epoch, 1), warmup_frac=warmup_frac)

    epoch_states = []
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xc, xn, ln, sh, y = train_ds.get_batch(idx)
            if label_smoothing > 0.0:
                y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            t_xc = torch.from_numpy(xc).to(device)
            t_xn = torch.from_numpy(xn).to(device)
            t_ln = torch.from_numpy(ln).to(device)
            t_sh = torch.from_numpy(sh).to(device)
            t_y = torch.from_numpy(y).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(t_xc, t_xn, t_ln, t_sh)
            loss = criterion(logits, t_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += float(loss.item())
            n_batches += 1

        model.set_calibration({s: (1.0, 0.0) for s in range(int(spec["cardinality"]["shop"]))})
        _, val_probs = predict_logits_and_probs(model, val_ds, batch_size=4096, device=device)
        ev_uncal = evaluate_predictions(val_ds.y, val_probs, val_ds.k, val_ds.shop_id)
        shop_aucs = [v["overall"]["roc_auc"] for v in ev_uncal["by_shop"].values() if v["overall"]["roc_auc"]]
        mean_shop_auc = float(np.mean(shop_aucs)) if shop_aucs else ev_uncal["overall"]["roc_auc"]
        overall_auc = float(ev_uncal["overall"]["roc_auc"] or 0.0)
        sel_score = 0.5 * overall_auc + 0.5 * mean_shop_auc
        dt = round(time.time() - t0, 2)
        history.append({
            "epoch": ep,
            "train_loss": round(ep_loss / max(n_batches, 1), 4),
            "val_mean_shop_auc": round(mean_shop_auc, 4),
            "val_overall_auc_uncal": overall_auc,
            "val_overall_pr_auc_uncal": float(ev_uncal["overall"]["pr_auc"] or 0.0),
            "time_s": dt,
        })
        epoch_states.append((sel_score, copy.deepcopy(model.state_dict())))

    epoch_states.sort(key=lambda item: item[0], reverse=True)
    if swa_top2 and len(epoch_states) >= 2:
        sd = {}
        st_a, st_b = epoch_states[0][1], epoch_states[1][1]
        for k_name in st_a:
            sd[k_name] = 0.5 * st_a[k_name] + 0.5 * st_b[k_name] if st_a[k_name].is_floating_point() else st_a[k_name]
        model.load_state_dict(sd)
    else:
        model.load_state_dict(epoch_states[0][1])

    model.set_calibration({s: (1.0, 0.0) for s in range(int(spec["cardinality"]["shop"]))})
    raw_val_logits, raw_val_probs = predict_logits_and_probs(model, val_ds, batch_size=4096, device=device)
    uncal_brier = float(np.mean((raw_val_probs - val_ds.y) ** 2))

    calib_params = {int(s): (1.0, 0.0) for s in np.unique(val_ds.shop_idx)}
    if calibrate_on_val:
        calib_params = fit_platt_calibration(raw_val_logits, val_ds.y, val_ds.shop_idx)
        model.set_calibration(calib_params)

    cal_logits = raw_val_logits.astype(np.float64).copy()
    for s_idx, (a_val, b_val) in calib_params.items():
        m = (val_ds.shop_idx == s_idx)
        cal_logits[m] = a_val * raw_val_logits[m] + b_val
    cal_val_probs = 1.0 / (1.0 + np.exp(-np.clip(cal_logits, -30.0, 30.0)))
    cal_brier = float(np.mean((cal_val_probs - val_ds.y) ** 2))

    info = {
        "pos_weight": round(pw_val, 4),
        "calibrated": bool(calibrate_on_val),
        "calibration_params_by_shop_idx": {str(k): list(v) for k, v in calib_params.items()},
        "val_brier_before_calib": round(uncal_brier, 4),
        "val_brier_after_calib": round(cal_brier, 4),
        "history": history,
    }
    return model, info


# ---------------------------------------------------------------- ONNX export, parity & latency
def export_and_verify_onnx(model, spec, test_ds, onnx_path, n_verify=1000):
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    model_cpu = copy.deepcopy(model).cpu().eval()
    L = int(spec["max_len"])
    xc1 = torch.zeros((1, L, FT.N_CAT), dtype=torch.int64)
    xn1 = torch.zeros((1, L, FT.N_NUM), dtype=torch.float32)
    ln1 = torch.ones((1,), dtype=torch.int64)
    sh1 = torch.full((1,), FT.FIRST_IDX, dtype=torch.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model_cpu,
            (xc1, xn1, ln1, sh1),
            onnx_path,
            input_names=["x_cat", "x_num", "length", "shop"],
            output_names=["prob"],
            dynamic_axes={"x_cat": {0: "batch"}, "x_num": {0: "batch"},
                          "length": {0: "batch"}, "shop": {0: "batch"}, "prob": {0: "batch"}},
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    n = min(n_verify, len(test_ds))
    idx = np.linspace(0, len(test_ds) - 1, num=n, dtype=np.int64)
    xc, xn, ln, sh, _ = test_ds.get_batch(idx)
    with torch.no_grad():
        pt_prob = model_cpu(torch.from_numpy(xc), torch.from_numpy(xn),
                            torch.from_numpy(ln), torch.from_numpy(sh)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_prob = sess.run(["prob"], {"x_cat": xc, "x_num": xn, "length": ln, "shop": sh})[0]
    abs_diff = np.abs(pt_prob - ort_prob)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    ort_b1 = sess.run(["prob"], {"x_cat": xc[:1], "x_num": xn[:1], "length": ln[:1], "shop": sh[:1]})[0]
    max_diff_b1 = float(np.max(np.abs(pt_prob[:1] - ort_b1)))

    assert max_diff < 1e-4 and max_diff_b1 < 1e-4, f"ONNX parity failed: max_diff={max_diff}, b1={max_diff_b1}"
    return {
        "onnx_path": onnx_path,
        "onnx_size_bytes": os.path.getsize(onnx_path),
        "samples_verified": int(n),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "max_abs_diff_batch1": max_diff_b1,
        "tolerance": 1e-4,
        "passed": bool(max_diff < 1e-4 and max_diff_b1 < 1e-4),
    }


def benchmark_onnx_latency(onnx_path, test_ds, warmup=20, runs_b1=200, runs_b128=50):
    """Measure CPU ONNX inference latency (single-thread, matching Spark Q2 worker) at b=1 and b=128."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    xc, xn, ln, sh, _ = test_ds.get_batch(np.arange(min(128, len(test_ds))))
    feed1 = {"x_cat": xc[:1], "x_num": xn[:1], "length": ln[:1], "shop": sh[:1]}
    feed128 = {"x_cat": xc, "x_num": xn, "length": ln, "shop": sh}

    for _ in range(warmup):
        sess.run(["prob"], feed1)
    t1 = []
    for _ in range(runs_b1):
        t0 = time.perf_counter()
        sess.run(["prob"], feed1)
        t1.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(max(warmup // 2, 5)):
        sess.run(["prob"], feed128)
    t128 = []
    for _ in range(runs_b128):
        t0 = time.perf_counter()
        sess.run(["prob"], feed128)
        t128.append((time.perf_counter() - t0) * 1000.0)

    return {
        "b1_median_ms": round(float(np.median(t1)), 4),
        "b1_mean_ms": round(float(np.mean(t1)), 4),
        "b128_median_ms": round(float(np.median(t128)), 4),
        "b128_mean_ms": round(float(np.mean(t128)), 4),
        "runtime": "onnx_cpu_1t",
    }


def benchmark_native_latency(predict_b1_fn, predict_b128_fn, warmup=10, runs_b1=100, runs_b128=30):
    for _ in range(warmup):
        predict_b1_fn()
    t1 = []
    for _ in range(runs_b1):
        t0 = time.perf_counter()
        predict_b1_fn()
        t1.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(max(warmup // 2, 5)):
        predict_b128_fn()
    t128 = []
    for _ in range(runs_b128):
        t0 = time.perf_counter()
        predict_b128_fn()
        t128.append((time.perf_counter() - t0) * 1000.0)

    return {
        "b1_median_ms": round(float(np.median(t1)), 4),
        "b1_mean_ms": round(float(np.mean(t1)), 4),
        "b128_median_ms": round(float(np.median(t128)), 4),
        "b128_mean_ms": round(float(np.mean(t128)), 4),
        "runtime": "native",
    }


# ---------------------------------------------------------------- Reporting helpers
def _selection_score(ev_val, latency_b1_ms=None, latency_budget_ms=1.0):
    """Balanced champion selection score on VAL: average of overall ROC-AUC and mean per-shop ROC-AUC,
    penalizing candidates that exceed the real-time b=1 latency budget."""
    ov = ev_val["overall"]
    s_aucs = [v["overall"]["roc_auc"] for v in ev_val["by_shop"].values() if v["overall"]["roc_auc"]]
    mean_shop_auc = float(np.mean(s_aucs)) if s_aucs else float(ov["roc_auc"])
    score = 0.5 * float(ov["roc_auc"]) + 0.5 * mean_shop_auc
    if latency_b1_ms is not None and latency_b1_ms > latency_budget_ms:
        score -= 0.05
    return score


def format_benchmark_markdown(benchmark_rows, champion_key, models_report):
    lines = [
        "## Bảng So sánh Toàn diện (Bài toán Phân loại Nhị phân: Session-prefix Purchase Intent)",
        "",
        "| Mô hình | VAL ROC-AUC | TEST ROC-AUC | TEST PR-AUC | TEST Log-Loss | TEST Brier | TEST Best F1 | Tham số | Độ trễ (b=1 / b=128) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows:
        tag = " **(Champion)**" if row["key"] == champion_key else ""
        lat = row["latency"]
        lat_str = f"{lat['b1_median_ms']:.2f} ms / {lat['b128_median_ms']:.2f} ms"
        if lat["runtime"] == "native":
            lat_str += " (native)"
        params_str = f"{row['n_parameters']:,}" if row["n_parameters"] else "—"
        lines.append(
            f"| `{row['key']}`{tag} | {row['val_roc_auc']:.4f} | {row['test_roc_auc']:.4f} | "
            f"{row['test_pr_auc']:.4f} | {row['test_logloss']:.4f} | {row['test_brier']:.4f} | "
            f"{row['test_best_f1']:.4f} | {params_str} | {lat_str} |"
        )
    lines.append("")

    # Ablation section
    lines.append("## Bảng Ablation (Đóng góp thành phần trên `TEST`)")
    lines.append("")
    lines.append("| Cấu hình | VAL ROC-AUC | TEST ROC-AUC | TEST PR-AUC | TEST Brier | TEST Log-Loss | `rees46` ROC-AUC | `synthetic` ROC-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, mdata in models_report.items():
        ev_v = mdata["eval"]["val"]["overall"]
        ev_t = mdata["eval"]["test"]
        ov_t = ev_t["overall"]
        r_auc = ev_t["by_shop"].get("rees46_cosmetics", {}).get("overall", {}).get("roc_auc") or 0.0
        s_auc = ev_t["by_shop"].get("synthetic", {}).get("overall", {}).get("roc_auc") or 0.0
        lines.append(
            f"| **{mdata['label']}** | {ev_v['roc_auc']:.4f} | {ov_t['roc_auc']:.4f} | "
            f"{ov_t['pr_auc']:.4f} | {ov_t['brier']:.4f} | {ov_t['logloss']:.4f} | "
            f"{r_auc:.4f} | {s_auc:.4f} |"
        )
    lines.append("")

    c_test = models_report[champion_key]["eval"]["test"]
    lines.append(f"### Chi tiết theo độ dài prefix `k` của Champion (`{champion_key}`) trên `TEST`")
    lines.append("")
    lines.append("| Độ dài prefix `k` | Số mẫu | Tỉ lệ mua | All-shop ROC-AUC | All-shop PR-AUC | `rees46` ROC-AUC | `rees46` PR-AUC | `synthetic` ROC-AUC | `synthetic` PR-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in BUCKET_NAMES:
        if b not in c_test["by_prefix_len"]:
            continue
        ob = c_test["by_prefix_len"][b]
        rb = c_test["by_shop"].get("rees46_cosmetics", {}).get("by_prefix_len", {}).get(b, {})
        sb = c_test["by_shop"].get("synthetic", {}).get("by_prefix_len", {}).get(b, {})
        r_auc = f"{rb['roc_auc']:.4f}" if rb.get("roc_auc") is not None else "—"
        r_pr = f"{rb['pr_auc']:.4f}" if rb.get("pr_auc") is not None else "—"
        s_auc = f"{sb['roc_auc']:.4f}" if sb.get("roc_auc") is not None else "—"
        s_pr = f"{sb['pr_auc']:.4f}" if sb.get("pr_auc") is not None else "—"
        lines.append(
            f"| `{b}` | {ob['samples']:,} | {ob['pos_rate'] * 100:.2f}% | "
            f"**{ob['roc_auc']:.4f}** | **{ob['pr_auc']:.4f}** | {r_auc} | {r_pr} | {s_auc} | {s_pr} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Unified Binary Classification Benchmark, Champion Improvement & ONNX Export")
    ap.add_argument("--prefix", default="data/prefix")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--report", default="data/prefix/_dl_ablation.json")
    ap.add_argument("--benchmark-report", default="data/prefix/_model_benchmark.json")
    ap.add_argument("--max-sessions-per-split", type=int, default=12000)
    ap.add_argument("--max-k-per-session", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--improved-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1536)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--improved-d-model", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(min(os.cpu_count() or 4, 4))
    spec = FT.load_spec(os.path.join(a.prefix, "feature_spec.json"))
    t0 = time.time()
    splits = load_sequence_splits(
        a.prefix, spec,
        max_sessions_per_split=a.max_sessions_per_split,
        max_k_per_session=a.max_k_per_session,
        seed=a.seed,
        include_tabular=True,
    )
    t_load = round(time.time() - t0, 2)
    print(f"Loaded unified binary-classification splits in {t_load}s: "
          f"train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}",
          flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    tmp_onnx_dir = os.path.join(a.out_dir, "_candidates")
    os.makedirs(tmp_onnx_dir, exist_ok=True)

    trained_models = {}
    models_report = {}
    benchmark_rows = []

    # ------------------------------------------------------------ Stage 1A: Classical Baselines (same splits!)
    tr_tab = splits["train"].tabular
    va_tab = splits["val"].tabular
    te_tab = splits["test"].tabular

    print("\n=== STAGE 1: BASELINE & CANDIDATE COMPARISON (BINARY CLASSIFICATION) ===", flush=True)
    t1 = time.time()
    markov = MarkovPurchaseBaseline(alpha=25.0, mix_generative=0.20).fit(tr_tab["markov"], tr_tab["y"])
    t_markov = round(time.time() - t1, 2)
    p_mk_val = markov.predict_proba(va_tab["markov"])
    p_mk_test = markov.predict_proba(te_tab["markov"])
    ev_mk_val = evaluate_predictions(va_tab["y"], p_mk_val, va_tab["k"], va_tab["shop_id"])
    ev_mk_test = evaluate_predictions(te_tab["y"], p_mk_test, te_tab["k"], te_tab["shop_id"], val_eval=ev_mk_val)
    mk_b1 = tuple(arr[:1] for arr in te_tab["markov"])
    mk_b128 = tuple(arr[:min(128, len(te_tab["y"]))] for arr in te_tab["markov"])
    lat_mk = benchmark_native_latency(lambda: markov.predict_proba(mk_b1), lambda: markov.predict_proba(mk_b128))
    benchmark_rows.append({
        "key": "markov", "family": "classical", "n_parameters": 0, "train_time_s": t_markov,
        "val_roc_auc": ev_mk_val["overall"]["roc_auc"],
        "test_roc_auc": ev_mk_test["overall"]["roc_auc"],
        "test_pr_auc": ev_mk_test["overall"]["pr_auc"],
        "test_logloss": ev_mk_test["overall"]["logloss"],
        "test_brier": ev_mk_test["overall"]["brier"],
        "test_best_f1": ev_mk_test["overall"]["best_f1"],
        "latency": lat_mk,
    })
    print(f"[markov] ({t_markov}s) VAL AUC={ev_mk_val['overall']['roc_auc']:.4f} | "
          f"TEST AUC={ev_mk_test['overall']['roc_auc']:.4f} PR-AUC={ev_mk_test['overall']['pr_auc']:.4f} "
          f"F1={ev_mk_test['overall']['best_f1']:.4f}", flush=True)

    t2 = time.time()
    gbm = GBMPurchaseBaseline(seed=a.seed).fit(tr_tab["X"], tr_tab["y"])
    t_gbm = round(time.time() - t2, 2)
    p_gb_val = gbm.predict_proba(va_tab["X"])[:, 1]
    p_gb_test = gbm.predict_proba(te_tab["X"])[:, 1]
    ev_gb_val = evaluate_predictions(va_tab["y"], p_gb_val, va_tab["k"], va_tab["shop_id"])
    ev_gb_test = evaluate_predictions(te_tab["y"], p_gb_test, te_tab["k"], te_tab["shop_id"], val_eval=ev_gb_val)
    x_b1 = te_tab["X"][:1]
    x_b128 = te_tab["X"][:min(128, len(te_tab["y"]))]
    lat_gb = benchmark_native_latency(lambda: gbm.predict_proba(x_b1), lambda: gbm.predict_proba(x_b128))
    benchmark_rows.append({
        "key": "histgbm", "family": "classical", "n_parameters": 0, "train_time_s": t_gbm,
        "val_roc_auc": ev_gb_val["overall"]["roc_auc"],
        "test_roc_auc": ev_gb_test["overall"]["roc_auc"],
        "test_pr_auc": ev_gb_test["overall"]["pr_auc"],
        "test_logloss": ev_gb_test["overall"]["logloss"],
        "test_brier": ev_gb_test["overall"]["brier"],
        "test_best_f1": ev_gb_test["overall"]["best_f1"],
        "latency": lat_gb,
    })
    print(f"[histgbm] ({t_gbm}s, trees={gbm.n_iter_}) VAL AUC={ev_gb_val['overall']['roc_auc']:.4f} | "
          f"TEST AUC={ev_gb_test['overall']['roc_auc']:.4f} PR-AUC={ev_gb_test['overall']['pr_auc']:.4f} "
          f"F1={ev_gb_test['overall']['best_f1']:.4f}", flush=True)

    classical_evals = {
        "markov": {"label": "Markov Chain (1st-order + backoff)", "train_time_s": t_markov,
                   "eval": {"val": ev_mk_val, "test": ev_mk_test}, "latency": lat_mk},
        "histgbm": {"label": "HistGradientBoosting (32 tabular features)", "train_time_s": t_gbm,
                    "n_iter": int(gbm.n_iter_), "eval": {"val": ev_gb_val, "test": ev_gb_test}, "latency": lat_gb},
    }

    # ------------------------------------------------------------ Stage 1B: 6 Deep Learning Baseline Candidates
    dl_labels = {
        "gru": "GRU (Sequence Classifier)",
        "hybrid": "Hybrid GRU + Wide & Deep (ONNX-native 44 wide features)",
        "tcn": "Causal TCN (Dilated Conv1d Residual)",
        "bilstm_attention": "Bi-LSTM + Additive Attention",
        "mixer": "MLP-Mixer (Fixed Window L=50)",
        "transformer": "Causal Transformer (Pre-LN Self-Attention)",
    }

    def run_dl_variant(key, label, cfg, d_model, epochs, lr=2.5e-3, label_smoothing=0.0,
                       swa_top2=False, measure_onnx=True):
        tm0 = time.time()
        mdl, info = train_model(
            spec, splits["train"], splits["val"],
            d_model=d_model, n_layers=1, n_heads=4, dropout=0.1,
            epochs=epochs, batch_size=a.batch_size, lr=lr,
            label_smoothing=label_smoothing, swa_top2=swa_top2, seed=a.seed, **cfg,
        )
        t_fit = round(time.time() - tm0, 2)
        _, p_val = predict_logits_and_probs(mdl, splits["val"], batch_size=4096)
        ev_val = evaluate_predictions(splits["val"].y, p_val, splits["val"].k, splits["val"].shop_id)
        _, p_test = predict_logits_and_probs(mdl, splits["test"], batch_size=4096)
        ev_test = evaluate_predictions(splits["test"].y, p_test, splits["test"].k, splits["test"].shop_id,
                                       val_eval=ev_val)
        n_params = int(sum(p.numel() for p in mdl.parameters()))

        parity = None
        latency = None
        if measure_onnx:
            cand_onnx = os.path.join(tmp_onnx_dir, f"{key}.onnx")
            parity = export_and_verify_onnx(mdl, spec, splits["test"], cand_onnx, n_verify=500)
            latency = benchmark_onnx_latency(cand_onnx, splits["test"])

        trained_models[key] = mdl
        models_report[key] = {
            "label": label,
            "config": {**cfg, "d_model": d_model, "epochs": epochs, "swa_top2": swa_top2},
            "n_parameters": n_params,
            "train_time_s": t_fit,
            "training_info": info,
            "onnx_parity": parity,
            "latency": latency,
            "eval": {"val": ev_val, "test": ev_test},
        }
        ov_v, ov_t = ev_val["overall"], ev_test["overall"]
        lat_msg = f", b1={latency['b1_median_ms']:.2f}ms" if latency else ""
        print(f"[{key}] ({t_fit}s, params={n_params:,}{lat_msg}) "
              f"VAL AUC={ov_v['roc_auc']:.4f} | TEST AUC={ov_t['roc_auc']:.4f} "
              f"PR-AUC={ov_t['pr_auc']:.4f} Brier={ov_t['brier']:.4f} F1={ov_t['best_f1']:.4f}", flush=True)
        return models_report[key]

    for arch in BASELINE_DL_ARCHS:
        rep = run_dl_variant(
            arch, dl_labels[arch],
            dict(arch=arch, use_cat_brand=True, use_num=True, use_wide=True,
                 use_attn=True, use_film=False, use_pos_weight=True, calibrate_on_val=True),
            d_model=a.d_model, epochs=a.epochs,
        )
        benchmark_rows.append({
            "key": arch, "family": "dl_baseline", "n_parameters": rep["n_parameters"],
            "train_time_s": rep["train_time_s"],
            "val_roc_auc": rep["eval"]["val"]["overall"]["roc_auc"],
            "test_roc_auc": rep["eval"]["test"]["overall"]["roc_auc"],
            "test_pr_auc": rep["eval"]["test"]["overall"]["pr_auc"],
            "test_logloss": rep["eval"]["test"]["overall"]["logloss"],
            "test_brier": rep["eval"]["test"]["overall"]["brier"],
            "test_best_f1": rep["eval"]["test"]["overall"]["best_f1"],
            "latency": rep["latency"],
        })

    best_baseline_key = max(
        BASELINE_DL_ARCHS,
        key=lambda k: _selection_score(models_report[k]["eval"]["val"], models_report[k]["latency"]["b1_median_ms"]),
    )
    print(f"\n>>> Best Stage-1 Baseline Selected for Improvement: {best_baseline_key} <<<", flush=True)

    # ------------------------------------------------------------ Stage 2: Develop Improved Champion + Ablations
    print("\n=== STAGE 2: IMPROVED CHAMPION (bilstm_attention_v2) & ABLATION STUDY ===", flush=True)

    rep_imp = run_dl_variant(
        "bilstm_attention_v2",
        "Bi-LSTM + Dual Attention + StepSkip + LogRatios + Top-2 SWA (Improved Champion)",
        dict(arch="bilstm_attention_v2", use_cat_brand=True, use_num=True, use_wide=True,
             use_attn=True, use_film=True, use_pos_weight=True, calibrate_on_val=True),
        d_model=a.improved_d_model, epochs=a.improved_epochs, swa_top2=True,
    )
    benchmark_rows.append({
        "key": "bilstm_attention_v2", "family": "dl_improved", "n_parameters": rep_imp["n_parameters"],
        "train_time_s": rep_imp["train_time_s"],
        "val_roc_auc": rep_imp["eval"]["val"]["overall"]["roc_auc"],
        "test_roc_auc": rep_imp["eval"]["test"]["overall"]["roc_auc"],
        "test_pr_auc": rep_imp["eval"]["test"]["overall"]["pr_auc"],
        "test_logloss": rep_imp["eval"]["test"]["overall"]["logloss"],
        "test_brier": rep_imp["eval"]["test"]["overall"]["brier"],
        "test_best_f1": rep_imp["eval"]["test"]["overall"]["best_f1"],
        "latency": rep_imp["latency"],
    })

    # Reuse Stage-1 gru & transformer as gru_full & transformer_full for compatibility with test_onnx_parity.py
    trained_models["gru_full"] = trained_models["gru"]
    models_report["gru_full"] = copy.deepcopy(models_report["gru"])
    models_report["gru_full"]["label"] = "GRU (Full: weighted loss + val calibration)"

    trained_models["transformer_full"] = trained_models["transformer"]
    models_report["transformer_full"] = copy.deepcopy(models_report["transformer"])
    models_report["transformer_full"]["label"] = "Transformer (Full: weighted loss + val calibration)"

    ablation_variants = [
        ("gru_unweighted", "GRU (Ablation: no class-weighted loss)",
         dict(arch="gru", use_cat_brand=True, use_num=True, use_wide=False,
              use_attn=False, use_film=False, use_pos_weight=False, calibrate_on_val=True),
         a.d_model, a.epochs, False, False),
        ("gru_uncalibrated", "GRU (Ablation: no val probability calibration)",
         dict(arch="gru", use_cat_brand=True, use_num=True, use_wide=False,
              use_attn=False, use_film=False, use_pos_weight=True, calibrate_on_val=False),
         a.d_model, a.epochs, False, False),
        ("gru_no_cat_brand", "GRU (Ablation: no category/brand embeddings)",
         dict(arch="gru", use_cat_brand=False, use_num=True, use_wide=False,
              use_attn=False, use_film=False, use_pos_weight=True, calibrate_on_val=True),
         a.d_model, a.epochs, False, False),
        ("gru_no_num", "GRU (Ablation: only categorical embeddings, no num/time)",
         dict(arch="gru", use_cat_brand=True, use_num=False, use_wide=False,
              use_attn=False, use_film=False, use_pos_weight=True, calibrate_on_val=True),
         a.d_model, a.epochs, False, False),
        ("bilstm_v2_no_swa", "BiLSTM v2 (Ablation: without Top-2 SWA averaging)",
         dict(arch="bilstm_attention_v2", use_cat_brand=True, use_num=True, use_wide=True,
              use_attn=True, use_film=True, use_pos_weight=True, calibrate_on_val=True),
         a.improved_d_model, a.improved_epochs, False, False),
        ("bilstm_v2_no_log_ratios", "BiLSTM v2 (Ablation: without 4 log-ratio features)",
         dict(arch="bilstm_attention_v2", use_cat_brand=True, use_num=True, use_wide=False,
              use_attn=True, use_film=True, use_pos_weight=True, calibrate_on_val=True),
         a.improved_d_model, a.improved_epochs, True, False),
        ("bilstm_v2_no_emb_skip", "BiLSTM v2 (Ablation: without StepEmbedding residual skip)",
         dict(arch="bilstm_attention_v2", use_cat_brand=True, use_num=True, use_wide=True,
              use_attn=True, use_film=False, use_pos_weight=True, calibrate_on_val=True),
         a.improved_d_model, a.improved_epochs, True, False),
    ]
    for key, label, cfg, dm, ep, swa_flag, onnx_flag in ablation_variants:
        run_dl_variant(key, label, cfg, d_model=dm, epochs=ep, swa_top2=swa_flag, measure_onnx=onnx_flag)

    # ------------------------------------------------------------ Stage 3: Final Champion Selection & ONNX Export
    champion_candidates = list(BASELINE_DL_ARCHS) + ["bilstm_attention_v2"]
    champion_key = max(
        champion_candidates,
        key=lambda k: _selection_score(models_report[k]["eval"]["val"], models_report[k]["latency"]["b1_median_ms"]),
    )
    champion_model = trained_models[champion_key]
    challenger_key = "transformer_full"

    onnx_path = os.path.join(a.out_dir, "intent_v1.onnx")
    parity_info = export_and_verify_onnx(champion_model, spec, splits["test"], onnx_path, n_verify=1000)
    champ_latency = benchmark_onnx_latency(onnx_path, splits["test"])

    challenger_onnx = os.path.join(a.out_dir, f"intent_{challenger_key}.onnx")
    challenger_parity = export_and_verify_onnx(trained_models[challenger_key], spec, splits["test"],
                                               challenger_onnx, n_verify=1000)

    spec_out = os.path.join(a.out_dir, "feature_spec.json")
    shutil.copyfile(os.path.join(a.prefix, "feature_spec.json"), spec_out)

    # Clean up temporary candidate ONNX files
    shutil.rmtree(tmp_onnx_dir, ignore_errors=True)

    champ_rep = models_report[champion_key]
    model_card = {
        "model_ver": "intent_v1",
        "task": "binary_classification_session_prefix_purchase_intent",
        "architecture": champ_rep["config"]["arch"],
        "champion_variant": champion_key,
        "best_stage1_baseline": best_baseline_key,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": spec["feature_version"],
        "feature_spec_file": "feature_spec.json",
        "onnx_file": "intent_v1.onnx",
        "onnx_size_bytes": parity_info["onnx_size_bytes"],
        "n_parameters": champ_rep["n_parameters"],
        "onnx_cpu_latency_ms": champ_latency,
        "max_len": spec["max_len"],
        "inputs": {
            "x_cat": {"dtype": "int64", "shape": ["batch", spec["max_len"], FT.N_CAT], "fields": list(FT.CAT_FIELDS),
                      "padding": "left (PAD=0, NULL=1, OOV=2, vocab>=3)"},
            "x_num": {"dtype": "float32", "shape": ["batch", spec["max_len"], FT.N_NUM],
                      "fields": list(FT.NUM_FEATURES), "padding": "left (0.0)"},
            "length": {"dtype": "int64", "shape": ["batch"], "description": "number of valid steps in window (1..50)"},
            "shop": {"dtype": "int64", "shape": ["batch"], "mapping": spec["shops"]},
        },
        "outputs": {
            "prob": {"dtype": "float32", "shape": ["batch"],
                     "description": "calibrated probability P(session ends with purchase | prefix)"}
        },
        "hyperparameters": {
            "d_model": champ_rep["config"]["d_model"],
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.1,
            "epochs": champ_rep["config"]["epochs"],
            "batch_size": a.batch_size,
            "pos_weight": champ_rep["training_info"]["pos_weight"],
            "calibration_params_by_shop_idx": champ_rep["training_info"]["calibration_params_by_shop_idx"],
            "val_optimal_threshold": champ_rep["eval"]["val"]["overall"]["best_threshold"],
            "val_optimal_threshold_by_shop": {
                s: v["overall"]["best_threshold"] for s, v in champ_rep["eval"]["val"]["by_shop"].items()
            },
        },
        "offline_metrics": {
            "val": champ_rep["eval"]["val"],
            "test": champ_rep["eval"]["test"],
        },
        "onnx_parity_check": parity_info,
        "challenger": {
            "variant": challenger_key,
            "architecture": models_report[challenger_key]["config"]["arch"],
            "onnx_file": os.path.basename(challenger_onnx),
            "val_overall": models_report[challenger_key]["eval"]["val"]["overall"],
            "test_overall": models_report[challenger_key]["eval"]["test"]["overall"],
            "onnx_parity_check": challenger_parity,
        },
    }
    card_path = os.path.join(a.out_dir, "model_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    full_report = {
        "generated_utc": model_card["created_utc"],
        "task": "binary_classification_session_prefix_purchase_intent",
        "load_time_s": t_load,
        "sample_counts": {sp: len(splits[sp]) for sp in ("train", "val", "test")},
        "best_stage1_baseline": best_baseline_key,
        "champion": champion_key,
        "onnx_parity": parity_info,
        "classical_baselines": classical_evals,
        "models": models_report,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    benchmark_rows_sorted = sorted(benchmark_rows, key=lambda r: r["test_roc_auc"], reverse=True)
    bench_report = {
        "generated_utc": model_card["created_utc"],
        "task": "binary_classification_session_prefix_purchase_intent",
        "sample_counts": full_report["sample_counts"],
        "best_stage1_baseline": best_baseline_key,
        "champion": champion_key,
        "comparison_table": benchmark_rows_sorted,
        "classical_baselines": classical_evals,
    }
    with open(a.benchmark_report, "w", encoding="utf-8") as f:
        json.dump(bench_report, f, indent=2, ensure_ascii=False)

    print("\n" + format_benchmark_markdown(benchmark_rows_sorted, champion_key, models_report))
    print(f"\nExported champion ({champion_key}) -> {onnx_path} "
          f"(max_abs_diff={parity_info['max_abs_diff']:.2e} on {parity_info['samples_verified']} samples, "
          f"b=1 median={champ_latency['b1_median_ms']:.2f}ms) and {card_path}")


if __name__ == "__main__":
    main()
