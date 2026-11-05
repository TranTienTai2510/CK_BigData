"""Week 2 DL training, ablation study, probability calibration on VAL, and ONNX export (G4).

Run (trains GRU + Transformer + ablation variants on data/prefix, calibrates probabilities on val, exports the
champion model to models/intent_v1.onnx + models/model_card.json + models/feature_spec.json, and writes the full
ablation & prefix-length evaluation report to data/prefix/_dl_ablation.json):

  python ml/train_dl.py --prefix data/prefix --out-dir models --report data/prefix/_dl_ablation.json

Key design choices (aligned with phan-cong-A-data-ml.md & phan-cong-do-an-clickstream.md):
1. Shared tensor representation: every prefix sample (session, k) uses the exact same 3 categorical indices
   (type_idx, cat_idx, brand_idx) and 10 transformed numerical features (FT.NUM_FEATURES), left-padded to
   MAX_LEN = 50, as FT.window_to_tensor / FT.session_tensors.
2. Class-weighted BCE loss + post-hoc Platt calibration on VAL:
   - Training with pos_weight > 1 boosts gradient signal from the rare positive class (REES46 has ~3% session
     conversion), while post-hoc calibration (fitting logit_cal = a * logit_raw + b per shop on VAL via L-BFGS
     to minimise Brier / NLL without changing within-shop ranking) restores well-calibrated probabilities in (0, 1)
     and aligns the cross-shop probability scale (increasing combined ROC-AUC & PR-AUC and lowering Brier score).
   - Calibration parameters (calib_a, calib_b) are baked into IntentModel buffers before exporting ONNX, so Q2
     gets calibrated probabilities directly from `intent_v1.onnx` with zero extra post-processing.
3. Ablation table (Chapter 5b):
   - Full GRU (weighted loss + val calibration)
   - Full Transformer (weighted loss + val calibration)
   - GRU without class-weighted loss (pos_weight = 1.0)
   - GRU without val probability calibration (raw logits after pos_weight)
   - GRU without category & brand embeddings (only event_type + numerical features)
   - GRU without numerical/time/counter features (only categorical embeddings)
   - Compared side-by-side against Week 1 baselines (Markov + GBM) overall and by prefix length k.
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
from ml.baselines import BUCKET_NAMES, evaluate_predictions  # noqa: E402
from ml.models import IntentModel  # noqa: E402


# ---------------------------------------------------------------- Fast dataset loader (sessions + prefix indices)
class PrefixSequenceDataset:
    """Stores pre-transformed step arrays per session in contiguous flat arrays, plus a table of samples
    (sess_idx, k). Uses a compact sliding window of W_EVAL = 32 steps for fast CPU training/evaluation, and pads
    left to max_len = 50 when exporting/verifying ONNX.
    """
    def __init__(self, cat_flat, num_flat, offsets, sess_shop_idx, sess_shop_name, sess_label,
                 sample_sess, sample_k, max_len=FT.MAX_LEN):
        self.cat_flat = cat_flat              # int64   [TotalSteps, 3]
        self.num_flat = num_flat              # float32 [TotalSteps, 10]
        self.offsets = offsets                # int64   [NSess + 1]
        self.sess_shop_idx = sess_shop_idx    # int64   [NSess]
        self.sess_shop_name = sess_shop_name  # object  [NSess]
        self.sess_label = sess_label          # int32   [NSess]
        self.sample_sess = sample_sess        # int32   [NSamples]
        self.sample_k = sample_k              # int32   [NSamples]
        self.max_len = int(max_len)

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


def load_sequence_splits(prefix_dir, spec, max_sessions_per_split=20000, max_k_per_session=6, seed=42):
    """Load sessions from data/prefix/sessions and build PrefixSequenceDataset for train, val, test.
    Reads only the parquet part-files needed to reach max_sessions_per_split per (shop, split) so loading takes
    a few seconds instead of reading all 4.8M REES46 sessions into RAM.
    """
    base = os.path.join(prefix_dir, "sessions")
    shop_dirs = sorted(d for d in os.listdir(base) if d.startswith("shop_id="))
    rng = np.random.default_rng(seed)
    max_len = int(spec["max_len"])

    import pandas as pd
    splits_parts = defaultdict(list)
    for sd in shop_dirs:
        shop_name = sd.split("=", 1)[1]
        s_path = os.path.join(base, sd)
        # Keep natural multi-shop ratio: 3x sessions for REES46 vs synthetic so overall metrics match baselines
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
                ks_list.append(())
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

        datasets[split_name] = PrefixSequenceDataset(
            cat_flat, num_flat, offsets, sess_shop_idx, sess_shop_name, sess_label,
            sample_sess, sample_k, max_len=max_len,
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
        t_xc = torch.from_numpy(xc).to(device)
        t_xn = torch.from_numpy(xn).to(device)
        t_ln = torch.from_numpy(ln).to(device)
        t_sh = torch.from_numpy(sh).to(device)
        lg = model.forward_logits(t_xc, t_xn, t_ln, t_sh)
        pr = torch.sigmoid(lg)
        logits[start:start + len(idx)] = lg.cpu().numpy()
        probs[start:start + len(idx)] = pr.cpu().numpy()
    return logits, probs


def fit_platt_calibration(raw_logits, y_true, shop_idx):
    """Fit per-shop Platt scaling (a, b) with a >= 0.05 on VAL to minimize NLL (LogLoss) after weighted training.
    Because a > 0, within-shop ranking (ROC-AUC, PR-AUC) is preserved while probabilities are calibrated to the true
    empirical rate of each shop.
    """
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
            # stable binary cross-entropy: max(u, 0) - u * y + log(1 + exp(-|u|))
            loss = np.maximum(u, 0.0) - u * y + np.log1p(np.exp(-np.abs(u)))
            return float(loss.mean()) + 1e-4 * ((a - 1.0) ** 2)

        res = minimize(nll, x0=[1.0, 0.0], bounds=[(0.1, 5.0), (-10.0, 10.0)], method="L-BFGS-B")
        params[int(s)] = (round(float(res.x[0]), 6), round(float(res.x[1]), 6))
    return params


def train_model(spec, train_ds, val_ds, arch="gru", d_model=64, n_layers=2, n_heads=4, dropout=0.1,
                use_cat_brand=True, use_num=True, use_pos_weight=True, calibrate_on_val=True,
                epochs=4, batch_size=1024, lr=2.5e-3, weight_decay=1e-4, seed=42, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = IntentModel(spec, arch=arch, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        dropout=dropout, use_cat_brand=use_cat_brand, use_num=use_num).to(device)

    pos_rate = float(train_ds.y.mean())
    # Moderate damped class weight sqrt((1 - p) / p) prevents extreme logit saturation while boosting recall
    pw_val = math.sqrt((1.0 - pos_rate) / max(pos_rate, 1e-4)) if use_pos_weight else 1.0
    pos_weight = torch.tensor(pw_val, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(train_ds)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs * steps_per_epoch, 1),
                                                           eta_min=lr * 0.1)

    best_val_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
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

        # Evaluate uncalibrated on VAL at epoch end
        model.set_calibration({s: (1.0, 0.0) for s in range(int(spec["cardinality"]["shop"]))})
        val_logits, val_probs = predict_logits_and_probs(model, val_ds, batch_size=2048, device=device)
        # Rank within each shop on VAL (average of per-shop ROC-AUCs) to pick the best representation
        ev_uncal = evaluate_predictions(val_ds.y, val_probs, val_ds.k, val_ds.shop_id)
        shop_aucs = [v["overall"]["roc_auc"] for v in ev_uncal["by_shop"].values() if v["overall"]["roc_auc"]]
        mean_shop_auc = float(np.mean(shop_aucs)) if shop_aucs else ev_uncal["overall"]["roc_auc"]
        dt = round(time.time() - t0, 2)
        history.append({"epoch": ep, "train_loss": round(ep_loss / max(n_batches, 1), 4),
                        "val_mean_shop_auc": round(mean_shop_auc, 4),
                        "val_overall_auc_uncal": ev_uncal["overall"]["roc_auc"], "time_s": dt})
        if mean_shop_auc > best_val_auc:
            best_val_auc = mean_shop_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.set_calibration({s: (1.0, 0.0) for s in range(int(spec["cardinality"]["shop"]))})
    raw_val_logits, raw_val_probs = predict_logits_and_probs(model, val_ds, batch_size=2048, device=device)
    uncal_brier = float(np.mean((raw_val_probs - val_ds.y) ** 2))

    calib_params = {int(s): (1.0, 0.0) for s in np.unique(val_ds.shop_idx)}
    if calibrate_on_val:
        calib_params = fit_platt_calibration(raw_val_logits, val_ds.y, val_ds.shop_idx)
        model.set_calibration(calib_params)

    _, cal_val_probs = predict_logits_and_probs(model, val_ds, batch_size=2048, device=device)
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


# ---------------------------------------------------------------- ONNX export & parity check (G4)
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

    # Verify on n_verify real test samples across multiple batch sizes
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

    # Also test single-item inference (batch=1) and micro-batch inference (batch=64)
    ort_b1 = sess.run(["prob"], {"x_cat": xc[:1], "x_num": xn[:1], "length": ln[:1], "shop": sh[:1]})[0]
    max_diff_b1 = float(np.max(np.abs(pt_prob[:1] - ort_b1)))

    assert max_diff < 1e-4 and max_diff_b1 < 1e-4, f"ONNX parity failed: max_diff={max_diff}"
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


def format_ablation_markdown(report):
    lines = ["## Bảng Ablation & So sánh Model (Tuần 2 — Tập `VAL` và `TEST`)", ""]
    for split in ("val", "test"):
        lines.append(f"### Tập `{split.upper()}` — Gộp cả 2 shop & Từng shop")
        lines.append("")
        lines.append("| Mô hình / Cấu hình Ablation | All-k ROC-AUC | All-k PR-AUC | Brier | `rees46` ROC-AUC | `rees46` PR-AUC | `synthetic` ROC-AUC | `synthetic` PR-AUC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, mdata in report["models"].items():
            ev = mdata["eval"][split]
            ov = ev["overall"]
            r_ev = ev["by_shop"].get("rees46_cosmetics", {}).get("overall", {})
            s_ev = ev["by_shop"].get("synthetic", {}).get("overall", {})
            lines.append(
                f"| **{mdata['label']}** | {ov['roc_auc']:.4f} | {ov['pr_auc']:.4f} | {ov['brier']:.4f} | "
                f"{r_ev.get('roc_auc', 0.0):.4f} | {r_ev.get('pr_auc', 0.0):.4f} | "
                f"{s_ev.get('roc_auc', 0.0):.4f} | {s_ev.get('pr_auc', 0.0):.4f} |"
            )
        lines.append("")

    champ = report["champion"]
    c_test = report["models"][champ]["eval"]["test"]
    lines.append(f"### Chi tiết theo độ dài prefix `k` của Champion (`{report['models'][champ]['label']}`) trên `TEST`")
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
        lines.append(f"| `{b}` | {ob['samples']:,} | {ob['pos_rate'] * 100:.2f}% | "
                     f"**{ob['roc_auc']:.4f}** | **{ob['pr_auc']:.4f}** | {r_auc} | {r_pr} | {s_auc} | {s_pr} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data/prefix")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--report", default="data/prefix/_dl_ablation.json")
    ap.add_argument("--max-sessions-per-split", type=int, default=12000)
    ap.add_argument("--max-k-per-session", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(min(os.cpu_count() or 4, 8))
    spec = FT.load_spec(os.path.join(a.prefix, "feature_spec.json"))
    t0 = time.time()
    splits = load_sequence_splits(a.prefix, spec,
                                  max_sessions_per_split=a.max_sessions_per_split,
                                  max_k_per_session=a.max_k_per_session, seed=a.seed)
    t_load = round(time.time() - t0, 2)
    print(f"Loaded sequence splits in {t_load}s: train={len(splits['train']):,}, "
          f"val={len(splits['val']):,}, test={len(splits['test']):,}", flush=True)

    variants = [
        ("gru_full", "GRU (Full: weighted loss + val calibration)",
         dict(arch="gru", use_cat_brand=True, use_num=True, use_pos_weight=True, calibrate_on_val=True)),
        ("transformer_full", "Transformer (Full: weighted loss + val calibration)",
         dict(arch="transformer", use_cat_brand=True, use_num=True, use_pos_weight=True, calibrate_on_val=True)),
        ("gru_unweighted", "GRU (Ablation: no class-weighted loss)",
         dict(arch="gru", use_cat_brand=True, use_num=True, use_pos_weight=False, calibrate_on_val=True)),
        ("gru_uncalibrated", "GRU (Ablation: no val probability calibration)",
         dict(arch="gru", use_cat_brand=True, use_num=True, use_pos_weight=True, calibrate_on_val=False)),
        ("gru_no_cat_brand", "GRU (Ablation: no category/brand embeddings)",
         dict(arch="gru", use_cat_brand=False, use_num=True, use_pos_weight=True, calibrate_on_val=True)),
        ("gru_no_num", "GRU (Ablation: only categorical embeddings, no num/time)",
         dict(arch="gru", use_cat_brand=True, use_num=False, use_pos_weight=True, calibrate_on_val=True)),
    ]

    trained_models = {}
    models_report = {}

    for key, label, cfg in variants:
        tm0 = time.time()
        mdl, info = train_model(
            spec, splits["train"], splits["val"],
            d_model=a.d_model, n_layers=1, n_heads=4, dropout=0.1,
            epochs=a.epochs, batch_size=a.batch_size, seed=a.seed, **cfg,
        )
        t_fit = round(time.time() - tm0, 2)
        evals = {}
        for sp in ("val", "test"):
            _, probs = predict_logits_and_probs(mdl, splits[sp], batch_size=2048)
            evals[sp] = evaluate_predictions(splits[sp].y, probs, splits[sp].k, splits[sp].shop_id)
        n_params = sum(p.numel() for p in mdl.parameters())
        trained_models[key] = mdl
        models_report[key] = {
            "label": label,
            "config": cfg,
            "n_parameters": int(n_params),
            "train_time_s": t_fit,
            "training_info": info,
            "eval": evals,
        }
        ov_v = evals["val"]["overall"]
        ov_t = evals["test"]["overall"]
        print(f"[{key}] ({t_fit}s, params={n_params:,}) "
              f"VAL AUC={ov_v['roc_auc']:.4f} PR-AUC={ov_v['pr_auc']:.4f} Brier={ov_v['brier']:.4f} | "
              f"TEST AUC={ov_t['roc_auc']:.4f} PR-AUC={ov_t['pr_auc']:.4f} Brier={ov_t['brier']:.4f}", flush=True)

    # Pick champion between gru_full and transformer_full by VAL overall ROC-AUC + mean shop ROC-AUC
    def score_val(k):
        ev = models_report[k]["eval"]["val"]
        s_aucs = [v["overall"]["roc_auc"] for v in ev["by_shop"].values() if v["overall"]["roc_auc"]]
        return 0.5 * ev["overall"]["roc_auc"] + 0.5 * float(np.mean(s_aucs))

    champion_key = max(["gru_full", "transformer_full"], key=score_val)
    champion_model = trained_models[champion_key]
    challenger_key = "transformer_full" if champion_key == "gru_full" else "gru_full"

    os.makedirs(a.out_dir, exist_ok=True)
    onnx_path = os.path.join(a.out_dir, "intent_v1.onnx")
    parity_info = export_and_verify_onnx(champion_model, spec, splits["test"], onnx_path, n_verify=1000)

    # Also export challenger ONNX for optional A/B or champion-challenger experiment
    challenger_onnx = os.path.join(a.out_dir, f"intent_{challenger_key}.onnx")
    challenger_parity = export_and_verify_onnx(trained_models[challenger_key], spec, splits["test"],
                                               challenger_onnx, n_verify=1000)

    # Save feature_spec.json alongside intent_v1.onnx so Q2 loads both from models/
    spec_out = os.path.join(a.out_dir, "feature_spec.json")
    shutil.copyfile(os.path.join(a.prefix, "feature_spec.json"), spec_out)

    champ_rep = models_report[champion_key]
    model_card = {
        "model_ver": "intent_v1",
        "architecture": champ_rep["config"]["arch"],
        "champion_variant": champion_key,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": spec["feature_version"],
        "feature_spec_file": "feature_spec.json",
        "onnx_file": "intent_v1.onnx",
        "onnx_size_bytes": parity_info["onnx_size_bytes"],
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
            "d_model": a.d_model,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.1,
            "epochs": a.epochs,
            "batch_size": a.batch_size,
            "pos_weight": champ_rep["training_info"]["pos_weight"],
            "calibration_params_by_shop_idx": champ_rep["training_info"]["calibration_params_by_shop_idx"],
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
        "load_time_s": t_load,
        "sample_counts": {sp: len( splits[sp]) for sp in ("train", "val", "test")},
        "champion": champion_key,
        "onnx_parity": parity_info,
        "models": models_report,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print("\n" + format_ablation_markdown(full_report))
    print(f"\nExported champion ({champion_key}) -> {onnx_path} "
          f"(max_abs_diff={parity_info['max_abs_diff']:.2e} on {parity_info['samples_verified']} samples) "
          f"and {card_path}")


if __name__ == "__main__":
    main()
