"""Train, evaluate, time and optionally export the session-prefix model zoo.

Run:
  python ml_test/benchmark_models.py --quick
  python ml_test/benchmark_models.py --export-best
"""
import argparse
import csv
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score

from ml import features as FT
from ml.baselines import BUCKET_NAMES
from ml_test.models_zoo import MODEL_SPECS, build_model, window_to_flat_features


class BenchmarkDataset:
    """A deterministic view that builds only requested sequence batches in memory."""
    def __init__(self, base, max_samples=None, seed=42):
        indices = np.arange(len(base), dtype=np.int64)
        if max_samples and len(indices) > max_samples:
            indices = np.sort(np.random.default_rng(seed).choice(indices, max_samples, replace=False))
        self.base = base
        self.indices = indices
        self.sample_sess = base.sample_sess[indices]
        self.sample_k = base.sample_k[indices]
        self.y = base.y[indices].astype(np.float32)
        self.k = base.k[indices]
        self.shop_idx = base.shop_idx[indices]
        self.shop_id = base.shop_id[indices]

    def __len__(self):
        return len(self.indices)

    def get_batch(self, indices):
        selected = self.indices[np.asarray(indices, dtype=np.int64)]
        sessions = self.base.sample_sess[selected]
        ks = self.base.sample_k[selected]
        batch = len(selected)
        width = self.base.max_len
        x_cat = np.zeros((batch, width, FT.N_CAT), dtype=np.int64)
        x_num = np.zeros((batch, width, FT.N_NUM), dtype=np.float32)
        length = np.empty(batch, dtype=np.int64)
        for i, (session, k_value) in enumerate(zip(sessions, ks)):
            k = int(k_value)
            end = int(self.base.offsets[session]) + k
            start = int(self.base.offsets[session])
            take = min(k, width)
            source_start = end - take
            x_cat[i, width - take:] = self.base.cat_flat[source_start:end]
            x_num[i, width - take:] = self.base.num_flat[source_start:end]
            length[i] = take
        return x_cat, x_num, length, self.shop_idx[np.asarray(indices)], self.y[np.asarray(indices)]


def _batches(dataset, batch_size):
    for start in range(0, len(dataset), batch_size):
        yield np.arange(start, min(start + batch_size, len(dataset)), dtype=np.int64)


@torch.no_grad()
def predict_logits(model, dataset, batch_size, device):
    model.eval()
    chunks = []
    for indices in _batches(dataset, batch_size):
        xc, xn, ln, sh, _ = dataset.get_batch(indices)
        logits = model.forward_logits(
            torch.from_numpy(xc).to(device), torch.from_numpy(xn).to(device),
            torch.from_numpy(ln).to(device), torch.from_numpy(sh).to(device),
        )
        chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0, dtype=np.float64)


def _probabilities_from_logits(logits, shops, params):
    calibrated = np.asarray(logits, dtype=np.float64).copy()
    for shop_idx, (scale, bias) in params.items():
        mask = np.asarray(shops) == int(shop_idx)
        calibrated[mask] = float(scale) * calibrated[mask] + float(bias)
    calibrated = np.clip(calibrated, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-calibrated))


def _best_f1_threshold(y_true, probabilities):
    if len(y_true) == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, np.max(f1)))
    return float(thresholds[best[-1]])


def _metrics(y_true, probabilities, threshold):
    y = np.asarray(y_true, dtype=np.int32)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    has_both_classes = bool(len(y) and np.unique(y).size == 2)
    return {
        "samples": int(len(y)),
        "positives": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else None,
        "roc_auc": float(roc_auc_score(y, p)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y, p)) if len(y) and y.sum() else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "best_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
    }


def evaluate_split(dataset, probabilities, threshold):
    result = {
        "overall": _metrics(dataset.y, probabilities, threshold),
        "by_shop": {},
        "by_prefix_len": {},
    }
    for shop in sorted(set(dataset.shop_id.tolist())):
        mask = dataset.shop_id == shop
        result["by_shop"][str(shop)] = _metrics(dataset.y[mask], probabilities[mask], threshold)
    bucket = np.asarray([_bucket(int(k)) for k in dataset.k], dtype=object)
    for name in BUCKET_NAMES:
        mask = bucket == name
        if mask.any():
            result["by_prefix_len"][name] = _metrics(dataset.y[mask], probabilities[mask], threshold)
    return result


def _bucket(k):
    for name in BUCKET_NAMES:
        if name == ">50" and k > 50:
            return name
        if name == "21-50" and 21 <= k <= 50:
            return name
        if name == "11-20" and 11 <= k <= 20:
            return name
        if name == "6-10" and 6 <= k <= 10:
            return name
        if name == "3-5" and 3 <= k <= 5:
            return name
        if name == str(k) and k in (1, 2):
            return name
    return BUCKET_NAMES[-1]


def _flat_data(dataset, batch_size):
    features = np.empty((len(dataset), 31), dtype=np.float32)
    for indices in _batches(dataset, batch_size):
        xc, xn, ln, _, _ = dataset.get_batch(indices)
        features[indices] = window_to_flat_features(xc, xn, ln)
    return features


def _fit_gbm(train_ds, val_ds, test_ds, seed, quick, pos_weight, batch_size):
    x_train = _flat_data(train_ds, batch_size)
    x_val = _flat_data(val_ds, batch_size)
    x_test = _flat_data(test_ds, batch_size)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=25 if quick else 250, max_leaf_nodes=31 if quick else 63,
        min_samples_leaf=max(10, min(100, len(train_ds) // 20)), l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=8 if quick else 15,
        scoring="roc_auc", random_state=seed,
    )
    sample_weight = np.where(train_ds.y > 0, pos_weight, 1.0)
    started = time.perf_counter()
    model.fit(x_train, train_ds.y.astype(np.int32), sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - started

    def logits(x):
        p = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1.0 - 1e-7)
        return np.log(p / (1.0 - p))

    raw_val = logits(x_val)
    from ml.train_dl import fit_platt_calibration
    calibration = fit_platt_calibration(raw_val, val_ds.y, val_ds.shop_idx)
    threshold = _best_f1_threshold(
        val_ds.y, _probabilities_from_logits(raw_val, val_ds.shop_idx, calibration)
    )
    evals = {}
    for split_name, ds, x in (("val", val_ds, x_val), ("test", test_ds, x_test)):
        prob = _probabilities_from_logits(logits(x), ds.shop_idx, calibration)
        evals[split_name] = evaluate_split(ds, prob, threshold)
    serialized_size = len(__import__("pickle").dumps(model, protocol=4))
    return {
        "label": "HistGradientBoosting (31 tabular features)",
        "n_parameters": None,
        "training_info": {"calibrated": True, "calibration_params_by_shop_idx": calibration,
                          "n_iter": int(model.n_iter_)},
        "threshold": threshold,
        "eval": evals,
        "system": {
            "onnx_size_bytes": None, "onnx_size_kb": None,
            "onnx_latency_ms_batch_1": None, "onnx_latency_ms_batch_128": None,
            "native_model_size_bytes": serialized_size,
            "native_latency_ms_batch_1": _time_predict(model, x_test[:1], 100),
            "native_latency_ms_batch_128": _time_predict(model, x_test[:128], 50),
            "train_seconds": fit_seconds, "seconds_per_epoch": None,
        },
    }


def _time_predict(model, features, repeats):
    if len(features) == 0:
        return None
    x = np.repeat(features, repeats=128 // len(features) + 1, axis=0)[:128]
    batch_sizes = (1, 128)
    out = []
    for size in batch_sizes:
        batch = x[:size]
        for _ in range(5):
            model.predict_proba(batch)
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict_proba(batch)
        out.append((time.perf_counter() - start) * 1000.0 / repeats)
    return out[0] if len(features) == 1 else out[1]


def _train_neural(name, spec, train_ds, val_ds, test_ds, seed, epochs, batch_size,
                  d_model, device, pos_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name, spec, d_model=d_model).to(device)
    model.set_calibration({int(s): (1.0, 0.0) for s in np.unique(train_ds.shop_idx)})
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    epoch_seconds = []
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_ds))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xc, xn, ln, sh, y = train_ds.get_batch(indices)
            logits = model.forward_logits(
                torch.from_numpy(xc).to(device), torch.from_numpy(xn).to(device),
                torch.from_numpy(ln).to(device), torch.from_numpy(sh).to(device),
            )
            target = torch.from_numpy(y).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epoch_seconds.append(time.perf_counter() - started)
        val_logits = predict_logits(model, val_ds, batch_size, device)
        if len(np.unique(val_ds.y)) > 1:
            score = float(roc_auc_score(val_ds.y, val_logits))
        else:
            score = -float(np.mean((val_logits - val_ds.y) ** 2))
        if score > best_auc:
            best_auc = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.set_calibration({int(s): (1.0, 0.0) for s in np.unique(train_ds.shop_idx)})
    raw_val = predict_logits(model, val_ds, batch_size, device)
    from ml.train_dl import fit_platt_calibration
    calibration = fit_platt_calibration(raw_val, val_ds.y, val_ds.shop_idx)
    model.set_calibration(calibration)
    val_probs = _probabilities_from_logits(raw_val, val_ds.shop_idx, calibration)
    threshold = _best_f1_threshold(val_ds.y, val_probs)
    evals = {}
    for split_name, ds in (("val", val_ds), ("test", test_ds)):
        logits = predict_logits(model, ds, batch_size, device)
        probs = _probabilities_from_logits(logits, ds.shop_idx, calibration)
        evals[split_name] = evaluate_split(ds, probs, threshold)
    total = float(sum(epoch_seconds))
    return model, {
        "label": MODEL_SPECS[name],
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "training_info": {
            "pos_weight": float(pos_weight), "calibrated": True,
            "calibration_params_by_shop_idx": {str(k): [float(v[0]), float(v[1])]
                                                for k, v in calibration.items()},
            "epochs_seconds": [float(x) for x in epoch_seconds],
        },
        "threshold": threshold,
        "eval": evals,
        "system": {
            "onnx_size_bytes": None, "onnx_size_kb": None,
            "onnx_latency_ms_batch_1": None, "onnx_latency_ms_batch_128": None,
            "native_model_size_bytes": None, "native_latency_ms_batch_1": None,
            "native_latency_ms_batch_128": None,
            "train_seconds": total, "seconds_per_epoch": total / max(len(epoch_seconds), 1),
        },
    }


def _export_onnx(model, onnx_path, spec, test_ds, verify_samples=1000):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX export requires both 'onnx' and 'onnxruntime' packages") from exc
    model_cpu = copy.deepcopy(model).cpu().eval()
    sample = test_ds.get_batch(np.asarray([0]))
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model_cpu,
                tuple(torch.from_numpy(x) for x in sample[:4]),
                onnx_path,
                input_names=["x_cat", "x_num", "length", "shop"],
                output_names=["prob"],
                dynamic_axes={"x_cat": {0: "batch"}, "x_num": {0: "batch"},
                              "length": {0: "batch"}, "shop": {0: "batch"}, "prob": {0: "batch"}},
                opset_version=18, do_constant_folding=True, dynamo=False,
            )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    if [value.name for value in graph.graph.input] != ["x_cat", "x_num", "length", "shop"]:
        raise RuntimeError("exported ONNX input names do not match the streaming contract")
    if [value.name for value in graph.graph.output] != ["prob"]:
        raise RuntimeError("exported ONNX output name does not match the streaming contract")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=session_options,
                                   providers=["CPUExecutionProvider"])
    n = min(verify_samples, len(test_ds))
    if n < 1:
        raise ValueError("test split has no samples for ONNX parity verification")
    indices = np.linspace(0, len(test_ds) - 1, num=n, dtype=np.int64)
    xc, xn, ln, sh, _ = test_ds.get_batch(indices)
    with torch.no_grad():
        pt_prob = model_cpu(torch.from_numpy(xc), torch.from_numpy(xn),
                            torch.from_numpy(ln), torch.from_numpy(sh)).numpy()
    ort_prob = session.run(["prob"], {"x_cat": xc, "x_num": xn, "length": ln, "shop": sh})[0]
    max_diff = float(np.max(np.abs(pt_prob - ort_prob)))
    if max_diff >= 1e-4:
        raise AssertionError(f"ONNX parity failed for {os.path.basename(onnx_path)}: {max_diff:.3e}")
    latencies = {}
    for batch_size, repeats in ((1, 100), (128, 40)):
        ids = np.resize(indices, batch_size)
        bx, bn, bl, bs, _ = test_ds.get_batch(ids)
        feed = {"x_cat": bx, "x_num": bn, "length": bl, "shop": bs}
        for _ in range(10):
            session.run(["prob"], feed)
        started = time.perf_counter()
        for _ in range(repeats):
            session.run(["prob"], feed)
        latencies[batch_size] = (time.perf_counter() - started) * 1000.0 / repeats
    return {
        "onnx_size_bytes": int(os.path.getsize(onnx_path)),
        "onnx_size_kb": os.path.getsize(onnx_path) / 1024.0,
        "onnx_latency_ms_batch_1": latencies[1],
        "onnx_latency_ms_batch_128": latencies[128],
        "onnx_parity_check": {
            "onnx_path": onnx_path, "onnx_size_bytes": int(os.path.getsize(onnx_path)),
            "samples_verified": int(n), "max_abs_diff": max_diff, "tolerance": 1e-4,
            "passed": True,
        },
    }


def _selection_score(model_report):
    ev = model_report["eval"]["val"]
    overall = ev["overall"]["roc_auc"]
    shop_scores = [v["roc_auc"] for v in ev["by_shop"].values() if v["roc_auc"] is not None]
    mean_shop = float(np.mean(shop_scores)) if shop_scores else overall
    pr = ev["overall"]["pr_auc"]
    latency = model_report["system"].get("onnx_latency_ms_batch_1")
    return (0.5 * (overall or 0.0) + 0.5 * mean_shop, pr or 0.0,
            -(latency if latency is not None else float("inf")))


def _markdown_tables(report):
    lines = ["## Bảng 1 — Hiệu năng phân loại (ngưỡng F1 được chọn trên VAL)", ""]
    for split in ("val", "test"):
        lines += [f"### {split.upper()}", "",
                  "| Model | Shop | ROC-AUC | PR-AUC | Log-Loss | Brier | Best F1 | Precision | Recall |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, result in report["models"].items():
            parts = [("All-shop", result["eval"][split]["overall"])]
            parts.extend(sorted(result["eval"][split]["by_shop"].items()))
            for shop, metrics in parts:
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    name, shop, *[_fmt(metrics.get(k)) for k in
                                  ("roc_auc", "pr_auc", "logloss", "brier", "best_f1",
                                   "precision", "recall")]))
        lines.append("")

    lines += ["## Bảng 2 — Hiệu năng theo độ dài tiền tố", ""]
    for split in ("val", "test"):
        lines += [f"### {split.upper()}", "",
                  "| Model | k | Samples | ROC-AUC | PR-AUC |",
                  "|---|---:|---:|---:|---:|"]
        for name, result in report["models"].items():
            for bucket, metrics in result["eval"][split]["by_prefix_len"].items():
                lines.append(f"| {name} | {bucket} | {metrics['samples']} | "
                             f"{_fmt(metrics.get('roc_auc'))} | {_fmt(metrics.get('pr_auc'))} |")
        lines.append("")

    lines += ["## Bảng 3 — Chi phí tính toán và suy luận CPU", "",
              "| Model | # Params | ONNX KB | s/epoch | Tổng train (s) | ONNX ms/batch=1 | ONNX ms/batch=128 | "
              "Native ms/batch=1 | Native ms/batch=128 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in report["models"].items():
        system = result["system"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, _fmt(result.get("n_parameters")), _fmt(system.get("onnx_size_kb")),
            _fmt(system.get("seconds_per_epoch")), _fmt(system.get("train_seconds")),
            _fmt(system.get("onnx_latency_ms_batch_1")), _fmt(system.get("onnx_latency_ms_batch_128")),
            _fmt(system.get("native_latency_ms_batch_1")), _fmt(system.get("native_latency_ms_batch_128"))))
    lines += ["", "`HistGBM` không được export ONNX trong suite này; cột ONNX để trống và độ trễ native được báo riêng."]
    return "\n".join(lines)


def _fmt(value):
    return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _csv_rows(report):
    rows = []
    for name, result in report["models"].items():
        for split in ("val", "test"):
            ev = result["eval"][split]
            groups = [("All-shop", ev["overall"])] + list(ev["by_shop"].items())
            for shop, metrics in groups:
                rows.append({"row_type": "classification", "model": name, "split": split,
                             "shop": shop, "prefix_bucket": "", **metrics})
            for bucket, metrics in ev["by_prefix_len"].items():
                rows.append({"row_type": "prefix_length", "model": name, "split": split,
                             "shop": "", "prefix_bucket": bucket, **metrics})
        rows.append({"row_type": "system", "model": name, **result["system"],
                     "n_parameters": result.get("n_parameters")})
    return rows


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="data/prefix")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--max-sessions-per-split", type=int)
    parser.add_argument("--max-k-per-session", type=int)
    parser.add_argument("--max-samples", type=int, help="cap samples independently in each split")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true", help="small deterministic run for pipeline validation")
    parser.add_argument("--skip-gbm", action="store_true")
    parser.add_argument("--export-best", action="store_true")
    parser.add_argument("--model-name", choices=tuple(MODEL_SPECS),
                        help="export this neural model instead of the automatic champion")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.d_model < 8 or args.d_model % 4:
        parser.error("--batch-size must be positive and --d-model must be >=8 and divisible by 4")
    if args.quick:
        sessions = args.max_sessions_per_split or 300
        max_k = args.max_k_per_session or 4
        max_samples = args.max_samples or 1200
        epochs = args.epochs or 1
    else:
        sessions = args.max_sessions_per_split or 12000
        max_k = args.max_k_per_session or 6
        max_samples = args.max_samples
        epochs = args.epochs or 3
    if max_samples is not None and max_samples < 1:
        parser.error("--max-samples must be positive")

    torch.set_num_threads(min(os.cpu_count() or 4, 8))
    spec = FT.load_spec(os.path.join(args.prefix, "feature_spec.json"))
    from ml.train_dl import load_sequence_splits
    load_start = time.perf_counter()
    raw_splits = load_sequence_splits(
        args.prefix, spec, max_sessions_per_split=sessions,
        max_k_per_session=max_k, seed=args.seed,
    )
    splits = {name: BenchmarkDataset(ds, max_samples, args.seed + i)
              for i, (name, ds) in enumerate(sorted(raw_splits.items()))}
    if not {"train", "val", "test"}.issubset(splits):
        raise ValueError("prefix dataset must contain train, val and test splits")
    load_seconds = time.perf_counter() - load_start
    print("Loaded split sample counts: " + ", ".join(
        f"{name}={len(splits[name]):,}" for name in ("train", "val", "test")), flush=True)
    pos_rate = float(splits["train"].y.mean())
    pos_weight = math.sqrt((1.0 - pos_rate) / max(pos_rate, 1e-4))
    results = {}
    trained = {}
    for name in MODEL_SPECS:
        model, entry = _train_neural(
            name, spec, splits["train"], splits["val"], splits["test"], args.seed, epochs,
            args.batch_size, args.d_model, args.device, pos_weight,
        )
        results[name] = entry
        trained[name] = model
        print(f"[{name}] VAL AUC={_fmt(entry['eval']['val']['overall']['roc_auc'])} "
              f"PR-AUC={_fmt(entry['eval']['val']['overall']['pr_auc'])} "
              f"train={entry['system']['train_seconds']:.1f}s", flush=True)
    if not args.skip_gbm:
        results["histgbm"] = _fit_gbm(
            splits["train"], splits["val"], splits["test"], args.seed, args.quick,
            pos_weight, args.batch_size,
        )

    requested_export = args.export_best or args.model_name is not None
    champion = args.model_name or max(MODEL_SPECS, key=lambda n: _selection_score(results[n]))
    with tempfile.TemporaryDirectory(prefix="intent_benchmark_") as tmp_dir:
        exported_paths = {}
        for name, model in trained.items():
            path = os.path.join(tmp_dir, name + ".onnx")
            export_info = _export_onnx(model, path, spec, splits["test"])
            exported_paths[name] = path
            results[name]["system"].update({k: export_info[k] for k in (
                "onnx_size_bytes", "onnx_size_kb", "onnx_latency_ms_batch_1",
                "onnx_latency_ms_batch_128")})
            results[name]["onnx_parity_check"] = export_info["onnx_parity_check"]
        if args.model_name is None:
            champion = max(MODEL_SPECS, key=lambda n: _selection_score(results[n]))
        if requested_export:
            os.makedirs(args.out_dir, exist_ok=True)
            final_path = os.path.join(args.out_dir, "intent_v1.onnx")
            shutil.copyfile(exported_paths[champion], final_path)
            results[champion]["onnx_parity_check"]["onnx_path"] = final_path

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "generated_utc": created, "prefix_dir": os.path.abspath(args.prefix),
        "seed": args.seed, "epochs": epochs, "batch_size": args.batch_size,
        "pos_weight": pos_weight, "d_model": args.d_model, "load_seconds": load_seconds,
        "sample_counts": {name: len(ds) for name, ds in splits.items()},
        "champion": champion, "models": results,
    }
    if requested_export:
        import onnx

        model_card = {
            "model_ver": "intent_v1", "architecture": champion,
            "champion_variant": champion, "created_utc": created,
            "feature_version": spec["feature_version"], "feature_spec_file": "feature_spec.json",
            "onnx_file": "intent_v1.onnx", "onnx_size_bytes": results[champion]["system"]["onnx_size_bytes"],
            "max_len": spec["max_len"],
            "inputs": {
                "x_cat": {"dtype": "int64", "shape": ["batch", spec["max_len"], FT.N_CAT],
                          "fields": list(FT.CAT_FIELDS)},
                "x_num": {"dtype": "float32", "shape": ["batch", spec["max_len"], FT.N_NUM],
                          "fields": list(FT.NUM_FEATURES)},
                "length": {"dtype": "int64", "shape": ["batch"]},
                "shop": {"dtype": "int64", "shape": ["batch"], "mapping": spec["shops"]},
            },
            "outputs": {"prob": {"dtype": "float32", "shape": ["batch"]}},
            "hyperparameters": {
                "d_model": args.d_model, "epochs": epochs, "batch_size": args.batch_size,
                "pos_weight": pos_weight, "calibration_params_by_shop_idx":
                    results[champion]["training_info"]["calibration_params_by_shop_idx"],
            },
            "offline_metrics": {
                "val": results[champion]["eval"]["val"], "test": results[champion]["eval"]["test"],
            },
            "onnx_parity_check": results[champion]["onnx_parity_check"],
        }
        os.makedirs(args.out_dir, exist_ok=True)
        shutil.copyfile(os.path.join(args.prefix, "feature_spec.json"),
                        os.path.join(args.out_dir, "feature_spec.json"))
        with open(os.path.join(args.out_dir, "model_card.json"), "w", encoding="utf-8") as file:
            json.dump(model_card, file, indent=2, ensure_ascii=False)
        report["exported_model"] = os.path.join(args.out_dir, "intent_v1.onnx")

    os.makedirs(args.prefix, exist_ok=True)
    json_path = os.path.join(args.prefix, "_benchmark_results.json")
    csv_path = os.path.join(args.prefix, "_benchmark_results.csv")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    csv_rows = _csv_rows(report)
    fieldnames = sorted({key for row in csv_rows for key in row})
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print("\n" + _markdown_tables(report))
    print(f"\nSaved {json_path} and {csv_path}; champion={champion}")
    if requested_export:
        print(f"Exported models/intent_v1.onnx (parity max_abs_diff="
              f"{results[champion]['onnx_parity_check']['max_abs_diff']:.2e})")


if __name__ == "__main__":
    main()
