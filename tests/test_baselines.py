"""Unit & end-to-end test for ml/baselines.py (Markov + GBM baselines and prefix-length AUC/PR-AUC tables).

Run:
  python tests/test_baselines.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from ml.baselines import BUCKET_NAMES, MarkovPurchaseBaseline, evaluate_predictions, fit_gbm  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    n = 2000
    # Synthetic separable prefix features: cart_units and add_to_cart strongly increase purchase probability
    s = np.full(n, 3, dtype=np.int32)
    t1 = rng.choice([3, 4, 5], size=n, p=[0.6, 0.3, 0.1]).astype(np.int32)
    t2 = rng.choice([0, 3, 4], size=n).astype(np.int32)
    t3 = rng.choice([0, 3, 4], size=n).astype(np.int32)
    cat = rng.integers(3, 10, size=n, dtype=np.int32)
    cart = (t1 == 4).astype(np.int32)
    k = rng.integers(1, 25, size=n, dtype=np.int32)
    kb = np.where(k == 1, 0, np.where(k == 2, 1, np.where(k <= 5, 2, np.where(k <= 10, 3, 4)))).astype(np.int32)
    trans = rng.integers(0, 3, size=(n, 16), dtype=np.int16)

    logit = -1.2 + 1.5 * cart + 0.05 * k + 0.4 * (t2 == 4)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(np.int32)

    X = np.column_stack([
        s, t1, cat, np.ones(n), t2, cat, t1, cat,
        rng.normal(2.0, 0.5, size=(n, 10)),
        rng.normal(1.0, 0.5, size=(n, 8)),
        cart.astype(np.float32),
        np.ones(n),
        cart.astype(np.float32),
        rng.normal(1.0, 0.2, size=(n, 2)),
    ]).astype(np.float32)

    tr = slice(0, 1400)
    te = slice(1400, 2000)
    markov_tr = (s[tr], t1[tr], t2[tr], t3[tr], cat[tr], cart[tr], kb[tr], trans[tr])
    markov_te = (s[te], t1[te], t2[te], t3[te], cat[te], cart[te], kb[te], trans[te])

    m = MarkovPurchaseBaseline().fit(markov_tr, y[tr])
    pm = m.predict_proba(markov_te)
    g = fit_gbm(X[tr], y[tr], seed=7)
    pg = g.predict_proba(X[te])[:, 1]

    ev_m = evaluate_predictions(y[te], pm, k[te], np.full(len(y[te]), "synthetic", dtype=object))
    ev_g = evaluate_predictions(y[te], pg, k[te], np.full(len(y[te]), "synthetic", dtype=object))

    checks = {
        "markov_probs_in_0_1": (bool(np.all((pm > 0) & (pm < 1))), True),
        "gbm_probs_in_0_1": (bool(np.all((pg > 0) & (pg < 1))), True),
        "markov_auc_above_0.65": (ev_m["overall"]["roc_auc"] > 0.65, True),
        "gbm_auc_above_0.65": (ev_g["overall"]["roc_auc"] > 0.65, True),
        "buckets_present": (set(ev_g["by_prefix_len"]).issubset(set(BUCKET_NAMES)) and len(ev_g["by_prefix_len"]) >= 4,
                            True),
    }

    if os.path.isfile("data/prefix/_baselines.json"):
        rep = json.load(open("data/prefix/_baselines.json", encoding="utf-8"))
        checks["report_has_val_and_test"] = ({"val", "test"}.issubset(rep["results"]), True)
        for sp in ("val", "test"):
            for mdl in ("markov", "gbm"):
                ov = rep["results"][sp][mdl]["overall"]
                checks[f"{sp}_{mdl}_auc_valid"] = (0.5 < ov["roc_auc"] < 1.0 and 0.0 < ov["pr_auc"] < 1.0, True)

    failed = [name for name, (got, exp) in checks.items() if got != exp]
    for name, (got, exp) in checks.items():
        print(f"{'OK ' if got == exp else 'BAD'} {name:35s} got={got} expected={exp}")
    print("TEST", "PASS" if not failed else f"FAIL {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
