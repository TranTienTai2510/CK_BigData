# Batch layer: raw → curated → unified (schema v3, 3 shops)

```
data/raw/          7 CSV synthetic   (Kaggle wafaaelhusseini/e-commerce-transactions-clickstream)
data/raw_rees46/   2019-Oct.csv ... 2020-Feb.csv   (Kaggle mkechinov/ecommerce-events-history-in-cosmetics-shop)
data/raw_olist/    9 CSV olist_*.csv + product_category_name_translation.csv   (Kaggle olistbr/brazilian-ecommerce)
data/fx/DEXBZUS.csv  (optional) daily BRL per USD from FRED, for Olist USD conversion
        │
        ├─ batch/preprocess.py ─► data/curated/ ─► batch/adapters/synthetic.py ─┐
        ├─ batch/adapters/rees46.py ────────────────────────────────────────────┤
        └─ batch/adapters/olist.py ─────────────────────────────────────────────┤
                                                                                 ▼
  data/unified/events/shop_id=<shop>/ym=<yyyy-MM>/        canonical events, all shops (olist: purchase only)
  data/unified/sessions/shop_id=<shop>/                   clickstream shops only (labels, cut_seq_no, split)
  data/unified/order_events/shop_id=olist/ym=<yyyy-MM>/   order lifecycle (SLA dashboard)
  data/unified/products/shop_id=<shop>/
  data/unified/olist/{orders,order_items,payments,reviews,products,sellers,customers,geolocation}/
  data/unified/quarantine/shop_id=rees46_cosmetics/
  data/unified/_dq/<shop>.json
        │
        └─ ml/prefixes.py ─► data/prefix/{feature_spec.json, sessions/shop_id=<shop>/split=<split>/, _report.json}
              │
              ├─ ml/baselines.py ─► data/prefix/_baselines.json   (Markov + GBM baselines, AUC/PR-AUC by prefix length)
              └─ ml/train_dl.py  ─► models/{intent_v1.onnx, model_card.json, feature_spec.json}, data/prefix/{_dl_ablation.json, _model_benchmark.json}
```

## Run (Python 3.10+, `pip install pyspark==3.5.5`, JDK 17; stop the Docker stack first)

On Windows every script configures itself (`common/runtime_env.py`): it picks a JDK 17, a HADOOP_HOME with
`winutils.exe` + `hadoop.dll` (D:\hadoop or C:\hadoop), and ignores a stray SPARK_HOME. So it works from a
terminal, the VS Code Run/Debug button or Jupyter. Diagnose with: `python -m common.runtime_env`.
Overrides: `CLICKSTREAM_JAVA_HOME`, `CLICKSTREAM_HADOOP_HOME`.

```bash
python tools/download_olist.py        # kagglehub -> data/raw_olist (needs a Kaggle token)
python batch/preprocess.py         --raw data/raw         --out data/curated --strict
python batch/adapters/synthetic.py --curated data/curated --out data/unified --strict
python batch/adapters/rees46.py    --raw data/raw_rees46  --out data/unified --strict --driver-memory 4g --cores 4
python batch/adapters/olist.py     --raw data/raw_olist   --out data/unified --strict [--fx-csv data/fx/DEXBZUS.csv]
python batch/check_unified.py      --unified data/unified --strict
python ml/prefixes.py              --unified data/unified --out data/prefix
python ml/baselines.py             --prefix  data/prefix  --out data/prefix/_baselines.json
python ml/train_dl.py              --prefix  data/prefix  --out-dir models --report data/prefix/_dl_ablation.json
```
`--strict` exits with code 1 if any hard invariant fails. Each adapter rewrites only its own `shop_id=` folders.
`ml/prefixes.py` builds the shared vocabulary (`data/prefix/feature_spec.json`, from the `train` split only) and the
prefix dataset (`data/prefix/sessions/shop_id=<shop>/split=<split>/`, cut at `cut_seq_no`) used by training and Q2
(`ml/features.py`). `ml/baselines.py` trains the Week 1 Markov and GBM baselines on `train` and outputs ROC-AUC and
PR-AUC tables by prefix length `k` on `val` and `test`. `ml/train_dl.py` runs the unified Binary Classification (Session-prefix Purchase Intent Prediction) benchmark across
Markov, HistGBM, GRU, Hybrid GRU+Wide&Deep, Causal TCN, Bi-LSTM+Attention, MLP-Mixer, and Causal Transformer,
develops the improved `bilstm_attention_v2` variant, runs the ablation suite, and exports the winning Champion to
`models/intent_v1.onnx` + `models/model_card.json` + `models/feature_spec.json` (handover G4), plus `_model_benchmark.json`.

## Tests without the real data (fixtures with independent ground truth)

```bash
python tests/test_runtime_env.py
python tests/test_features_parity.py                                          # G2 batch-vs-stream tensor parity
python tests/test_baselines.py                                                # Markov + GBM baseline checks
python tests/test_onnx_parity.py                                              # G4 PyTorch-vs-ONNX parity (< 1e-4 on 1,000 samples)
python tests/test_q2_parity.py                                                # G5 Q2 Batch-vs-Stream parity (< 1e-5)
python tests/test_online_eval.py                                              # G6 Online AUC vs Offline (|delta| <= 0.02) + A/B test
python analysis/live.py eval --out data/prefix/_online_eval.json              # Online evaluation report
python analysis/live.py monitor --dry-run                                     # Pipeline monitor (STALL / BACKPRESSURE check)
python analysis/live.py tail --threshold 0.5                                  # High intent session tail
python tests/make_rees46_fixture.py --out tests/fixture_rees46
python batch/adapters/rees46.py --raw tests/fixture_rees46 --out data/unified --files-out 4 --shuffle-partitions 16 --strict
python tests/test_rees46_adapter.py --unified data/unified --fixture tests/fixture_rees46     # 22 checks
python tests/make_olist_fixture.py --out tests/fixture_olist
python batch/adapters/olist.py --raw tests/fixture_olist --out data/unified --strict
python tests/test_olist_adapter.py --unified data/unified --fixture tests/fixture_olist       # 33 checks
python tests/scale_fixture.py --copies 180      # ~20.6M REES46-format rows, 2.3 GB (scale test)
```

## Canonical schema v3 (`common/canonical.py`)
- Key: `(shop_id, session_id)`, `(shop_id, event_id)`. Hash ids (REES46 sessions/orders, all Olist ids) are xxhash64 → **signed** Int64.
- `event_type`: page_view | add_to_cart | remove_from_cart | checkout | purchase.
- REES46 sessions = (user_session, user_id) split at gaps > 30 min; cart/remove/purchase duplicate rows → `qty`.
- `purchase` = 1 event per order: `order_id`, `amount_usd`, `items: array<{product_id, category, brand, price, qty}>`;
  `currency` + `amount_local` when the shop does not sell in USD (Olist: BRL). NULL currency = USD.
- Olist: one purchase event per order with items, a one-event "session" (no browsing data: conversion KPIs are N/A).
- `order_events`: order_placed → payment_approved → handed_to_carrier → delivered → review_created; stages 1-4 use a
  running max (event_ts ≥ ts_raw), review is clamped to ≥ order_placed; `days_late` on delivered.
- Olist core window 2017-01..2018-08 (`in_core_window` in olist/orders); 2016 and Sep-Oct 2018 are sparse.
- Order inside a session = `seq_no`. All timestamps UTC (Olist source assumed America/Sao_Paulo, DST-aware).
- Files inside each `shop_id/ym` folder, read in file-name order, are sorted by `(event_ts, event_id)`.
