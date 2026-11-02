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
```
`--strict` exits with code 1 if any hard invariant fails. Each adapter rewrites only its own `shop_id=` folders.

## Tests without the real data (fixtures with independent ground truth)

```bash
python tests/test_runtime_env.py
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
