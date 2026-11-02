"""Adapter: curated synthetic shop (output of batch/preprocess.py) -> canonical unified layer.

Run:  python batch/adapters/synthetic.py --curated data/curated --out data/unified

Mapping
- shop_id = 'synthetic'. event_id, seq_no, event_ts, ts_raw and ts_fixed are kept from preprocess
  (seq_no is the logical order, event_ts is the corrected time).
- category and price come from the product catalog. Unit price == catalog price for 100% of order lines.
- purchase: items = the order's lines from order_items (identical lines kept, each with qty = quantity).
  amount_usd = order total after discount (sum(items.price * qty) = subtotal before discount).
- remove_from_cart does not exist in this source. brand is NULL.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pyspark.sql import functions as F  # noqa: E402
from common.canonical import get_spark, conform, build_sessions, write_events, write_by_shop, read_table  # noqa: E402
from common.contract import CANONICAL_RULES, failed_rules_col  # noqa: E402

SHOP = "synthetic"
ap = argparse.ArgumentParser()
ap.add_argument("--curated", default="data/curated")
ap.add_argument("--out", default="data/unified")
ap.add_argument("--driver-memory", default="4g")
ap.add_argument("--cores", type=int, default=4)
ap.add_argument("--strict", action="store_true")
args = ap.parse_args()
spark = get_spark("adapter-synthetic", args.driver_memory, args.cores, shuffle_partitions=16)

ev = spark.read.parquet(os.path.join(args.curated, "events"))
products = spark.read.parquet(os.path.join(args.curated, "products"))
order_items = spark.read.parquet(os.path.join(args.curated, "order_items"))

prod = products.select(F.col("product_id").cast("long").alias("product_id"), "category",
                       F.col("price_usd").alias("price"))
items = (order_items.join(products.select("product_id", "category"), "product_id")
         .groupBy("order_id")
         .agg(F.array_sort(F.collect_list(F.struct(
             F.col("product_id").cast("long").alias("product_id"),
             F.col("category"),
             F.lit(None).cast("string").alias("brand"),
             F.col("unit_price_usd").alias("price"),
             F.col("quantity").alias("qty")))).alias("items")))

canon = conform(ev.withColumn("shop_id", F.lit(SHOP))
                .withColumn("product_id", F.col("product_id").cast("long"))
                .join(F.broadcast(prod), "product_id", "left")
                .join(items, "order_id", "left"))

bad = canon.filter(F.size(failed_rules_col(CANONICAL_RULES)) > 0)
n_bad = bad.count()
write_events(canon, args.out, SHOP, n_files=6)

out_ev = read_table(spark, args.out, "events").filter(F.col("shop_id") == SHOP)
sessions = build_sessions(out_ev, SHOP)
write_by_shop(sessions, args.out, "sessions", SHOP)
write_by_shop(products.select(F.lit(SHOP).alias("shop_id"), F.col("product_id").cast("long").alias("product_id"),
                              "category", F.lit(None).cast("string").alias("brand"),
                              F.col("price_usd").alias("price")), args.out, "products", SHOP)

cur_sessions = spark.read.parquet(os.path.join(args.curated, "sessions"))
s = spark.read.parquet(os.path.join(args.out, "sessions")).filter(F.col("shop_id") == SHOP)
report = {
    "shop_id": SHOP,
    "rows_in": ev.count(),
    "rows_out": out_ev.count(),
    "contract_violations": n_bad,
    "event_types": {r.event_type: r["count"] for r in out_ev.groupBy("event_type").count().collect()},
    "purchase_amount_vs_items_subtotal_mismatch": out_ev.filter("event_type = 'purchase'").filter(
        F.abs(F.aggregate("items", F.lit(0.0), lambda acc, i: acc + i.price * i.qty)
              * (1 - F.col("discount_pct") / 100) - F.col("amount_usd")) > 0.02).count(),
    "sessions": s.count(),
    "label_mismatch_vs_curated": s.select("session_id", "label_purchase").join(
        cur_sessions.select("session_id", F.col("label_purchase").alias("l0")), "session_id", "full")
        .filter("label_purchase IS NULL OR l0 IS NULL OR label_purchase != l0").count(),
    "splits": {r.split: r["count"] for r in s.groupBy("split").count().collect()},
}
hard = ["contract_violations", "purchase_amount_vs_items_subtotal_mismatch", "label_mismatch_vs_curated"]
failed = [k for k in hard if report[k] != 0] + (["row_count"] if report["rows_in"] != report["rows_out"] else [])
report["status"] = "PASS" if not failed else f"FAIL: {failed}"
os.makedirs(os.path.join(args.out, "_dq"), exist_ok=True)
with open(os.path.join(args.out, "_dq", f"{SHOP}.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
spark.stop()
sys.exit(1 if (failed and args.strict) else 0)
