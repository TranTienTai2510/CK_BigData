"""Adapter: REES46 "eCommerce Events History in Cosmetics Shop" (Kaggle, monthly CSVs) -> canonical unified layer.

Run (5 monthly files, ~20.7M rows, ~2.3 GB CSV; ~4 GB JVM heap, a few GB of shuffle spill on disk):
  python batch/adapters/rees46.py --raw data/raw_rees46 --out data/unified --strict

Why PySpark and not pandas: a user_session can cross month-file boundaries (Oct 31 -> Nov 1), so
seq_no must be computed on a shuffle by session over ALL files. Spark spills to disk; pandas would need
the whole 20M rows in RAM.

Source columns: event_time ("2019-10-01 00:00:00 UTC"), event_type (view|cart|remove_from_cart|purchase),
product_id, category_id, category_code, brand, price, user_id, user_session (uuid).

Mapping / cleaning (v2, after the first run on the real data)
1. Quarantine (with reasons): corrupt CSV rows, unparsable time, unknown type, NULL user_session / user_id /
   product_id, NULL or negative price, no category at all.
2. Exact duplicate rows (all 9 source columns equal) are COLLAPSED:
   - cart / remove_from_cart / purchase: one row with qty = number of copies. On the real data 25% of
     remove_from_cart rows and 2% of cart rows are exact duplicates, vs 0.01% of views. Logging noise
     would hit every type about equally, so the copies are read as one row per unit.
   - view: copies dropped.
3. view -> page_view, cart -> add_to_cart, remove_from_cart, purchase -> see 6.
4. category = category_code if present else 'cid:<category_id>' (category_code is ~98% NULL in this shop).
5. Re-sessionization: REES46 user_session is not a reliable session. On the real data the max inner
   gap p99 is 3.9 days, the longest "session" spans 151 days, and 272 ids are shared by several user_id.
   A session = (user_session, user_id) split at every inactivity gap > --session-gap-s (default 1800 s,
   the same 30 min rule the synthetic shop follows).
   session_id = xxhash64(shop, user_session, user_id, segment_no). Collisions are checked.
6. purchase rows -> ONE order-level purchase event per (session_id, event_time):
   order_id = xxhash64(shop, session_id, event_time), items = product rows with qty,
   amount_usd = sum(price * qty). REES46 states no currency; prices are treated as USD.
7. seq_no inside a session: ORDER BY event_ts, funnel rank (view < add < remove < checkout < purchase),
   original file position. Same-second ties are common, and the funnel rank resolves them.
8. event_id = original row position (min over the rows of an order). It is unique and deterministic
   for the same input files.
9. No device/source/country in the source -> NULL. session_start_ts = first event of the session.
All timestamps are UTC. The report formats them inside Spark (date_format), because Python collect()
converts them to the local timezone of the machine (UTC+7 on a Vietnamese Windows laptop).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pyspark import StorageLevel  # noqa: E402
from pyspark.sql import functions as F, Window as W  # noqa: E402
from common.canonical import (get_spark, conform, build_sessions, write_events, write_by_shop,  # noqa: E402
                              type_rank_col, read_table)
from common.contract import CANONICAL_RULES, failed_rules_col  # noqa: E402

SHOP = "rees46_cosmetics"
SRC_COLS = ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price",
            "user_id", "user_session"]
RAW_SCHEMA = ("event_time STRING, event_type STRING, product_id BIGINT, category_id BIGINT, category_code STRING, "
              "brand STRING, price DOUBLE, user_id BIGINT, user_session STRING, _corrupt STRING")
TYPE_MAP = {"view": "page_view", "cart": "add_to_cart", "remove_from_cart": "remove_from_cart",
            "purchase": "purchase"}
TS_FMT = "yyyy-MM-dd HH:mm:ss"

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw_rees46", help="folder with 2019-Oct.csv ... 2020-Feb.csv (.csv or .csv.gz)")
ap.add_argument("--out", default="data/unified")
ap.add_argument("--driver-memory", default="4g")
ap.add_argument("--cores", type=int, default=4)
ap.add_argument("--shuffle-partitions", type=int, default=64)
ap.add_argument("--files-out", type=int, default=40, help="number of output parquet files")
ap.add_argument("--session-gap-s", type=int, default=1800, help="split a session at inactivity gaps > this")
ap.add_argument("--strict", action="store_true")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.raw, "*.csv")) + glob.glob(os.path.join(args.raw, "*.csv.gz")))
if not files:
    sys.exit(f"no CSV files in {args.raw}")
spark = get_spark("adapter-rees46", args.driver_memory, args.cores, args.shuffle_partitions)

type_map = F.create_map(*[x for k, v in TYPE_MAP.items() for x in (F.lit(k), F.lit(v))])
parse_ts = lambda c: F.to_timestamp(F.regexp_replace(c, r"\s*UTC$", ""), TS_FMT)  # noqa: E731
fmt_ts = lambda c: F.date_format(c, TS_FMT)  # noqa: E731

# ---------------------------------------------------------------- 1. read + source validation
raw = (spark.read.csv(files, header=True, schema=RAW_SCHEMA, mode="PERMISSIVE", columnNameOfCorruptRecord="_corrupt")
       .withColumn("_src_file", F.col("_metadata.file_name"))
       .withColumn("_row", F.monotonically_increasing_id()))
reason = lambda cond, name: F.when(cond, F.lit(name))  # noqa: E731
flagged = raw.withColumn("reasons", F.filter(F.array(
    reason(F.col("_corrupt").isNotNull(), "corrupt_record"),
    reason(parse_ts("event_time").isNull(), "bad_timestamp"),
    reason(type_map[F.col("event_type")].isNull(), "unknown_event_type"),
    reason(F.col("user_session").isNull() | (F.trim("user_session") == ""), "null_session"),
    reason(F.col("user_id").isNull(), "null_user"),
    reason(F.col("product_id").isNull(), "null_product"),
    reason(F.col("price").isNull() | (F.col("price") < 0), "bad_price"),
    reason(F.col("category_id").isNull() & F.col("category_code").isNull(), "no_category"),
), lambda x: x.isNotNull()))

quarantine_dir = os.path.join(args.out, "quarantine", f"shop_id={SHOP}")
flagged.filter(F.size("reasons") > 0).drop("_row").write.mode("overwrite").parquet(quarantine_dir)
per_file = {r._src_file: {"rows": r.rows, "quarantined": r.q} for r in
            flagged.groupBy("_src_file").agg(F.count("*").alias("rows"),
                                             F.sum((F.size("reasons") > 0).cast("int")).alias("q")).collect()}
q = spark.read.parquet(quarantine_dir)
q_reasons = {r.reason: r["count"] for r in q.select(F.explode("reasons").alias("reason")).groupBy("reason").count().collect()}

# ---------------------------------------------------------------- 2. collapse exact duplicates
dedup = (flagged.filter(F.size("reasons") == 0)
         .groupBy(*SRC_COLS).agg(F.min("_row").alias("_row"), F.count("*").alias("_n")))

# ---------------------------------------------------------------- 3-5. map + re-sessionize
w_user = W.partitionBy("user_session", "user_id").orderBy("event_ts", "type_rank", "_row")
ev = (dedup
      .withColumn("raw_type", F.col("event_type"))
      .withColumn("event_ts", parse_ts("event_time"))
      .withColumn("event_type", type_map[F.col("event_type")])
      .withColumn("type_rank", type_rank_col())
      .withColumn("customer_id", F.col("user_id"))
      .withColumn("category", F.coalesce("category_code", F.concat(F.lit("cid:"), F.col("category_id").cast("string"))))
      .withColumn("_gap", F.col("event_ts").cast("long") - F.lag(F.col("event_ts").cast("long")).over(w_user))
      .withColumn("_seg", F.sum(F.when(F.col("_gap") > args.session_gap_s, 1).otherwise(0))
                  .over(w_user.rowsBetween(W.unboundedPreceding, W.currentRow)))
      .withColumn("session_id", F.xxhash64(F.lit(SHOP), "user_session", "user_id", "_seg"))
      .persist(StorageLevel.DISK_ONLY))

dup_stats = {r.raw_type: {"copies_collapsed": r.d, "rows_with_copies": r.k} for r in
             ev.groupBy("raw_type").agg(F.sum(F.col("_n") - 1).alias("d"),
                                        F.sum((F.col("_n") > 1).cast("int")).alias("k")).collect()}
src_ids = ev.select("user_session").distinct().count()
src_pairs = ev.select("user_session", "user_id").distinct().count()
collisions = (ev.select("session_id", "user_session", "user_id", "_seg").distinct()
              .groupBy("session_id").count().filter("count > 1").count())

item_events = (ev.filter("event_type <> 'purchase'")
               .select("session_id", "customer_id", "event_type", "event_ts", "product_id",
                       "category", "brand", "price", F.col("_row").alias("event_id"),
                       F.when(F.col("event_type").isin("add_to_cart", "remove_from_cart"),
                              F.col("_n").cast("int")).alias("qty")))
orders = (ev.filter("event_type = 'purchase'")
          .groupBy("session_id", "event_ts")
          .agg(F.min("_row").alias("event_id"),
               F.min("customer_id").alias("customer_id"),
               F.array_sort(F.collect_list(F.struct(
                   F.col("product_id").alias("product_id"), "category", "brand", "price",
                   F.col("_n").cast("int").alias("qty")))).alias("items"),
               F.round(F.sum(F.col("price") * F.col("_n")), 2).alias("amount_usd"))
          .withColumn("event_type", F.lit("purchase"))
          .withColumn("order_id", F.xxhash64(F.lit(SHOP), "session_id", fmt_ts("event_ts"))))

# ---------------------------------------------------------------- 7-9. order inside session + canonical
w = W.partitionBy("session_id").orderBy("event_ts", "type_rank", "event_id")
unified = (item_events.unionByName(orders, allowMissingColumns=True)
           .withColumn("type_rank", type_rank_col())
           .withColumn("seq_no", F.row_number().over(w))
           .withColumn("session_start_ts", F.min("event_ts").over(W.partitionBy("session_id")))
           .withColumn("shop_id", F.lit(SHOP))
           .withColumn("ts_raw", F.col("event_ts"))
           .withColumn("ts_fixed", F.lit(False)))
write_events(conform(unified), args.out, SHOP, n_files=args.files_out)
ev.unpersist()

# ---------------------------------------------------------------- sessions, products
out_ev = read_table(spark, args.out, "events").filter(F.col("shop_id") == SHOP)
write_by_shop(build_sessions(out_ev, SHOP), args.out, "sessions", SHOP)
products = (out_ev.filter(F.col("product_id").isNotNull())
            .groupBy("shop_id", "product_id")
            .agg(F.max_by("category", "event_ts").alias("category"),
                 F.max_by("brand", "event_ts").alias("brand"),
                 F.max_by("price", "event_ts").alias("price"),
                 F.min("price").alias("price_min"), F.max("price").alias("price_max"),
                 F.count("*").alias("n_events")))
write_by_shop(products, args.out, "products", SHOP)

# ---------------------------------------------------------------- data quality report
s = spark.read.parquet(os.path.join(args.out, "sessions")).filter(F.col("shop_id") == SHOP)
p = spark.read.parquet(os.path.join(args.out, "products")).filter(F.col("shop_id") == SHOP)
o = out_ev.filter("event_type = 'purchase'")
qs = [0.5, 0.9, 0.99, 0.999, 1.0]
quant = lambda df, c: dict(zip(["p50", "p90", "p99", "p999", "max"], df.approxQuantile(c, qs, 0.0005)))  # noqa: E731
viol = out_ev.select(F.explode(failed_rules_col(CANONICAL_RULES)).alias("rule")).groupBy("rule").count().collect()
n_sessions = s.count()
seq = s.select(F.sum("n_events").alias("n")).first().n
items_rows = o.select(F.sum(F.size("items"))).first()[0] or 0
dups_total = sum(v["copies_collapsed"] for v in dup_stats.values())
t = out_ev.select(fmt_ts(F.min("event_ts")).alias("t0"), fmt_ts(F.max("event_ts")).alias("t1")).first()

report = {
    "shop_id": SHOP,
    "adapter_version": 2,
    "session_gap_s": args.session_gap_s,
    "input_files": per_file,
    "rows_raw": sum(v["rows"] for v in per_file.values()),
    "quarantined": {"total": q.count(), "by_reason": q_reasons},
    "duplicates_collapsed": {"total": dups_total, "by_raw_type": dup_stats,
                             "rule": "cart/remove_from_cart/purchase copies -> qty; view copies dropped"},
    "qty_gt1": {"add_to_cart": out_ev.filter("event_type = 'add_to_cart' AND qty > 1").count(),
                "remove_from_cart": out_ev.filter("event_type = 'remove_from_cart' AND qty > 1").count(),
                "purchase_items": o.select(F.explode("items").alias("i")).filter("i.qty > 1").count()},
    "rows_out": out_ev.count(),
    "event_types": {r.event_type: r["count"] for r in out_ev.groupBy("event_type").count().collect()},
    "time_range_utc": [t.t0, t.t1],
    "rows_per_month": {r.ym: r["count"] for r in out_ev.groupBy("ym").count().orderBy("ym").collect()},
    "sessionization": {"source_user_session_ids": src_ids, "source_session_user_pairs": src_pairs,
                       "sessions_out": n_sessions,
                       "sessions_added_by_gap_split": n_sessions - src_pairs,
                       "sessions_added_by_user_split": src_pairs - src_ids},
    "entities": {"sessions": n_sessions, "customers": out_ev.select("customer_id").distinct().count(),
                 "products": p.count(), "categories": out_ev.select("category").distinct().count(),
                 "category_from_code_pct": round(out_ev.filter("category IS NOT NULL").select(
                     F.avg((~F.col("category").startswith("cid:")).cast("double"))).first()[0] * 100, 2)},
    "orders": {"n_orders": o.count(), "items_per_order": quant(o.withColumn("k", F.size("items")), "k"),
               "amount_usd": quant(o, "amount_usd"),
               "sessions_with_purchase": s.filter("label_purchase = 1").count(),
               "conversion_rate": round(s.select(F.avg("label_purchase")).first()[0], 4),
               "sessions_with_multiple_orders": s.filter("n_orders > 1").count(),
               "purchase_not_last_event_pct": round(s.filter("label_purchase = 1").select(
                   F.avg((~F.col("purchase_is_last")).cast("double"))).first()[0] * 100, 2)},
    "session_shape": {"n_events": quant(s, "n_events"), "duration_s": quant(s, "duration_s"),
                      "max_gap_s": quant(s.filter("n_events > 1"), "max_gap_s"),
                      "single_event_pct": round(s.select(F.avg((F.col("n_events") == 1).cast("double"))).first()[0]
                                                * 100, 2),
                      "sessions_gt_200_events": s.filter("n_events > 200").count(),
                      "prefix_len_before_cut": quant(s.withColumn(
                          "k", F.coalesce(F.col("cut_seq_no") - 1, F.col("n_events"))), "k")},
    "price": {**quant(out_ev.filter("price IS NOT NULL"), "price"),
              "zero_price_rows": out_ev.filter("price = 0").count()},
    "splits": {r.split: {"sessions": r.n, "conv_rate": round(r.cr, 4)} for r in
               s.groupBy("split").agg(F.count("*").alias("n"), F.avg("label_purchase").alias("cr")).collect()},
    "invariants": {
        "contract_violations": {r.rule: r["count"] for r in viol},
        "event_id_duplicates": out_ev.count() - out_ev.select("event_id").distinct().count(),
        "seq_no_not_contiguous": (out_ev.groupBy("session_id")
                                  .agg(F.min("seq_no").alias("lo"), F.max("seq_no").alias("hi"),
                                       F.countDistinct("seq_no").alias("k"), F.count("*").alias("c"))
                                  .filter("lo != 1 OR hi != c OR k != c").count()),
        "session_id_hash_collisions": collisions,
        "sessions_multi_customer": s.filter("n_customers > 1").count(),
        "sessions_gap_over_threshold": s.filter(F.col("max_gap_s") > args.session_gap_s).count(),
        "rows_balance": (sum(v["rows"] for v in per_file.values()) - q.count()
                         - dups_total - (seq + items_rows - o.count())),
    },
}
inv = report["invariants"]
failed = [k for k, v in inv.items() if (v if isinstance(v, int) else len(v)) != 0]
report["status"] = "PASS" if not failed else f"FAIL: {failed}"
os.makedirs(os.path.join(args.out, "_dq"), exist_ok=True)
with open(os.path.join(args.out, "_dq", f"{SHOP}.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
spark.stop()
sys.exit(1 if (failed and args.strict) else 0)
