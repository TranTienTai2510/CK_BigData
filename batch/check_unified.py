"""Validate the unified layer (all shops) against the canonical contracts.

Run:  python batch/check_unified.py --unified data/unified --strict
Tables: events (all shops), sessions (clickstream shops), order_events (shops with an order lifecycle).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import functions as F, Window as W  # noqa: E402
from common.canonical import get_spark, read_table, EVENT_SCHEMA, EVENT_COLS, ORDER_EVENT_SCHEMA  # noqa: E402
from common.contract import CANONICAL_RULES, ORDER_EVENT_RULES, failed_rules_col  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--unified", default="data/unified")
ap.add_argument("--strict", action="store_true")
args = ap.parse_args()
spark = get_spark("check-unified", shuffle_partitions=32)
fmt = lambda c: F.date_format(c, "yyyy-MM-dd HH:mm:ss")  # noqa: E731  (UTC, formatted in Spark)

ev = read_table(spark, args.unified, "events")
s = read_table(spark, args.unified, "sessions")
has_oe = os.path.isdir(os.path.join(args.unified, "order_events"))
oe = read_table(spark, args.unified, "order_events") if has_oe else None

per_shop = {}
for r in ev.groupBy("shop_id").agg(F.count("*").alias("rows"), F.countDistinct("session_id").alias("sessions"),
                                   fmt(F.min("event_ts")).alias("t0"), fmt(F.max("event_ts")).alias("t1")).collect():
    per_shop[r.shop_id] = {"rows": r.rows, "sessions": r.sessions, "time_range_utc": [r.t0, r.t1]}
for r in ev.groupBy("shop_id", "event_type").count().collect():
    per_shop[r.shop_id].setdefault("event_types", {})[r.event_type] = r["count"]
for r in s.groupBy("shop_id", "split").agg(F.count("*").alias("n"), F.avg("label_purchase").alias("cr")).collect():
    per_shop[r.shop_id].setdefault("splits", {})[r.split] = {"sessions": r.n, "conv_rate": round(r.cr, 4)}
if has_oe:
    for r in oe.groupBy("shop_id", "event_type").count().collect():
        per_shop.setdefault(r.shop_id, {}).setdefault("order_events", {})[r.event_type] = r["count"]

clickstream_shops = [r.shop_id for r in s.select("shop_id").distinct().collect()]
inv = {
    "missing_columns": [f.name for f in EVENT_SCHEMA.fields if f.name not in ev.columns],
    "contract_violations": {f"{r.shop_id}:{r.rule}": r["count"] for r in
                            ev.select("shop_id", F.explode(failed_rules_col(CANONICAL_RULES)).alias("rule"))
                            .groupBy("shop_id", "rule").count().collect()},
    "duplicate_event_ids": ev.groupBy("shop_id", "event_id").count().filter("count > 1").count(),
    "duplicate_session_seq": ev.groupBy("shop_id", "session_id", "seq_no").count().filter("count > 1").count(),
    "time_order_ne_seq_order": (ev.withColumn("p", F.row_number().over(
        W.partitionBy("shop_id", "session_id").orderBy("event_ts", "seq_no"))).filter("p != seq_no").count()),
    "sessions_without_events": s.join(ev.select("shop_id", "session_id").distinct(),
                                      ["shop_id", "session_id"], "left_anti").count(),
    "clickstream_events_without_session": (ev.filter(F.col("shop_id").isin(clickstream_shops))
                                           .select("shop_id", "session_id").distinct()
                                           .join(s.select("shop_id", "session_id"), ["shop_id", "session_id"],
                                                 "left_anti").count()),
}
if has_oe:
    inv["order_events_missing_columns"] = [f.name for f in ORDER_EVENT_SCHEMA.fields if f.name not in oe.columns]
    inv["order_events_contract_violations"] = {
        f"{r.shop_id}:{r.rule}": r["count"] for r in
        oe.select("shop_id", F.explode(failed_rules_col(ORDER_EVENT_RULES)).alias("rule"))
        .groupBy("shop_id", "rule").count().collect()}
    inv["order_events_duplicate_ids"] = oe.groupBy("shop_id", "event_id").count().filter("count > 1").count()

# Replay order: inside each (shop_id, ym) directory, rows must be sorted by (event_ts, event_id) when files
# are read in file-name order. Checked per file (row position = monotonically_increasing_id, valid because
# maxPartitionBytes is raised so each file is read as one split) plus at the boundaries between files.
spark.conf.set("spark.sql.files.maxPartitionBytes", str(1 << 30))
for table in ["events"] + (["order_events"] if has_oe else []):
    pos = (spark.read.parquet(os.path.join(args.unified, table))
           .select("shop_id", "ym", F.struct("event_ts", "event_id").alias("k"),
                   F.col("_metadata.file_name").alias("f"))
           .withColumn("i", F.monotonically_increasing_id()))
    in_file = (pos.withColumn("prev", F.lag("k").over(W.partitionBy("shop_id", "ym", "f").orderBy("i")))
               .filter("prev > k").groupBy("shop_id", "ym").count())
    bounds = (pos.groupBy("shop_id", "ym", "f").agg(F.min_by("k", "i").alias("first"), F.max_by("k", "i").alias("last"))
              .withColumn("prev_last", F.lag("last").over(W.partitionBy("shop_id", "ym").orderBy("f")))
              .filter("prev_last > first").groupBy("shop_id", "ym").count())
    inv[f"{table}_unsorted_pairs_in_files"] = {f"{r.shop_id}/{r.ym}": r["count"] for r in in_file.collect()}
    inv[f"{table}_unsorted_file_boundaries"] = {f"{r.shop_id}/{r.ym}": r["count"] for r in bounds.collect()}

failed = [k for k, v in inv.items() if (len(v) if isinstance(v, (dict, list)) else v) != 0]
sample = {r.shop_id: json.loads(r.j) for r in ev.filter("event_type = 'purchase'")
          .withColumn("r", F.row_number().over(W.partitionBy("shop_id").orderBy("event_id"))).filter("r = 1")
          .select("shop_id", F.to_json(F.struct(*EVENT_COLS)).alias("j")).collect()}
report = {"per_shop": per_shop, "invariants": inv, "sample_purchase_payload": sample,
          "status": "PASS" if not failed else f"FAIL: {failed}"}
print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
spark.stop()
sys.exit(1 if (failed and args.strict) else 0)
