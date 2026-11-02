"""Batch preprocessing: raw Kaggle CSV -> curated Parquet.

The curated layer is the single source for the simulator (replay), offline training and dimension
tables used by streaming joins.

Run (local):  python batch/preprocess.py --raw data/raw --out data/curated --strict
Run (cluster): spark-submit --py-files common/contract.py batch/preprocess.py --raw ... --out ...

Fixes applied (all measured on the raw files; see dq_report.json after each run):
  1. Logical order inside a session is event_id, not timestamp: ~1.5K add_to_cart events carry a
     timestamp later than the next event (generator jitter of 1-30 s). event_ts = reverse cumulative
     min of ts_raw in event_id order, so time order == logical order. ts_raw is kept, ts_fixed flags it.
  2. seq_no (1..n per session) is added: streaming state orders by seq_no, never by timestamp.
  3. purchase events get order_id via exact (customer_id, timestamp) match (1:1 in raw data).
  4. events are denormalised with session attributes (customer_id, device, source, country,
     session_start_ts) because a real clickstream event carries them in its payload.
  5. order_items: identical lines are LEGIT (same product added to cart twice; subtotal counts both).
     They are kept and numbered (line_no); an aggregated view by (order_id, product_id) is also written.
  6. customers: PII (name, email) dropped; signup_date is inconsistent with activity for ~50% of
     customers -> flagged, first_session_ts added; do not use signup_date for tenure features.
  7. reviews: review_time precedes order_time for ~50% -> flagged; duplicate (order_id, product_id)
     reviews -> keep lowest review_id.
  8. All timestamps are interpreted as UTC.
"""
import argparse
import json
import os
import sys

from pyspark.sql import SparkSession, functions as F, Window as W

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.contract import EVENT_RULES, LINK_RULES, failed_rules_col  # noqa: E402
from common.runtime_env import configure  # noqa: E402

configure()   # JDK 17 + winutils/hadoop.dll on Windows, whatever launched the script
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="data/curated")
ap.add_argument("--strict", action="store_true", help="exit 1 if a hard invariant fails")
args = ap.parse_args()

spark = (SparkSession.builder.appName("clickstream-preprocess")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.driver.memory", "2g")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

TS_FMT = "yyyy-MM-dd'T'HH:mm:ss"
SCHEMAS = {
    "events": "event_id LONG, session_id LONG, timestamp STRING, event_type STRING, product_id DOUBLE, "
              "qty DOUBLE, cart_size DOUBLE, payment STRING, discount_pct DOUBLE, amount_usd DOUBLE",
    "sessions": "session_id LONG, customer_id LONG, start_time STRING, device STRING, source STRING, country STRING",
    "products": "product_id INT, category STRING, name STRING, price_usd DOUBLE, cost_usd DOUBLE, margin_usd DOUBLE",
    "customers": "customer_id LONG, name STRING, email STRING, country STRING, age INT, signup_date STRING, "
                 "marketing_opt_in BOOLEAN",
    "orders": "order_id LONG, customer_id LONG, order_time STRING, payment_method STRING, discount_pct INT, "
              "subtotal_usd DOUBLE, total_usd DOUBLE, country STRING, device STRING, source STRING",
    "order_items": "order_id LONG, product_id INT, unit_price_usd DOUBLE, quantity INT, line_total_usd DOUBLE",
    "reviews": "review_id LONG, order_id LONG, product_id INT, rating INT, review_text STRING, review_time STRING",
}


def read(name):
    return spark.read.csv(f"{args.raw}/{name}.csv", header=True, schema=SCHEMAS[name], mode="FAILFAST")


def out(name):
    return os.path.join(args.out, name)


report = {"rows_raw": {}, "fixes": {}, "info": {}, "invariants": {}}

# ---------------------------------------------------------------- dimensions
products = read("products")
cats = [r.category for r in products.select("category").distinct().orderBy("category").collect()]
cat_idx = {c: i + 1 for i, c in enumerate(cats)}  # 0 is reserved for PAD / unknown
cat_map = F.create_map(*[x for c, i in cat_idx.items() for x in (F.lit(c), F.lit(i))])
products = products.withColumn("category_idx", cat_map[F.col("category")])

sessions = (read("sessions")
            .withColumn("start_time", F.to_timestamp("start_time", TS_FMT))
            .withColumn("split", F.when(F.year("start_time") <= 2023, "train")
                                  .when(F.year("start_time") == 2024, "val").otherwise("test")))

orders = read("orders").withColumn("order_time", F.to_timestamp("order_time", TS_FMT))

# ---------------------------------------------------------------- events
raw_events = read("events")
w_seq = W.partitionBy("session_id").orderBy("event_id")
events = (raw_events
          .withColumn("ts_raw", F.to_timestamp("timestamp", TS_FMT)).drop("timestamp")
          .withColumn("product_id", F.col("product_id").cast("int"))
          .withColumn("qty", F.col("qty").cast("int"))
          .withColumn("cart_size", F.col("cart_size").cast("int"))
          .withColumn("discount_pct", F.col("discount_pct").cast("int"))
          .withColumn("seq_no", F.row_number().over(w_seq))
          # an event cannot happen after its logical successor -> clamp to min of all later timestamps
          .withColumn("event_ts", F.min("ts_raw").over(w_seq.rowsBetween(W.currentRow, W.unboundedFollowing)))
          .withColumn("ts_fixed", F.col("event_ts") != F.col("ts_raw"))
          .join(sessions.select("session_id", "customer_id", "device", "source", "country",
                                F.col("start_time").alias("session_start_ts")), "session_id", "left"))

purchase_link = (events.filter("event_type = 'purchase'")
                 .select("event_id", "customer_id", F.col("ts_raw").alias("order_time"))
                 .join(orders.select("order_id", "customer_id", "order_time"), ["customer_id", "order_time"], "left")
                 .select("event_id", "order_id"))
events = (events.join(purchase_link, "event_id", "left")
          .withColumn("dq_failed", failed_rules_col({**EVENT_RULES, **LINK_RULES}))
          .withColumn("year", F.year("event_ts"))
          .cache())

EVENT_COLS = ["event_id", "seq_no", "session_id", "customer_id", "event_type", "product_id", "qty",
              "cart_size", "payment", "discount_pct", "amount_usd", "order_id", "device", "source",
              "country", "session_start_ts", "event_ts", "ts_raw", "ts_fixed", "year"]
good = events.filter(F.size("dq_failed") == 0)
bad = events.filter(F.size("dq_failed") > 0)

# ---------------------------------------------------------------- session labels (for training / online eval)
session_labels = (good.groupBy("session_id").agg(
        F.count("*").alias("n_events"),
        F.sum((F.col("event_type") == "page_view").cast("int")).alias("n_views"),
        F.sum((F.col("event_type") == "add_to_cart").cast("int")).alias("n_atc"),
        F.max((F.col("event_type") == "checkout").cast("int")).alias("has_checkout"),
        F.max((F.col("event_type") == "purchase").cast("int")).alias("label_purchase"),
        F.min(F.when(F.col("event_type") == "checkout", F.col("seq_no"))).alias("checkout_seq_no"),
        F.min("event_ts").alias("first_event_ts"),
        F.max("event_ts").alias("last_event_ts"),
        F.max("order_id").alias("order_id"))
    .join(sessions, "session_id", "right")
    .withColumn("duration_s", F.col("last_event_ts").cast("long") - F.col("first_event_ts").cast("long")))

orders_out = orders.join(events.filter("event_type = 'purchase'").select("order_id", "session_id"), "order_id", "left")

# ---------------------------------------------------------------- order_items, customers, reviews
order_items = (read("order_items").withColumn("_rid", F.monotonically_increasing_id())
               .withColumn("line_no", F.row_number().over(W.partitionBy("order_id").orderBy("_rid")))
               .drop("_rid"))
order_items_by_product = (order_items.groupBy("order_id", "product_id")
                          .agg(F.first("unit_price_usd").alias("unit_price_usd"),
                               F.sum("quantity").alias("quantity"),
                               F.round(F.sum("line_total_usd"), 2).alias("line_total_usd"),
                               F.count("*").alias("n_lines")))

first_session = sessions.groupBy("customer_id").agg(F.min("start_time").alias("first_session_ts"))
customers = (read("customers").drop("name", "email")
             .withColumn("signup_date", F.to_date("signup_date"))
             .join(first_session, "customer_id", "left")
             .withColumn("signup_after_first_session",
                         F.coalesce(F.col("first_session_ts") < F.col("signup_date").cast("timestamp"), F.lit(False))))

reviews_raw = read("reviews").withColumn("review_time", F.to_timestamp("review_time", TS_FMT))
reviews = (reviews_raw
           .withColumn("_r", F.row_number().over(W.partitionBy("order_id", "product_id").orderBy("review_id")))
           .filter("_r = 1").drop("_r")
           .join(orders.select("order_id", "order_time"), "order_id", "left")
           .withColumn("review_before_order", F.col("review_time") < F.to_date("order_time").cast("timestamp"))
           .drop("order_time"))

# ---------------------------------------------------------------- write
# "year" must lead the sort key: Spark >= 3.4 planned writes require ordering by the partition column and
# would otherwise insert its own sort on (year) that discards the (event_ts, event_id) order.
(good.select(*EVENT_COLS).repartition("year").sortWithinPartitions("year", "event_ts", "event_id")
 .write.mode("overwrite").partitionBy("year").parquet(out("events")))
bad.write.mode("overwrite").parquet(out("quarantine/events"))
session_labels.write.mode("overwrite").partitionBy("split").parquet(out("sessions"))
products.write.mode("overwrite").parquet(out("products"))
orders_out.write.mode("overwrite").parquet(out("orders"))
order_items.write.mode("overwrite").parquet(out("order_items"))
order_items_by_product.write.mode("overwrite").parquet(out("order_items_by_product"))
customers.write.mode("overwrite").parquet(out("customers"))
reviews.write.mode("overwrite").parquet(out("reviews"))
os.makedirs(out("vocab"), exist_ok=True)
with open(out("vocab/category.json"), "w") as f:
    json.dump({"PAD": 0, **cat_idx}, f, indent=2)

# ---------------------------------------------------------------- data quality report
def anti(left, right, on):
    return left.join(right.select(*on).distinct(), on, "left_anti").count()

for n in SCHEMAS:
    report["rows_raw"][n] = read(n).count()

inv = report["invariants"]
inv["fk_events_session"] = anti(raw_events, sessions, ["session_id"])
inv["fk_events_product"] = anti(events.filter("product_id IS NOT NULL"), products, ["product_id"])
inv["fk_sessions_customer"] = anti(sessions, customers, ["customer_id"])
inv["fk_orders_customer"] = anti(orders, customers, ["customer_id"])
inv["fk_items_order"] = anti(order_items, orders, ["order_id"])
inv["fk_items_product"] = anti(order_items, products, ["product_id"])
inv["fk_reviews_order_product"] = anti(reviews_raw, order_items, ["order_id", "product_id"])
inv["events_quarantined"] = bad.count()
inv["rule_violations"] = {r.rule: r.n for r in bad.select(F.explode("dq_failed").alias("rule"))
                          .groupBy("rule").agg(F.count("*").alias("n")).collect()}
inv["orders_without_purchase_event"] = orders_out.filter("session_id IS NULL").count()
inv["orders_matched_by_multiple_purchases"] = (events.filter("order_id IS NOT NULL").groupBy("order_id").count()
                                               .filter("count > 1").count())
sub = order_items.groupBy("order_id").agg(F.sum("line_total_usd").alias("s"))
inv["order_subtotal_mismatch"] = (orders.join(sub, "order_id", "left")
                                  .filter(F.abs(F.coalesce("s", F.lit(0.0)) - F.col("subtotal_usd")) > 0.02).count())
inv["order_total_mismatch"] = orders.filter(
    F.abs(F.round(F.col("subtotal_usd") * (1 - F.col("discount_pct") / 100), 2) - F.col("total_usd")) > 0.011).count()
inv["time_order_ne_logical_order_after_fix"] = (good.withColumn(
    "seq_by_time", F.row_number().over(W.partitionBy("session_id").orderBy("event_ts", "event_id")))
    .filter("seq_by_time != seq_no").count())
inv["purchase_not_last_event"] = (session_labels.filter("label_purchase = 1")
                                  .join(good.filter("event_type = 'purchase'").select("session_id", F.col("seq_no").alias("p")),
                                        "session_id").filter("p != n_events").count())

report["fixes"]["event_ts_clamped"] = events.filter("ts_fixed").count()
report["fixes"]["event_ts_clamped_by_type"] = {r.event_type: r["count"] for r in
                                               events.filter("ts_fixed").groupBy("event_type").count().collect()}
report["fixes"]["reviews_duplicate_dropped"] = report["rows_raw"]["reviews"] - reviews.count()
report["info"]["order_items_identical_lines_kept"] = report["rows_raw"]["order_items"] - order_items.dropDuplicates(
    ["order_id", "product_id", "unit_price_usd", "quantity", "line_total_usd"]).count()
report["info"]["order_lines_same_product"] = order_items_by_product.filter("n_lines > 1").count()
report["info"]["customers_signup_after_first_session"] = customers.filter("signup_after_first_session").count()
report["info"]["reviews_before_order"] = reviews.filter("review_before_order").count()
report["info"]["session_labels"] = {f"{r.split}": {"sessions": r.n, "conv_rate": round(r.cr, 4)} for r in
                                    session_labels.groupBy("split").agg(F.count("*").alias("n"),
                                                                        F.avg("label_purchase").alias("cr")).collect()}
report["info"]["category_vocab"] = {"PAD": 0, **cat_idx}

HARD = [k for k in inv if k != "rule_violations"]
failed = [k for k in HARD if inv[k] != 0]
report["status"] = "PASS" if not failed else f"FAIL: {failed}"
with open(out("dq_report.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
spark.stop()
sys.exit(1 if (failed and args.strict) else 0)
