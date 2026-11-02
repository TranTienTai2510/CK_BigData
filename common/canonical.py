"""Canonical (multi-shop) event schema.

One schema for every shop. It is used by the batch adapters, the simulator payload (Kafka value),
Spark Structured Streaming (from_json schema) and ClickHouse.

Design decisions
- Composite key: (shop_id, session_id) and (shop_id, event_id). IDs are only unique within a shop.
- Event types: page_view, add_to_cart, remove_from_cart, checkout, purchase.
  Not every shop emits every type: REES46 has no checkout; synthetic has no remove_from_cart.
- Item events (page_view / add_to_cart / remove_from_cart) carry scalar product fields
  (product_id, category, brand, price).
- purchase is ORDER-level: one event per order with order_id, amount_usd and an items array.
  This is the GA4-style ecommerce model. REES46 logs one purchase row per product, so the
  adapter groups those rows into one order per (session, timestamp).
- seq_no (1..n per session) is the logical order. Consumers order by seq_no, never by timestamp.
- event_ts is the corrected event time (always <= ts_raw). ts_raw is the time as given by the source.
- All timestamps are UTC.
- Money: amount_usd / price are USD. A shop that sells in another currency also fills currency and
  amount_local (Olist: BRL). NULL currency means the source is USD (or treated as USD).

Second table, ORDER_EVENT_SCHEMA (order lifecycle, Olist only for now): order_placed -> payment_approved ->
handed_to_carrier -> delivered -> review_created. One row per stage that has a timestamp. Here seq_no is the
stage order, and event_ts is a running max over the stages, so event_ts >= ts_raw: the purchase time is
trusted and a later stage that the source dates earlier is moved forward.
"""
from pyspark.sql import SparkSession, functions as F, types as T, Window as W

SCHEMA_VERSION = 3
SHOPS = ("synthetic", "rees46_cosmetics", "olist")
EVENT_TYPES = ("page_view", "add_to_cart", "remove_from_cart", "checkout", "purchase")
ITEM_EVENTS = ("page_view", "add_to_cart", "remove_from_cart")
TYPE_RANK = {t: i for i, t in enumerate(EVENT_TYPES)}  # tie-break for identical timestamps (funnel order)
PARTITION_COLS = ["shop_id", "ym"]

ITEM_STRUCT = T.StructType([
    T.StructField("product_id", T.LongType()),
    T.StructField("category", T.StringType()),
    T.StructField("brand", T.StringType()),
    T.StructField("price", T.DoubleType()),
    T.StructField("qty", T.IntegerType()),
])

EVENT_SCHEMA = T.StructType([
    T.StructField("shop_id", T.StringType()),
    T.StructField("event_id", T.LongType()),
    T.StructField("session_id", T.LongType()),
    T.StructField("seq_no", T.IntegerType()),
    T.StructField("customer_id", T.LongType()),
    T.StructField("event_type", T.StringType()),
    # item-level fields (item events only)
    T.StructField("product_id", T.LongType()),
    T.StructField("category", T.StringType()),
    T.StructField("brand", T.StringType()),
    T.StructField("price", T.DoubleType()),
    T.StructField("qty", T.IntegerType()),
    # checkout
    T.StructField("cart_size", T.IntegerType()),
    # purchase (order-level)
    T.StructField("order_id", T.LongType()),
    T.StructField("items", T.ArrayType(ITEM_STRUCT)),
    T.StructField("payment", T.StringType()),
    T.StructField("discount_pct", T.IntegerType()),
    T.StructField("amount_usd", T.DoubleType()),
    T.StructField("currency", T.StringType()),        # NULL = USD
    T.StructField("amount_local", T.DoubleType()),    # order total in `currency`
    # session context (null when the source has none)
    T.StructField("device", T.StringType()),
    T.StructField("source", T.StringType()),
    T.StructField("country", T.StringType()),
    # time
    T.StructField("session_start_ts", T.TimestampType()),
    T.StructField("event_ts", T.TimestampType()),
    T.StructField("ts_raw", T.TimestampType()),
    T.StructField("ts_fixed", T.BooleanType()),
    T.StructField("ym", T.StringType()),  # partition column, yyyy-MM of event_ts
])
EVENT_COLS = [f.name for f in EVENT_SCHEMA.fields]
REQUIRED = {"shop_id", "event_id", "session_id", "seq_no", "customer_id", "event_type", "event_ts", "ts_raw"}

ORDER_EVENT_TYPES = ("order_placed", "payment_approved", "handed_to_carrier", "delivered", "review_created")
ORDER_EVENT_SCHEMA = T.StructType([
    T.StructField("shop_id", T.StringType()),
    T.StructField("event_id", T.LongType()),
    T.StructField("order_id", T.LongType()),
    T.StructField("order_key", T.StringType()),          # original id in the source
    T.StructField("customer_id", T.LongType()),
    T.StructField("seq_no", T.IntegerType()),            # stage order inside the order (1 = order_placed)
    T.StructField("event_type", T.StringType()),
    T.StructField("order_status", T.StringType()),       # final status in the source snapshot
    T.StructField("n_items", T.IntegerType()),
    T.StructField("amount_usd", T.DoubleType()),
    T.StructField("currency", T.StringType()),
    T.StructField("amount_local", T.DoubleType()),
    T.StructField("customer_state", T.StringType()),
    T.StructField("estimated_delivery_ts", T.TimestampType()),
    T.StructField("days_late", T.DoubleType()),          # delivered only: (delivered - estimated) / 1 day
    T.StructField("review_score", T.IntegerType()),      # review_created only
    T.StructField("event_ts", T.TimestampType()),
    T.StructField("ts_raw", T.TimestampType()),
    T.StructField("ts_fixed", T.BooleanType()),
    T.StructField("ym", T.StringType()),
])
ORDER_EVENT_COLS = [f.name for f in ORDER_EVENT_SCHEMA.fields]
ORDER_REQUIRED = {"shop_id", "event_id", "order_id", "customer_id", "seq_no", "event_type", "event_ts", "ts_raw"}

# Session-level split for the DL module, by session start (left-closed, right-open, UTC).
SPLITS = {
    "synthetic": [("train", None, "2024-01-01"), ("val", "2024-01-01", "2025-01-01"), ("test", "2025-01-01", None)],
    "rees46_cosmetics": [("train", None, "2020-01-01"), ("val", "2020-01-01", "2020-02-01"),
                         ("test", "2020-02-01", None)],
}


def get_spark(app, driver_memory="4g", cores=4, shuffle_partitions=64):
    """Local Spark tuned for a 16 GB laptop (run while the streaming stack is down)."""
    from common.runtime_env import configure
    configure()   # JDK 17 + winutils/hadoop.dll on Windows, whatever launched the script (terminal, VS Code, ...)
    spark = (SparkSession.builder.appName(app)
             .master(f"local[{cores}]")
             .config("spark.driver.memory", driver_memory)
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
             .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))  # keeps generated ids stable
             .config("spark.sql.csv.parser.columnPruning.enabled", "false")      # consistent corrupt-record detection
             .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             # Planned writes (Spark >= 3.4) drop the constant shop_id from our sort, decide the data is not
             # ordered by the partition columns and add their own sort, which breaks event_ts order inside
             # files (measured: 45 inversions in 3 months; 0 with this off).
             .config("spark.sql.optimizer.plannedWrite.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def type_rank_col(col="event_type"):
    m = F.create_map(*[x for t, r in TYPE_RANK.items() for x in (F.lit(t), F.lit(r))])
    return m[F.col(col)]


def conform(df, schema=EVENT_SCHEMA, required=REQUIRED):
    """Cast/select to the canonical column order. Missing optional columns become typed NULLs."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required canonical columns: {sorted(missing)}")
    if "ym" not in df.columns:
        df = df.withColumn("ym", F.date_format("event_ts", "yyyy-MM"))
    cols = [(F.col(f.name).cast(f.dataType) if f.name in df.columns else F.lit(None).cast(f.dataType)).alias(f.name)
            for f in schema.fields]
    return df.select(*cols)


def read_table(spark, out_dir, name):
    """Read a unified table across shops. mergeSchema: shops written by older adapter versions may lack
    newer optional columns (e.g. currency)."""
    import os
    return spark.read.option("mergeSchema", "true").parquet(os.path.join(out_dir, name))


def split_col(shop, ts_col="session_start_ts"):
    expr = None
    for name, lo, hi in SPLITS[shop]:
        cond = F.lit(True)
        if lo:
            cond = cond & (F.col(ts_col) >= F.lit(lo).cast("timestamp"))
        if hi:
            cond = cond & (F.col(ts_col) < F.lit(hi).cast("timestamp"))
        expr = F.when(cond, name) if expr is None else expr.when(cond, name)
    return expr


def build_sessions(events, shop):
    """One row per (shop_id, session_id) with labels used by training and online evaluation.

    cut_seq_no: first checkout or purchase. A DL prefix may only use events with seq_no < cut_seq_no.
    """
    w = W.partitionBy("shop_id", "session_id").orderBy("seq_no")
    gaps = events.withColumn("gap_s", F.col("event_ts").cast("long") - F.lag(F.col("event_ts").cast("long")).over(w))
    is_t = lambda t: (F.col("event_type") == t).cast("int")  # noqa: E731
    return (gaps.groupBy("shop_id", "session_id").agg(
                F.min("customer_id").alias("customer_id"),
                F.countDistinct("customer_id").alias("n_customers"),
                F.count("*").alias("n_events"),
                F.sum(is_t("page_view")).alias("n_views"),
                F.sum(is_t("add_to_cart")).alias("n_add"),
                F.sum(is_t("remove_from_cart")).alias("n_remove"),
                F.max(is_t("checkout")).alias("has_checkout"),
                F.sum(is_t("purchase")).alias("n_orders"),
                F.max(is_t("purchase")).alias("label_purchase"),
                F.min(F.when(F.col("event_type").isin("checkout", "purchase"), F.col("seq_no"))).alias("cut_seq_no"),
                F.min(F.when(F.col("event_type") == "purchase", F.col("seq_no"))).alias("first_purchase_seq_no"),
                F.round(F.sum("amount_usd"), 2).alias("revenue_usd"),
                F.min("session_start_ts").alias("session_start_ts"),
                F.min("event_ts").alias("first_event_ts"),
                F.max("event_ts").alias("last_event_ts"),
                F.max("gap_s").alias("max_gap_s"),
                F.first("device", ignorenulls=True).alias("device"),
                F.first("source", ignorenulls=True).alias("source"),
                F.first("country", ignorenulls=True).alias("country"))
            .withColumn("duration_s", F.col("last_event_ts").cast("long") - F.col("first_event_ts").cast("long"))
            .withColumn("purchase_is_last", F.when(F.col("label_purchase") == 1,
                                                   F.col("first_purchase_seq_no") == F.col("n_events")))
            .withColumn("split", split_col(shop)))


def write_sorted(df, out_dir, name, shop, n_files):
    """Write one shop's rows to <out>/<name>/shop_id=<shop>/ym=.../ (other shops are left untouched).

    Files inside one (shop_id, ym) directory are globally sorted by (event_ts, event_id) when read in
    file-name order: range partitioning gives ordered tasks, each task sorts locally, and planned writes
    are disabled in get_spark() so Spark does not re-sort. check_unified.py verifies this.
    """
    import os, shutil
    shutil.rmtree(os.path.join(out_dir, name, f"shop_id={shop}"), ignore_errors=True)
    (df.repartitionByRange(n_files, "shop_id", "ym", "event_ts", "event_id")
       .sortWithinPartitions("shop_id", "ym", "event_ts", "event_id")
       .write.mode("append").partitionBy(*PARTITION_COLS).parquet(os.path.join(out_dir, name)))


def write_events(df, out_dir, shop, n_files):
    write_sorted(df, out_dir, "events", shop, n_files)


def write_by_shop(df, out_dir, name, shop):
    import os, shutil
    shutil.rmtree(os.path.join(out_dir, name, f"shop_id={shop}"), ignore_errors=True)
    df.write.mode("append").partitionBy("shop_id").parquet(os.path.join(out_dir, name))
