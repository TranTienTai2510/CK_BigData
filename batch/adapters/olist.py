"""Adapter: Olist "Brazilian E-Commerce Public Dataset" (Kaggle olistbr/brazilian-ecommerce) -> unified layer.

Olist has no clickstream (no views, carts or sessions). It is the third shop and adds:
  1. events/shop_id=olist        : ONE canonical `purchase` event per order that has items (order-level KPIs
                                   across all shops: orders/min, revenue, basket size). Never used by the
                                   clickstream DL model.
  2. order_events/shop_id=olist  : the order lifecycle (ORDER_EVENT_SCHEMA), for the delivery-SLA dashboard:
                                   order_placed -> payment_approved -> handed_to_carrier -> delivered -> review_created
  3. olist/<table>               : cleaned dimensions (orders, order_items, payments, reviews, products, sellers,
                                   customers, geolocation) for joins, analytics and reports.

Run:
  python batch/adapters/olist.py --raw data/raw_olist --out data/unified --strict [--fx-csv data/fx/DEXBZUS.csv]

Cleaning rules (each is counted in _dq/olist.json):
- CSVs read as strings, BOM stripped from headers, reviews parsed multiLine (comments contain newlines/quotes).
- Timestamps are local Brazil time (assumption, --source-tz America/Sao_Paulo, DST-aware) -> UTC.
- customer_id in Olist is per ORDER; the person is customer_unique_id -> canonical customer_id = hash of it.
- Ids are 32-hex strings -> canonical LONG ids = xxhash64('olist', kind, key). The original key is kept.
- order_items: one row per unit (order_item_id = 1..n) -> collapsed to (order, product) with qty.
- Category: English translation, Portuguese name when no translation exists, 'unknown' when NULL.
- Lifecycle timestamps out of order (e.g. carrier date before approval): stages 1-4 use a running max,
  so the purchase time is trusted and a stage the source dates earlier is moved forward (ts_fixed=true).
  review_created is not a delivery stage: it may precede delivery, it is only clamped to >= order_placed.
- Reviews: one per order (latest answer, then latest creation, then review_id). Duplicated review_id
  across orders is reported.
- Geolocation: points outside Brazil's bounding box dropped, median lat/lng per zip prefix.
- Money: amount_local = sum(payment_value) in BRL (fallback: items price + freight when an order has no
  payment rows). amount_usd = amount_local / BRL-per-USD: monthly mean from --fx-csv (FRED DEXBZUS format:
  date,value), else the constant --brl-per-usd (approximate; the report says which was used).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pyspark.sql import functions as F, Window as W  # noqa: E402
from common.canonical import (get_spark, conform, write_sorted, write_by_shop, read_table,  # noqa: E402
                              ORDER_EVENT_SCHEMA, ORDER_REQUIRED)
from common.contract import CANONICAL_RULES, ORDER_EVENT_RULES, failed_rules_col  # noqa: E402

SHOP = "olist"
FILES = {
    "orders": ("olist_orders_dataset.csv", ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
               "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"]),
    "items": ("olist_order_items_dataset.csv", ["order_id", "order_item_id", "product_id", "seller_id",
              "shipping_limit_date", "price", "freight_value"]),
    "payments": ("olist_order_payments_dataset.csv", ["order_id", "payment_sequential", "payment_type",
                 "payment_installments", "payment_value"]),
    "reviews": ("olist_order_reviews_dataset.csv", ["review_id", "order_id", "review_score", "review_comment_title",
                "review_comment_message", "review_creation_date", "review_answer_timestamp"]),
    "customers": ("olist_customers_dataset.csv", ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                  "customer_city", "customer_state"]),
    "products": ("olist_products_dataset.csv", ["product_id", "product_category_name", "product_weight_g",
                 "product_length_cm", "product_height_cm", "product_width_cm"]),
    "sellers": ("olist_sellers_dataset.csv", ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]),
    "geo": ("olist_geolocation_dataset.csv", ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
            "geolocation_city", "geolocation_state"]),
    "translation": ("product_category_name_translation.csv", ["product_category_name", "product_category_name_english"]),
}
BR_BBOX = (-33.75, 5.27, -73.99, -34.79)   # lat_min, lat_max, lng_min, lng_max
TS_FMT = "yyyy-MM-dd HH:mm:ss"

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw_olist")
ap.add_argument("--out", default="data/unified")
ap.add_argument("--source-tz", default="America/Sao_Paulo")
ap.add_argument("--fx-csv", default=None, help="daily BRL per USD, 2 columns (date,value), e.g. FRED DEXBZUS")
ap.add_argument("--brl-per-usd", type=float, default=3.4, help="fallback constant (approximate 2016-2018 level)")
ap.add_argument("--core-start", default="2017-01-01", help="local date; earlier months are sparse (2016: ~330 orders)")
ap.add_argument("--core-end", default="2018-09-01", help="local date, exclusive; Sep-Oct 2018 hold ~20 orders")
ap.add_argument("--driver-memory", default="3g")
ap.add_argument("--cores", type=int, default=4)
ap.add_argument("--strict", action="store_true")
args = ap.parse_args()
spark = get_spark("adapter-olist", args.driver_memory, args.cores, shuffle_partitions=16)


# ---------------------------------------------------------------- read
def read(name):
    fname, cols = FILES[name]
    path = os.path.join(args.raw, fname)
    if not os.path.exists(path):
        sys.exit(f"missing input file: {path}")
    df = (spark.read.option("header", True).option("multiLine", name == "reviews")
          .option("quote", '"').option("escape", '"').csv(path))
    df = df.toDF(*[c.lstrip("﻿").strip() for c in df.columns])
    miss = [c for c in cols if c not in df.columns]
    if miss:
        sys.exit(f"{fname}: missing columns {miss}; found {df.columns}")
    return df


raw = {n: read(n) for n in FILES}
rows_raw = {FILES[n][0]: raw[n].count() for n in FILES}
local_ts = lambda c: F.to_utc_timestamp(F.to_timestamp(F.col(c), TS_FMT), args.source_tz)  # noqa: E731
oid = lambda kind, c: F.xxhash64(F.lit(SHOP), F.lit(kind), F.col(c))  # noqa: E731
fmt = lambda c: F.date_format(c, TS_FMT)  # noqa: E731
dq = {"rows_raw": rows_raw}

# ---------------------------------------------------------------- FX (BRL per USD by month)
if args.fx_csv:
    fx_raw = spark.read.option("header", True).csv(args.fx_csv)
    dcol, vcol = fx_raw.columns[0], fx_raw.columns[1]
    fx = (fx_raw.select(F.to_date(F.col(dcol)).alias("d"), F.col(vcol).cast("double").alias("v"))
          .filter("d IS NOT NULL AND v > 0")
          .groupBy(F.date_format("d", "yyyy-MM").alias("fx_ym")).agg(F.avg("v").alias("brl_per_usd")))
    fx_source = f"monthly mean of {os.path.basename(args.fx_csv)}"
else:
    fx = spark.createDataFrame([], "fx_ym STRING, brl_per_usd DOUBLE")
    fx_source = f"constant {args.brl_per_usd} BRL/USD (approximate; pass --fx-csv for monthly rates)"

# ---------------------------------------------------------------- geolocation + dims
lat_min, lat_max, lng_min, lng_max = BR_BBOX
g = raw["geo"].select(F.col("geolocation_zip_code_prefix").alias("zip"),
                      F.col("geolocation_lat").cast("double").alias("lat"),
                      F.col("geolocation_lng").cast("double").alias("lng"))
in_br = F.col("lat").between(lat_min, lat_max) & F.col("lng").between(lng_min, lng_max)
dq["geolocation"] = {"rows": rows_raw[FILES["geo"][0]], "outside_brazil_dropped": g.filter(~in_br).count()}
geo = (g.filter(in_br).groupBy("zip")
       .agg(F.percentile_approx("lat", 0.5).alias("lat"), F.percentile_approx("lng", 0.5).alias("lng"),
            F.count("*").alias("n_points")))
dq["geolocation"]["zip_prefixes"] = geo.count()

customers = (raw["customers"]
             .withColumn("customer_uid", oid("customer", "customer_unique_id"))
             .join(geo.withColumnRenamed("zip", "customer_zip_code_prefix"), "customer_zip_code_prefix", "left"))
sellers = (raw["sellers"].join(geo.withColumnRenamed("zip", "seller_zip_code_prefix"), "seller_zip_code_prefix", "left"))
dq["customers"] = {"order_level_ids": customers.count(),
                   "unique_persons": customers.select("customer_unique_id").distinct().count(),
                   "without_geo": customers.filter("lat IS NULL").count()}
dq["sellers"] = {"rows": sellers.count(), "without_geo": sellers.filter("lat IS NULL").count()}

tr = raw["translation"]
products = (raw["products"].join(tr, "product_category_name", "left")
            .withColumn("category", F.coalesce("product_category_name_english", "product_category_name",
                                               F.lit("unknown")))
            .withColumn("product_uid", oid("product", "product_id")))
dq["products"] = {
    "rows": products.count(),
    "null_category": products.filter("product_category_name IS NULL").count(),
    "untranslated_categories": sorted(r.c for r in products.filter(
        "product_category_name IS NOT NULL AND product_category_name_english IS NULL")
        .select(F.col("product_category_name").alias("c")).distinct().collect()),
}

# ---------------------------------------------------------------- orders
o = (raw["orders"]
     .withColumn("purchase_ts", local_ts("order_purchase_timestamp"))
     .withColumn("approved_ts", local_ts("order_approved_at"))
     .withColumn("carrier_ts", local_ts("order_delivered_carrier_date"))
     .withColumn("delivered_ts", local_ts("order_delivered_customer_date"))
     .withColumn("estimated_ts", local_ts("order_estimated_delivery_date")))
bad_ts = {c: o.filter(F.col(src).isNotNull() & F.col(c).isNull()).count() for c, src in
          [("purchase_ts", "order_purchase_timestamp"), ("approved_ts", "order_approved_at"),
           ("carrier_ts", "order_delivered_carrier_date"), ("delivered_ts", "order_delivered_customer_date"),
           ("estimated_ts", "order_estimated_delivery_date")]}
dq["orders"] = {
    "rows": o.count(),
    "duplicate_order_id": o.count() - o.select("order_id").distinct().count(),
    "unparsable_timestamps": bad_ts,
    "by_status": {r.order_status: r["count"] for r in o.groupBy("order_status").count().collect()},
    "timestamp_anomalies_raw": {
        "approved_before_purchase": o.filter("approved_ts < purchase_ts").count(),
        "carrier_before_approved": o.filter("carrier_ts < approved_ts").count(),
        "carrier_before_purchase": o.filter("carrier_ts < purchase_ts").count(),
        "delivered_before_carrier": o.filter("delivered_ts < carrier_ts").count(),
        "delivered_before_purchase": o.filter("delivered_ts < purchase_ts").count(),
        "status_delivered_without_delivered_ts": o.filter("order_status = 'delivered' AND delivered_ts IS NULL").count(),
        "status_canceled_but_delivered": o.filter("order_status = 'canceled' AND delivered_ts IS NOT NULL").count(),
    },
}
o = o.filter("purchase_ts IS NOT NULL").dropDuplicates(["order_id"])

# ---------------------------------------------------------------- items, payments, reviews
items_raw = (raw["items"]
             .withColumn("price", F.col("price").cast("double"))
             .withColumn("freight_value", F.col("freight_value").cast("double"))
             .withColumn("order_item_id", F.col("order_item_id").cast("int"))
             .withColumn("shipping_limit_ts", local_ts("shipping_limit_date")))
dq["order_items"] = {"rows": items_raw.count(),
                     "orphan_order": items_raw.join(o, "order_id", "left_anti").count(),
                     "orphan_product": items_raw.join(raw["products"], "product_id", "left_anti").count(),
                     "negative_price_or_freight": items_raw.filter("price < 0 OR freight_value < 0").count()}
lines = (items_raw.join(products.select("product_id", "product_uid", "category"), "product_id", "left")
         .groupBy("order_id", "product_id", "product_uid", "category")
         .agg(F.count("*").alias("qty"), F.avg("price").alias("unit_price_brl"),
              F.countDistinct("price").alias("n_prices"), F.sum("price").alias("lines_brl"),
              F.sum("freight_value").alias("freight_brl"), F.collect_set("seller_id").alias("sellers")))
dq["order_items"]["order_product_pairs"] = lines.count()
dq["order_items"]["pairs_with_qty_gt1"] = lines.filter("qty > 1").count()
dq["order_items"]["pairs_with_varying_unit_price"] = lines.filter("n_prices > 1").count()

pay = (raw["payments"].withColumn("payment_value", F.col("payment_value").cast("double"))
       .withColumn("payment_installments", F.col("payment_installments").cast("int")))
dq["payments"] = {"rows": pay.count(), "not_defined_type": pay.filter("payment_type = 'not_defined'").count(),
                  "zero_value": pay.filter("payment_value = 0").count(),
                  "orphan_order": pay.join(o, "order_id", "left_anti").count()}
pay_o = (pay.groupBy("order_id").agg(
    F.round(F.sum("payment_value"), 2).alias("paid_brl"), F.count("*").alias("n_payments"),
    F.max("payment_installments").alias("max_installments"),
    F.max_by("payment_type", "payment_value").alias("main_payment_type")))

rv = (raw["reviews"].withColumn("review_score_i", F.col("review_score").cast("int"))
      .withColumn("review_created_ts", local_ts("review_creation_date"))
      .withColumn("review_answer_ts", local_ts("review_answer_timestamp")))
valid_rv = F.col("review_score_i").between(1, 5) & F.col("review_created_ts").isNotNull() & F.col("order_id").isNotNull()
rv_ok = rv.filter(valid_rv)
reviews = (rv_ok.withColumn("_r", F.row_number().over(W.partitionBy("order_id").orderBy(
               F.col("review_answer_ts").desc_nulls_last(), F.col("review_created_ts").desc(), "review_id")))
           .filter("_r = 1").drop("_r"))
dq["reviews"] = {"rows_parsed": rv.count(), "invalid_rows": rv.filter(~valid_rv).count(),
                 "review_id_shared_by_orders": rv_ok.groupBy("review_id").agg(F.countDistinct("order_id").alias("k"))
                 .filter("k > 1").count(),
                 "orders_with_multiple_reviews": rv_ok.groupBy("order_id").count().filter("count > 1").count(),
                 "kept_one_per_order": reviews.count(),
                 "orphan_order": reviews.join(o, "order_id", "left_anti").count(),
                 "with_comment": reviews.filter("review_comment_message IS NOT NULL").count()}

# ---------------------------------------------------------------- enriched orders
items_o = (lines.groupBy("order_id").agg(
    F.array_sort(F.collect_list(F.struct(
        F.col("product_uid").alias("product_id"), F.col("category"), F.lit(None).cast("string").alias("brand"),
        F.col("unit_price_brl").alias("price_brl"), F.col("qty").cast("int").alias("qty")))).alias("items_brl"),
    F.sum("qty").alias("n_items"), F.round(F.sum("lines_brl"), 2).alias("items_brl_total"),
    F.round(F.sum("freight_brl"), 2).alias("freight_brl_total"),
    F.array_distinct(F.flatten(F.collect_list("sellers"))).alias("seller_ids")))
seller_geo = F.broadcast(sellers.select("seller_id", F.col("lat").alias("s_lat"), F.col("lng").alias("s_lng")))
hav = lambda la1, lo1, la2, lo2: 2 * 6371.0 * F.asin(F.sqrt(  # noqa: E731
    F.pow(F.sin(F.radians(la2 - la1) / 2), 2)
    + F.cos(F.radians(la1)) * F.cos(F.radians(la2)) * F.pow(F.sin(F.radians(lo2 - lo1) / 2), 2)))
dist = (items_raw.select("order_id", "seller_id").distinct().join(seller_geo, "seller_id", "left")
        .join(o.select("order_id", "customer_id"), "order_id")
        .join(customers.select("customer_id", F.col("lat").alias("c_lat"), F.col("lng").alias("c_lng")), "customer_id",
              "left")
        .groupBy("order_id").agg(F.round(F.avg(hav(F.col("s_lat"), F.col("s_lng"), F.col("c_lat"), F.col("c_lng"))), 1)
                                 .alias("distance_km")))

orders = (o.join(customers.select("customer_id", "customer_unique_id", "customer_uid", "customer_state",
                                  "customer_city", "customer_zip_code_prefix"), "customer_id", "left")
          .join(items_o, "order_id", "left").join(pay_o, "order_id", "left").join(dist, "order_id", "left")
          .join(reviews.select("order_id", F.col("review_score_i").alias("review_score"), "review_created_ts"),
                "order_id", "left")
          .withColumn("customer_uid", F.coalesce("customer_uid", oid("order_customer", "customer_id")))
          .withColumn("order_uid", oid("order", "order_id"))
          .withColumn("fx_ym", F.date_format("purchase_ts", "yyyy-MM"))
          .join(F.broadcast(fx), "fx_ym", "left")
          .withColumn("fx_fallback", F.col("brl_per_usd").isNull())
          .withColumn("brl_per_usd", F.coalesce("brl_per_usd", F.lit(args.brl_per_usd)))
          .withColumn("amount_brl", F.coalesce("paid_brl", F.round(F.col("items_brl_total") + F.col("freight_brl_total"), 2)))
          .withColumn("amount_usd", F.round(F.col("amount_brl") / F.col("brl_per_usd"), 2))
          .withColumn("items", F.transform("items_brl", lambda i: F.struct(
              i.product_id.alias("product_id"), i.category.alias("category"), i.brand.alias("brand"),
              F.round(i.price_brl / F.col("brl_per_usd"), 2).alias("price"), i.qty.alias("qty"))))
          .withColumn("delivery_days", F.round((F.col("delivered_ts").cast("long") - F.col("purchase_ts").cast("long"))
                                               / 86400, 2))
          .withColumn("days_late", F.round((F.col("delivered_ts").cast("long") - F.col("estimated_ts").cast("long"))
                                           / 86400, 2))
          .withColumn("is_late", F.col("days_late") > 0)
          # replay / dashboard window: dense months only (see --core-start/--core-end)
          .withColumn("in_core_window", F.col("purchase_ts").between(
              F.to_utc_timestamp(F.lit(args.core_start).cast("timestamp"), args.source_tz),
              F.to_utc_timestamp(F.lit(args.core_end).cast("timestamp"), args.source_tz) - F.expr("INTERVAL 1 SECOND")))
          .withColumn("payment_minus_items_brl", F.round(F.col("paid_brl") - F.col("items_brl_total")
                                                         - F.col("freight_brl_total"), 2))
          .cache())
dq["orders"]["without_items"] = orders.filter("n_items IS NULL").count()
dq["orders"]["without_payments"] = orders.filter("paid_brl IS NULL").count()
dq["orders"]["without_review"] = orders.filter("review_score IS NULL").count()
dq["orders"]["customer_id_not_in_customers"] = orders.filter("customer_unique_id IS NULL").count()
dq["orders"]["payment_ne_items_plus_freight_gt_1brl"] = orders.filter(F.abs("payment_minus_items_brl") > 1).count()
dq["orders"]["fx_fallback_orders"] = orders.filter("fx_fallback").count()

# ---------------------------------------------------------------- 1. canonical purchase events
purchases = (orders.filter("n_items IS NOT NULL")
             .select(F.lit(SHOP).alias("shop_id"),
                     F.xxhash64(F.lit(SHOP), F.lit("purchase"), F.col("order_id")).alias("event_id"),
                     F.xxhash64(F.lit(SHOP), F.lit("session"), F.col("order_id")).alias("session_id"),
                     F.lit(1).alias("seq_no"), F.col("customer_uid").alias("customer_id"),
                     F.lit("purchase").alias("event_type"), F.col("order_uid").alias("order_id"), "items",
                     F.col("main_payment_type").alias("payment"), "amount_usd",
                     F.lit("BRL").alias("currency"), F.col("amount_brl").alias("amount_local"),
                     F.lit("BR").alias("country"), F.col("purchase_ts").alias("session_start_ts"),
                     F.col("purchase_ts").alias("event_ts"), F.col("purchase_ts").alias("ts_raw"),
                     F.lit(False).alias("ts_fixed")))
write_sorted(conform(purchases), args.out, "events", SHOP, n_files=4)

# ---------------------------------------------------------------- 2. order lifecycle events
stages = F.array(*[F.struct(F.lit(k).alias("stage"), F.lit(n).alias("event_type"), F.col(c).alias("ts_raw"))
                   for k, (n, c) in enumerate([("order_placed", "purchase_ts"), ("payment_approved", "approved_ts"),
                                               ("handed_to_carrier", "carrier_ts"), ("delivered", "delivered_ts"),
                                               ("review_created", "review_created_ts")], start=1)])
wo = W.partitionBy("order_uid").orderBy("stage")
oe = (orders.select("order_id", "order_uid", "customer_uid", "order_status", "n_items", "amount_usd", "amount_brl",
                    "customer_state", "estimated_ts", "review_score", F.explode(stages).alias("s"))
      .select("*", "s.stage", "s.event_type", "s.ts_raw").drop("s")
      .filter("ts_raw IS NOT NULL")
      .withColumn("placed_ts", F.min(F.when(F.col("stage") == 1, F.col("ts_raw"))).over(W.partitionBy("order_uid")))
      .withColumn("run_max", F.max(F.when(F.col("stage") <= 4, F.col("ts_raw")))
                  .over(wo.rowsBetween(W.unboundedPreceding, W.currentRow)))
      .withColumn("event_ts", F.when(F.col("stage") <= 4, F.col("run_max"))
                  .otherwise(F.greatest("ts_raw", "placed_ts")))
      .withColumn("seq_no", F.row_number().over(wo)))
order_events = conform(
    oe.select(F.lit(SHOP).alias("shop_id"), F.xxhash64(F.lit(SHOP), F.col("order_id"), F.col("event_type")).alias("event_id"),
              F.col("order_uid").alias("order_id"), F.col("order_id").alias("order_key"),
              F.col("customer_uid").alias("customer_id"), "seq_no", "event_type", "order_status", "n_items",
              "amount_usd", F.lit("BRL").alias("currency"), F.col("amount_brl").alias("amount_local"),
              "customer_state", F.col("estimated_ts").alias("estimated_delivery_ts"),
              F.when(F.col("event_type") == "delivered",
                     F.round((F.col("event_ts").cast("long") - F.col("estimated_ts").cast("long")) / 86400, 2))
              .alias("days_late"),
              F.when(F.col("event_type") == "review_created", F.col("review_score")).alias("review_score"),
              "event_ts", "ts_raw", (F.col("event_ts") != F.col("ts_raw")).alias("ts_fixed")),
    schema=ORDER_EVENT_SCHEMA, required=ORDER_REQUIRED)
write_sorted(order_events, args.out, "order_events", SHOP, n_files=4)

# ---------------------------------------------------------------- 3. dimensions
dim = os.path.join(args.out, "olist")
orders.drop("items_brl", "items").write.mode("overwrite").parquet(os.path.join(dim, "orders"))
(items_raw.join(products.select("product_id", "product_uid", "category"), "product_id", "left")
 .withColumn("order_uid", oid("order", "order_id"))
 .write.mode("overwrite").parquet(os.path.join(dim, "order_items")))
pay.withColumn("order_uid", oid("order", "order_id")).write.mode("overwrite").parquet(os.path.join(dim, "payments"))
(reviews.select("review_id", "order_id", F.col("review_score_i").alias("review_score"), "review_comment_title",
                "review_comment_message", "review_created_ts", "review_answer_ts")
 .withColumn("order_uid", oid("order", "order_id")).write.mode("overwrite").parquet(os.path.join(dim, "reviews")))
products.write.mode("overwrite").parquet(os.path.join(dim, "products"))
sellers.write.mode("overwrite").parquet(os.path.join(dim, "sellers"))
customers.write.mode("overwrite").parquet(os.path.join(dim, "customers"))
geo.write.mode("overwrite").parquet(os.path.join(dim, "geolocation"))
write_by_shop(lines.join(orders.select("order_id", "brl_per_usd"), "order_id")
              .groupBy(F.lit(SHOP).alias("shop_id"), F.col("product_uid").alias("product_id"))
              .agg(F.first("category").alias("category"), F.lit(None).cast("string").alias("brand"),
                   F.round(F.percentile_approx(F.col("unit_price_brl") / F.col("brl_per_usd"), 0.5), 2).alias("price"),
                   F.round(F.min(F.col("unit_price_brl") / F.col("brl_per_usd")), 2).alias("price_min"),
                   F.round(F.max(F.col("unit_price_brl") / F.col("brl_per_usd")), 2).alias("price_max"),
                   F.sum("qty").alias("n_events")), args.out, "products", SHOP)

# ---------------------------------------------------------------- report
ev = read_table(spark, args.out, "events").filter(F.col("shop_id") == SHOP)
oe_out = read_table(spark, args.out, "order_events").filter(F.col("shop_id") == SHOP)
n_orders = orders.count()
q = lambda df, c: dict(zip(["p50", "p90", "p99", "max"], df.approxQuantile(c, [0.5, 0.9, 0.99, 1.0], 0.001)))  # noqa: E731
t = oe_out.select(fmt(F.min("event_ts")).alias("t0"), fmt(F.max("event_ts")).alias("t1")).first()
dq["fx"] = {"source": fx_source, "fallback_orders": dq["orders"]["fx_fallback_orders"]}
dq["output"] = {
    "purchase_events": ev.count(),
    "order_events": {r.event_type: r["count"] for r in oe_out.groupBy("event_type").count().collect()},
    "order_events_ts_fixed": {r.event_type: r["count"] for r in oe_out.filter("ts_fixed").groupBy("event_type").count()
                              .collect()},
    "reviews_before_delivery": orders.filter("review_created_ts < delivered_ts").count(),
    "time_range_utc": [t.t0, t.t1],
    "orders_per_month": {r.ym: r["count"] for r in ev.groupBy("ym").count().orderBy("ym").collect()},
    "amount_usd": q(ev, "amount_usd"), "items_per_order": q(ev.withColumn("k", F.size("items")), "k"),
    "delivery_days": q(orders.filter("delivered_ts IS NOT NULL"), "delivery_days"),
    "late_rate_delivered": round(orders.filter("delivered_ts IS NOT NULL").select(
        F.avg(F.col("is_late").cast("double"))).first()[0] or 0, 4),
    "review_score_mean": round(orders.select(F.avg("review_score")).first()[0] or 0, 3),
    "distance_km": q(orders.filter("distance_km IS NOT NULL"), "distance_km"),
    "core_window": {"local": [args.core_start, args.core_end], "orders_in": orders.filter("in_core_window").count(),
                    "orders_out": orders.filter("NOT in_core_window").count()},
    # right-censoring: orders placed close to the snapshot may not be delivered yet -> late rate is biased
    "delivered_share_last_months": {r.m: round(r.d, 4) for r in orders.groupBy(
        F.date_format(F.from_utc_timestamp("purchase_ts", args.source_tz), "yyyy-MM").alias("m"))
        .agg(F.avg(F.col("delivered_ts").isNotNull().cast("double")).alias("d")).orderBy(F.desc("m")).limit(6)
        .collect()},
    "ambiguous_dst_hour_orders": orders.filter(F.date_format(F.from_utc_timestamp("purchase_ts", args.source_tz),
                                                             "yyyy-MM-dd HH").isin("2017-02-18 23", "2018-02-17 23"))
                                       .count(),
}
inv = {
    "event_contract_violations": {r.rule: r["count"] for r in ev.select(
        F.explode(failed_rules_col(CANONICAL_RULES)).alias("rule")).groupBy("rule").count().collect()},
    "order_event_contract_violations": {r.rule: r["count"] for r in oe_out.select(
        F.explode(failed_rules_col(ORDER_EVENT_RULES)).alias("rule")).groupBy("rule").count().collect()},
    "purchase_events_ne_orders_with_items": ev.count() - orders.filter("n_items IS NOT NULL").count(),
    "order_placed_ne_orders": oe_out.filter("event_type = 'order_placed'").count() - n_orders,
    "item_units_ne_item_rows": (ev.select(F.sum(F.aggregate("items", F.lit(0), lambda a, i: a + i.qty))).first()[0] or 0)
                               - items_raw.join(o.select("order_id"), "order_id").count(),
    "event_id_duplicates": ev.count() - ev.select("event_id").distinct().count()
                           + oe_out.count() - oe_out.select("event_id").distinct().count(),
    "order_events_seq_not_contiguous": (oe_out.groupBy("order_id").agg(
        F.max("seq_no").alias("m"), F.count("*").alias("c"), F.countDistinct("seq_no").alias("k"))
        .filter("m != c OR k != c").count()),
    "lifecycle_time_not_monotonic": (oe_out.filter("event_type <> 'review_created'")
                                     .withColumn("p", F.lag("event_ts").over(W.partitionBy("order_id").orderBy("seq_no")))
                                     .filter("event_ts < p").count()),
}
dq["invariants"] = inv
failed = [k for k, v in inv.items() if (v if isinstance(v, int) else len(v)) != 0]
dq["status"] = "PASS" if not failed else f"FAIL: {failed}"
report = {"shop_id": SHOP, "adapter_version": 2, "source_tz": args.source_tz, **dq}
os.makedirs(os.path.join(args.out, "_dq"), exist_ok=True)
with open(os.path.join(args.out, "_dq", f"{SHOP}.json"), "w") as f:
    json.dump(report, f, indent=2, default=str, ensure_ascii=False)
print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
spark.stop()
sys.exit(1 if (failed and args.strict) else 0)
