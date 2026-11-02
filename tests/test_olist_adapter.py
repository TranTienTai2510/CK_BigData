"""Compare the Olist adapter report + output against the fixture's independent ground truth.

  python tests/make_olist_fixture.py --out tests/fixture_olist
  python batch/adapters/olist.py --raw tests/fixture_olist --out data/unified --strict
  python tests/test_olist_adapter.py --unified data/unified --fixture tests/fixture_olist
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import functions as F  # noqa: E402
from common.canonical import get_spark, read_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--unified", default="data/unified")
ap.add_argument("--fixture", default="tests/fixture_olist")
a = ap.parse_args()
gt = json.load(open(os.path.join(a.fixture, "ground_truth.json")))
r = json.load(open(os.path.join(a.unified, "_dq", "olist.json")))
sp = get_spark("test-olist", cores=2, shuffle_partitions=8)
ev = read_table(sp, a.unified, "events").filter("shop_id = 'olist'")
oe = read_table(sp, a.unified, "order_events").filter("shop_id = 'olist'")
rv = sp.read.parquet(os.path.join(a.unified, "olist", "reviews"))

checks = {
    "status": (r["status"], "PASS"),
    "orders": (r["orders"]["rows"], gt["orders"]),
    "purchase_events": (r["output"]["purchase_events"], gt["orders_with_items"]),
    "orders_without_items": (r["orders"]["without_items"], gt["orders_without_items"]),
    "orders_without_payments": (r["orders"]["without_payments"], gt["orders_without_payments"]),
    "item_rows": (r["order_items"]["rows"], gt["item_rows"]),
    "order_product_pairs": (r["order_items"]["order_product_pairs"], gt["pairs"]),
    "pairs_qty_gt1": (r["order_items"]["pairs_with_qty_gt1"], gt["pairs_qty_gt1"]),
    "not_defined_payments": (r["payments"]["not_defined_type"], gt["not_defined_payments"]),
    "reviews_parsed": (r["reviews"]["rows_parsed"], gt["reviews_rows"]),
    "reviews_kept": (r["reviews"]["kept_one_per_order"], gt["reviews_kept"]),
    "orders_with_multiple_reviews": (r["reviews"]["orders_with_multiple_reviews"], gt["orders_with_multiple_reviews"]),
    "review_id_shared": (r["reviews"]["review_id_shared_by_orders"], gt["review_id_shared"]),
    "reviews_before_delivery": (r["output"]["reviews_before_delivery"], gt["reviews_before_delivery"]),
    "order_events": (r["output"]["order_events"], gt["order_events"]),
    "order_events_ts_fixed": (r["output"]["order_events_ts_fixed"], gt["order_events_ts_fixed"]),
    "null_category": (r["products"]["null_category"], gt["null_category_products"]),
    "untranslated": (r["products"]["untranslated_categories"], gt["untranslated_categories"]),
    "geo_outside": (r["geolocation"]["outside_brazil_dropped"], gt["geo_outside_brazil"]),
    "zip_prefixes": (r["geolocation"]["zip_prefixes"], gt["zip_prefixes"]),
    "unique_persons": (r["customers"]["unique_persons"], gt["unique_persons"]),
    "delivered_without_ts": (r["orders"]["timestamp_anomalies_raw"]["status_delivered_without_delivered_ts"],
                             gt["status_delivered_without_delivered_ts"]),
    "carrier_before_approved": (r["orders"]["timestamp_anomalies_raw"]["carrier_before_approved"],
                                gt["carrier_before_approved"]),
    "delivered_before_carrier": (r["orders"]["timestamp_anomalies_raw"]["delivered_before_carrier"],
                                 gt["delivered_before_carrier"]),
    "multiline_comment_intact": (rv.filter(F.col("review_comment_message").contains('"ótimo",\nchegou')).count() > 0, True),
}
for tag, s in gt["samples"].items():   # DST-aware UTC conversion, FX, days_late on concrete orders
    row = (ev.join(oe.filter(f"order_key = '{s['order_key']}' AND event_type = 'delivered'")
                   .select("order_id", "days_late"), "order_id")
           .select(F.date_format("event_ts", "yyyy-MM-dd HH:mm:ss").alias("t"), "amount_local", "amount_usd",
                   "days_late").first())
    checks[f"{tag}_purchase_utc"] = (row.t, s["purchase_utc"])
    checks[f"{tag}_amount_brl"] = (round(row.amount_local, 2), s["amount_brl"])
    checks[f"{tag}_amount_usd"] = (row.amount_usd, s["amount_usd"])
    checks[f"{tag}_days_late"] = (row.days_late, s["days_late"])
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
for k, (got, exp) in checks.items():
    print(f"{'OK ' if got == exp else 'BAD'} {k:30s} got={got} expected={exp}")
print("TEST", "PASS" if not bad else f"FAIL {list(bad)}")
sp.stop()
sys.exit(1 if bad else 0)
