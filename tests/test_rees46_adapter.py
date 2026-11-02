"""Compare the REES46 adapter DQ report + output against the fixture's independent ground truth.

Run after:  python tests/make_rees46_fixture.py && python batch/adapters/rees46.py --raw tests/fixture_rees46 ...
  python tests/test_rees46_adapter.py --unified data/unified --fixture tests/fixture_rees46
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import functions as F  # noqa: E402
from common.canonical import get_spark  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--unified", default="data/unified")
ap.add_argument("--fixture", default="tests/fixture_rees46")
a = ap.parse_args()
gt = json.load(open(os.path.join(a.fixture, "ground_truth.json")))
rep = json.load(open(os.path.join(a.unified, "_dq", "rees46_cosmetics.json")))
sp = get_spark("test-rees46", cores=2, shuffle_partitions=8)
e = sp.read.parquet(os.path.join(a.unified, "events")).filter("shop_id = 'rees46_cosmetics'")
o = e.filter("event_type = 'purchase'")

fv = e.filter("event_type = 'page_view'").groupBy("session_id", "product_id").agg(F.min("seq_no").alias("v"))
fa = e.filter("event_type = 'add_to_cart'").groupBy("session_id", "product_id").agg(F.min("seq_no").alias("c"))
fr = e.filter("event_type = 'remove_from_cart'").groupBy("session_id", "product_id").agg(F.min("seq_no").alias("r"))

checks = {
    "status": (rep["status"], "PASS"),
    "quarantined": (rep["quarantined"]["total"], gt["expected_quarantined"]),
    "duplicates_collapsed": (rep["duplicates_collapsed"]["total"], gt["duplicates_collapsed"]),
    "rows_out": (rep["rows_out"], gt["rows_out"]),
    "sessions": (rep["entities"]["sessions"], gt["sessions"]),
    "source_user_session_ids": (rep["sessionization"]["source_user_session_ids"], gt["source_user_session_ids"]),
    "source_session_user_pairs": (rep["sessionization"]["source_session_user_pairs"], gt["source_session_user_pairs"]),
    "sessions_added_by_gap_split": (rep["sessionization"]["sessions_added_by_gap_split"],
                                    gt["sessions_added_by_gap_split"]),
    "orders": (rep["orders"]["n_orders"], gt["orders"]),
    "sessions_with_purchase": (rep["orders"]["sessions_with_purchase"], gt["sessions_with_purchase"]),
    "sessions_multi_order": (rep["orders"]["sessions_with_multiple_orders"], gt["sessions_multi_order"]),
    "purchase_not_last_pct": (rep["orders"]["purchase_not_last_event_pct"],
                              round(gt["purchase_not_last"] / gt["sessions_with_purchase"] * 100, 2)),
    "single_event_pct": (rep["session_shape"]["single_event_pct"],
                         round(gt["single_event_sessions"] / gt["sessions"] * 100, 2)),
    "qty_gt1_add_to_cart": (rep["qty_gt1"]["add_to_cart"], gt.get("qty_gt1_add_to_cart", 0)),
    "qty_gt1_remove_from_cart": (rep["qty_gt1"]["remove_from_cart"], gt.get("qty_gt1_remove_from_cart", 0)),
    "qty_gt1_purchase_items": (rep["qty_gt1"]["purchase_items"], gt.get("qty_gt1_purchase_items", 0)),
    "purchase_items": (o.select(F.sum(F.size("items"))).first()[0], gt["purchase_items"]),
    "cross_month_sessions": (e.groupBy("session_id").agg(F.countDistinct("ym").alias("k")).filter("k > 1").count(),
                             gt["cross_month_sessions"]),
    "cart_before_its_view": (fa.join(fv, ["session_id", "product_id"], "left").filter("v IS NULL OR c < v").count(), 0),
    "remove_before_its_cart": (fr.join(fa, ["session_id", "product_id"], "left").filter("c IS NULL OR r < c").count(), 0),
    "amount_ne_sum_items": (o.filter(F.abs(F.aggregate("items", F.lit(0.0), lambda acc, i: acc + i.price * i.qty)
                                           - F.col("amount_usd")) > 0.011).count(), 0),
    "time_range_utc_start": (rep["time_range_utc"][0][:10], "2019-10-25"),
}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
for k, (got, exp) in checks.items():
    print(f"{'OK ' if (got == exp) else 'BAD'} {k:28s} got={got} expected={exp}")
print("TEST", "PASS" if not bad else f"FAIL {list(bad)}")
sp.stop()
sys.exit(1 if bad else 0)
