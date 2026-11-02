"""Generate an Olist-format fixture (9 CSVs, original headers incl. the 'lenght' typos and a BOM in the
translation file) with known quirks, plus ground_truth.json computed in pure Python (zoneinfo for DST),
independent of Spark.

Quirks: orders without items (canceled/unavailable), multi-unit lines (same product, order_item_id 1..k),
multi-payment orders, an order without payments, not_defined payments of 0, carrier-before-approval and
delivered-before-carrier timestamps, delivered status without delivery date, reviews with multi-line quoted
comments, review_id shared by two orders, orders with two reviews, reviews created before delivery,
products without category, a category without translation, geolocation points outside Brazil, several
order-level customer_ids for one customer_unique_id, orders inside and outside Brazilian DST.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests/fixture_olist")
ap.add_argument("--orders", type=int, default=2000)
ap.add_argument("--brl-per-usd", type=float, default=3.4)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)
SP = ZoneInfo("America/Sao_Paulo")
F = "%Y-%m-%d %H:%M:%S"
hx = lambda: "%032x" % rng.getrandbits(128)  # noqa: E731
to_utc = lambda s: datetime.strptime(s, F).replace(tzinfo=SP).astimezone(timezone.utc).replace(tzinfo=None)  # noqa: E731

# ---------------------------------------------------------------- dims
cats = ["beleza_saude", "esporte_lazer", "informatica_acessorios", "moveis_decoracao", "pc_gamer"]
trans = {"beleza_saude": "health_beauty", "esporte_lazer": "sports_leisure",
         "informatica_acessorios": "computers_accessories", "moveis_decoracao": "furniture_decor"}  # pc_gamer: none
products = []
for i in range(300):
    cat = "" if i < 6 else rng.choice(cats)
    products.append({"product_id": hx(), "product_category_name": cat, "product_name_lenght": rng.randint(20, 60),
                     "product_description_lenght": rng.randint(100, 2000), "product_photos_qty": rng.randint(1, 5),
                     "product_weight_g": rng.randint(100, 5000), "product_length_cm": rng.randint(10, 60),
                     "product_height_cm": rng.randint(2, 40), "product_width_cm": rng.randint(10, 50),
                     "_price": round(rng.uniform(10, 500), 2)})
zips = [f"{z:05d}" for z in rng.sample(range(1000, 99999), 120)]
geo_rows, outliers = [], 0
for z in zips:
    for _ in range(rng.randint(1, 4)):
        geo_rows.append([z, round(rng.uniform(-30, -3), 6), round(rng.uniform(-60, -35), 6), "cidade", "SP"])
for z in zips[:8]:   # outside Brazil (bad source rows)
    geo_rows.append([z, 40.0, -3.7, "madrid", "SP"])
    outliers += 1
sellers = [{"seller_id": hx(), "seller_zip_code_prefix": rng.choice(zips), "seller_city": "sao paulo",
            "seller_state": "SP"} for _ in range(60)]
persons = [hx() for _ in range(1500)]

# ---------------------------------------------------------------- orders
orders, items, payments, reviews, customers = [], [], [], [], []
base_days = [datetime(2017, 6, 15, 10, 0), datetime(2017, 12, 15, 10, 0), datetime(2018, 3, 1, 9, 0)]
for k in range(a.orders):
    oid, cid = hx(), hx()
    customers.append({"customer_id": cid, "customer_unique_id": rng.choice(persons),
                      "customer_zip_code_prefix": rng.choice(zips), "customer_city": "x", "customer_state": "RJ"})
    t0 = rng.choice(base_days) + timedelta(days=rng.randint(0, 40), seconds=rng.randint(0, 86000))
    status = "delivered" if k >= 120 else ["canceled", "unavailable", "shipped", "invoiced", "processing",
                                          "approved", "created", "canceled"][k % 8]
    appr = t0 + timedelta(minutes=rng.randint(5, 600)) if status not in ("created",) else None
    carr = (appr + timedelta(days=rng.randint(1, 4))) if status in ("delivered", "shipped") else None
    deliv = (carr + timedelta(days=rng.randint(2, 20))) if status == "delivered" else None
    est = t0.replace(hour=0, minute=0, second=0) + timedelta(days=rng.randint(10, 30))
    if 200 <= k < 230:                      # carrier date before approval
        carr = appr - timedelta(hours=rng.randint(1, 20))
    if 230 <= k < 250:                      # delivered before carrier
        deliv = carr - timedelta(hours=rng.randint(1, 30))
    if 250 <= k < 255:                      # delivered status, no delivery date
        deliv = None
    fmt = lambda d: d.strftime(F) if d else ""  # noqa: E731
    orders.append({"order_id": oid, "customer_id": cid, "order_status": status, "order_purchase_timestamp": fmt(t0),
                   "order_approved_at": fmt(appr), "order_delivered_carrier_date": fmt(carr),
                   "order_delivered_customer_date": fmt(deliv), "order_estimated_delivery_date": fmt(est)})
    o = orders[-1]
    no_items = status in ("canceled", "unavailable") and k % 2 == 0
    total = 0.0
    if not no_items:
        n = 1
        for p in rng.sample(products, rng.choice([1, 1, 1, 2, 3])):
            units = rng.choice([1, 1, 1, 2, 3])
            s = rng.choice(sellers)
            for _ in range(units):
                fr = round(rng.uniform(5, 40), 2)
                items.append({"order_id": oid, "order_item_id": n, "product_id": p["product_id"],
                              "seller_id": s["seller_id"], "shipping_limit_date": fmt(t0 + timedelta(days=3)),
                              "price": p["_price"], "freight_value": fr})
                total += p["_price"] + fr
                n += 1
    total = round(total, 2)
    o["_total"] = total
    if k == 300:
        o["_payments"] = 0                  # order without payment rows
    elif status == "canceled" and no_items and k % 3 == 0:
        payments.append({"order_id": oid, "payment_sequential": 1, "payment_type": "not_defined",
                         "payment_installments": 1, "payment_value": 0.0})
        o["_payments"] = 1
    elif k % 10 == 0 and total > 20:        # voucher + card
        v = round(total * 0.3, 2)
        payments.append({"order_id": oid, "payment_sequential": 1, "payment_type": "voucher",
                         "payment_installments": 1, "payment_value": v})
        payments.append({"order_id": oid, "payment_sequential": 2, "payment_type": "credit_card",
                         "payment_installments": rng.randint(1, 10), "payment_value": round(total - v, 2)})
        o["_payments"] = 2
    else:
        payments.append({"order_id": oid, "payment_sequential": 1, "payment_type": rng.choice(["credit_card", "boleto"]),
                         "payment_installments": 1, "payment_value": total})
        o["_payments"] = 1
    if status in ("delivered", "shipped", "canceled") and k % 7 != 0:
        rc_base = deliv or est
        rc = (rc_base - timedelta(days=3)) if (k % 50 == 1) else (rc_base + timedelta(days=1))
        rc = rc.replace(hour=0, minute=0, second=0)
        msg = 'Produto "ótimo",\nchegou rápido' if k % 13 == 0 else ("" if k % 2 else "bom")
        reviews.append({"review_id": hx(), "order_id": oid, "review_score": rng.randint(1, 5),
                        "review_comment_title": "", "review_comment_message": msg,
                        "review_creation_date": fmt(rc), "review_answer_timestamp": fmt(rc + timedelta(hours=30))})
        if k % 97 == 5:                     # second, later-answered review for the same order (this one wins)
            reviews.append({"review_id": hx(), "order_id": oid, "review_score": 5, "review_comment_title": "",
                            "review_comment_message": "atualizado", "review_creation_date": fmt(rc + timedelta(days=2)),
                            "review_answer_timestamp": fmt(rc + timedelta(days=3))})
# review_id shared by two orders (source quirk): copy the id of review 0 onto review 1
reviews[1]["review_id"] = reviews[0]["review_id"]


# ---------------------------------------------------------------- write CSVs (Olist headers)
def write(name, header, rows, bom=False):
    with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for r in rows:
            w.writerow([r[h] if isinstance(r, dict) else r[i] for i, h in enumerate(header)])


ocols = ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
         "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
write("olist_orders_dataset.csv", ocols, orders)
write("olist_order_items_dataset.csv", ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date",
                                        "price", "freight_value"], items)
write("olist_order_payments_dataset.csv", ["order_id", "payment_sequential", "payment_type", "payment_installments",
                                           "payment_value"], payments)
write("olist_order_reviews_dataset.csv", ["review_id", "order_id", "review_score", "review_comment_title",
                                          "review_comment_message", "review_creation_date", "review_answer_timestamp"],
      reviews)
write("olist_customers_dataset.csv", ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city",
                                      "customer_state"], customers)
write("olist_products_dataset.csv", ["product_id", "product_category_name", "product_name_lenght",
                                     "product_description_lenght", "product_photos_qty", "product_weight_g",
                                     "product_length_cm", "product_height_cm", "product_width_cm"], products)
write("olist_sellers_dataset.csv", ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"], sellers)
write("olist_geolocation_dataset.csv", ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
                                        "geolocation_city", "geolocation_state"], geo_rows)
write("product_category_name_translation.csv", ["product_category_name", "product_category_name_english"],
      [[k, v] for k, v in trans.items()], bom=True)

# ---------------------------------------------------------------- ground truth (pure Python)
items_by_order = Counter(i["order_id"] for i in items)
pairs = Counter((i["order_id"], i["product_id"]) for i in items)
best_review = {}
for r in reviews:
    key = (r["review_answer_timestamp"], r["review_creation_date"])
    if r["order_id"] not in best_review or key > best_review[r["order_id"]][0]:
        best_review[r["order_id"]] = (key, r)
oe = Counter()
fixed = Counter()
rev_before_deliv = 0
for o in orders:
    stages = [("order_placed", o["order_purchase_timestamp"]), ("payment_approved", o["order_approved_at"]),
              ("handed_to_carrier", o["order_delivered_carrier_date"]),
              ("delivered", o["order_delivered_customer_date"])]
    run = None
    for name, s in stages:
        if s:
            t = to_utc(s)
            ev = t if run is None else max(run, t)
            run = ev
            oe[name] += 1
            fixed[name] += ev != t
    r = best_review.get(o["order_id"])
    if r:
        oe["review_created"] += 1
        rt = to_utc(r[1]["review_creation_date"])
        fixed["review_created"] += rt < to_utc(o["order_purchase_timestamp"])
        if o["order_delivered_customer_date"] and rt < to_utc(o["order_delivered_customer_date"]):
            rev_before_deliv += 1

sample = {}
for o in orders:   # one order in DST, one outside, with items and one payment row
    t = datetime.strptime(o["order_purchase_timestamp"], F)
    tag = "dst" if t.month in (12, 1) else ("std" if t.month in (6, 7) else None)
    if tag and tag not in sample and items_by_order[o["order_id"]] and o["_payments"] == 1 and o["order_delivered_customer_date"]:
        d_utc, e_utc = to_utc(o["order_delivered_customer_date"]), to_utc(o["order_estimated_delivery_date"])
        sample[tag] = {"order_key": o["order_id"], "purchase_utc": to_utc(o["order_purchase_timestamp"]).strftime(F),
                       "amount_brl": o["_total"], "amount_usd": round(o["_total"] / a.brl_per_usd, 2),
                       "days_late": round((d_utc - e_utc).total_seconds() / 86400, 2)}

gt = {
    "orders": len(orders), "orders_with_items": sum(1 for o in orders if items_by_order[o["order_id"]]),
    "orders_without_items": sum(1 for o in orders if not items_by_order[o["order_id"]]),
    "orders_without_payments": sum(1 for o in orders if o["_payments"] == 0),
    "item_rows": len(items), "pairs": len(pairs), "pairs_qty_gt1": sum(1 for v in pairs.values() if v > 1),
    "payments_rows": len(payments), "not_defined_payments": sum(1 for p in payments if p["payment_type"] == "not_defined"),
    "reviews_rows": len(reviews), "reviews_kept": len(best_review),
    "orders_with_multiple_reviews": sum(1 for v in Counter(r["order_id"] for r in reviews).values() if v > 1),
    "review_id_shared": 1, "reviews_before_delivery": rev_before_deliv,
    "order_events": dict(oe), "order_events_ts_fixed": {k: v for k, v in fixed.items() if v},
    "null_category_products": sum(1 for p in products if not p["product_category_name"]),
    "untranslated_categories": sorted({p["product_category_name"] for p in products
                                       if p["product_category_name"] and p["product_category_name"] not in trans}),
    "geo_outside_brazil": outliers, "zip_prefixes": len(zips),
    "unique_persons": len({c["customer_unique_id"] for c in customers}),
    "status_delivered_without_delivered_ts": sum(1 for o in orders if o["order_status"] == "delivered"
                                                 and not o["order_delivered_customer_date"]),
    "carrier_before_approved": sum(1 for o in orders if o["order_delivered_carrier_date"] and o["order_approved_at"]
                                   and o["order_delivered_carrier_date"] < o["order_approved_at"]),
    "delivered_before_carrier": sum(1 for o in orders if o["order_delivered_customer_date"] and
                                    o["order_delivered_carrier_date"] and
                                    o["order_delivered_customer_date"] < o["order_delivered_carrier_date"]),
    "samples": sample, "brl_per_usd": a.brl_per_usd,
}
with open(os.path.join(a.out, "ground_truth.json"), "w") as f:
    json.dump(gt, f, indent=2, sort_keys=True)
print(json.dumps(gt, indent=2, sort_keys=True))
