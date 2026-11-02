"""Generate a small REES46-format fixture with known quirks + ground truth, to test batch/adapters/rees46.py.

Format matches the Kaggle files: header
  event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
event_time like "2019-10-01 00:00:00 UTC", NULLs are empty fields, one file per month
(Nov is written gzipped to exercise .csv.gz input).

Injected quirks (ground truth in ground_truth.json, computed by an independent pure-Python
sessionizer, not by Spark):
  exact duplicate rows (cart/remove/purchase copies -> qty 2, view copies dropped), negative prices,
  NULL user_session, rows without category, unknown event type, corrupt rows, multi-item orders,
  purchases followed by browsing, sessions with 2 orders, same-second view->cart ties, sessions crossing
  the Oct/Nov boundary, zero prices, long inactivity gaps inside a user_session (must split), and
  user_session ids shared by two user_id (must split).
"""
import argparse
import csv
import gzip
import json
import os
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests/fixture_rees46")
ap.add_argument("--sessions", type=int, default=6000)
ap.add_argument("--gap-s", type=int, default=1800)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)

CATS = [1487580005000000000 + 1000 * k for k in range(30)]
CODES = {CATS[0]: "appliances.environment.vacuum", CATS[1]: "furniture.bathroom.bath", CATS[2]: "accessories.bag"}
products = []
for i in range(400):
    cid = rng.choice(CATS)
    price = 0.0 if i < 4 else round(rng.uniform(0.5, 80), 2)
    products.append((5_000_000 + i * 37, cid, CODES.get(cid, ""), "" if rng.random() < 0.4 else f"brand{i % 25}", price))
by_cat = {}
for p in products:
    by_cat.setdefault(p[1], []).append(p)

sessions_raw = []   # list of event lists: (t, kind, product, user, session_uuid)
t_lo, t_hi = datetime(2019, 10, 25), datetime(2019, 11, 5)
for s in range(a.sessions):
    user = 100_000 + rng.randrange(2000)
    sess = str(uuid.UUID(int=rng.getrandbits(128)))
    if s < 40:      # straddle the month boundary
        t = datetime(2019, 10, 31, 23, 40) + timedelta(seconds=rng.randrange(900))
    else:
        t = t_lo + timedelta(seconds=rng.randrange(int((t_hi - t_lo).total_seconds())))
    ev, cart = [], []
    cat = rng.choice(CATS)
    n_steps = rng.randint(4, 25) if 40 <= s < 140 else rng.randint(1, 25)
    for step in range(n_steps):
        if 40 <= s < 100 and step == n_steps // 2:
            t += timedelta(seconds=rng.choice([a.gap_s + 1, 2 * 3600, 3 * 86400]))   # long gap -> split
        if 100 <= s < 140 and step == n_steps // 2:
            user = 100_000 + 2000 + s                                               # shared id -> split
        if rng.random() < 0.3:
            cat = rng.choice(CATS)
        p = rng.choice(by_cat[cat])
        ev.append((t, "view", p, user, sess))
        if rng.random() < 0.3 and p not in cart:
            if rng.random() >= 0.3:
                t += timedelta(seconds=rng.randint(1, 20))
            ev.append((t, "cart", p, user, sess))
            cart.append(p)
            if rng.random() < 0.15:
                t += timedelta(seconds=rng.randint(0, 30))
                ev.append((t, "remove_from_cart", cart.pop(), user, sess))
        t += timedelta(seconds=rng.randint(1, 300))
        last_step = step == n_steps - 1
        if cart and (last_step or rng.random() < 0.03) and rng.random() < 0.45:
            for item in cart:
                ev.append((t, "purchase", item, user, sess))
            cart = []
            t += timedelta(seconds=rng.randint(1, 120))
            if last_step and rng.random() < 0.25:
                for _ in range(rng.randint(1, 3)):
                    ev.append((t, "view", rng.choice(products), user, sess))
                    t += timedelta(seconds=rng.randint(1, 120))
    sessions_raw.append(ev)


def fmt(r, price=None, session=None, cat=True, etype=None):
    t, kind, p, user, sess = r
    pid, cid, code, brand, pr = p
    return [t.strftime("%Y-%m-%d %H:%M:%S") + " UTC", etype or kind, pid, cid if cat else "", code if cat else "",
            brand, pr if price is None else price, user, sess if session is None else session]


rows = [r for ev in sessions_raw for r in ev]
base, seen = [], set()
for r in rows:
    k = tuple(map(str, fmt(r)))
    if k not in seen:
        seen.add(k)
        base.append(fmt(r))
dups = rng.sample(base, 700)
neg = [fmt(r, price=-round(rng.uniform(1, 50), 2)) for r in rng.sample(rows, 25)]
nos = [fmt(r, session="") for r in rng.sample(rows, 30)]
nocat = [fmt(r, cat=False) for r in rng.sample(rows, 5)]
unk = [fmt(r, etype="click") for r in rng.sample(rows, 4)]


# ---------------------------------------------------------------- independent ground truth (pure Python)
def ground_truth(valid_rows, gap_s):
    """Re-implements the adapter contract: collapse exact copies, split by (user_session, user_id, gap)."""
    counts = Counter(tuple(map(str, r)) for r in valid_rows)
    rank = {"view": 0, "cart": 1, "remove_from_cart": 2, "purchase": 4}
    by_key = {}
    for k, n in counts.items():
        t = datetime.strptime(k[0][:19], "%Y-%m-%d %H:%M:%S")
        by_key.setdefault((k[8], k[7]), []).append((t, rank[k[1]], k, n))
    g = Counter()
    for key, evs in by_key.items():
        evs.sort(key=lambda x: (x[0], x[1]))
        segs, cur = [], [evs[0]]
        for prev, e in zip(evs, evs[1:]):
            if (e[0] - prev[0]).total_seconds() > gap_s:
                segs.append(cur)
                cur = []
            cur.append(e)
        segs.append(cur)
        g["sessions_added_by_gap_split"] += len(segs) - 1
        for seg in segs:
            g["sessions"] += 1
            purchase_ts = sorted({e[0] for e in seg if e[2][1] == "purchase"})
            n_items = sum(1 for e in seg if e[2][1] == "purchase")
            n_canon = sum(1 for e in seg if e[2][1] != "purchase") + len(purchase_ts)
            g["rows_out"] += n_canon
            g["orders"] += len(purchase_ts)
            g["purchase_items"] += n_items
            g["single_event_sessions"] += n_canon == 1
            g["cross_month_sessions"] += seg[0][0].month != seg[-1][0].month
            if purchase_ts:
                g["sessions_with_purchase"] += 1
                g["sessions_multi_order"] += len(purchase_ts) > 1
                g["purchase_not_last"] += any(e[0] > purchase_ts[0] for e in seg)
            for e in seg:
                if e[3] > 1:
                    if e[2][1] == "cart":
                        g["qty_gt1_add_to_cart"] += 1
                    elif e[2][1] == "remove_from_cart":
                        g["qty_gt1_remove_from_cart"] += 1
                    elif e[2][1] == "purchase":
                        g["qty_gt1_purchase_items"] += 1
    g["source_user_session_ids"] = len({k[0] for k in by_key})
    g["source_session_user_pairs"] = len(by_key)
    g["duplicates_collapsed"] = sum(n - 1 for n in counts.values())
    return dict(g)


gt = ground_truth(base + dups, a.gap_s)
gt.update({"base_rows": len(base), "injected_duplicates": len(dups),
           "injected_duplicates_by_type": dict(Counter(r[1] for r in dups)),
           "expected_quarantined": 25 + 30 + 5 + 4 + 3, "zero_price_products": 4})

allrows = base + dups + neg + nos + nocat + unk
rng.shuffle(allrows)
allrows.sort(key=lambda r: r[0])
header = ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price", "user_id",
          "user_session"]
oct_rows = [r for r in allrows if r[0].startswith("2019-10")]
nov_rows = [r for r in allrows if not r[0].startswith("2019-10")]
with open(os.path.join(a.out, "2019-Oct.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(oct_rows[:1000])
    f.write("2019-10-30 10:00:00 UTC,view,5000000,1487580005000000000,,brandX,abc,100001,"
            "11111111-1111-1111-1111-111111111111\n")
    f.write("2019-10-30 10:00:01 UTC,cart,5000037,1487580005000001000,,,not_a_price,100002,"
            "22222222-2222-2222-2222-222222222222\n")
    f.write("2019-10-30 10:00:02 UTC,view,5000074\n")
    w.writerows(oct_rows[1000:])
with gzip.open(os.path.join(a.out, "2019-Nov.csv.gz"), "wt", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(nov_rows)
gt["file_rows"] = {"2019-Oct.csv": len(oct_rows) + 3, "2019-Nov.csv.gz": len(nov_rows)}
with open(os.path.join(a.out, "ground_truth.json"), "w") as f:
    json.dump(gt, f, indent=2, sort_keys=True)
print(json.dumps(gt, indent=2, sort_keys=True))
