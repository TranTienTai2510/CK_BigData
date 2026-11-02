"""Scale the small REES46 fixture to real size (~20M rows, 5 monthly files) by streaming copies.

Each copy k gets a distinct user_session (last 12 hex digits replaced) and a time shift of k * step,
so copies spread over Oct 2019 - Feb 2020. Memory stays constant (line-by-line).
Expected counts scale linearly: every per-copy ground-truth count * copies.
"""
import argparse
import csv
import gzip
import os
from datetime import datetime, timedelta

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="tests/fixture_rees46")
ap.add_argument("--out", default="tests/fixture_rees46_scaled")
ap.add_argument("--copies", type=int, default=180)
ap.add_argument("--step-hours", type=float, default=20.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
MONTHS = {(2019, 10): "2019-Oct", (2019, 11): "2019-Nov", (2019, 12): "2019-Dec", (2020, 1): "2020-Jan",
          (2020, 2): "2020-Feb", (2020, 3): "2020-Mar"}
header = ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price", "user_id",
          "user_session"]


def src_rows():
    for name in sorted(os.listdir(a.src)):
        if not (name.endswith(".csv") or name.endswith(".csv.gz")):
            continue
        op = gzip.open if name.endswith(".gz") else open
        with op(os.path.join(a.src, name), "rt", newline="") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield row


rows = list(src_rows())  # ~115K rows, small
writers, handles = {}, {}
for k in range(a.copies):
    shift = timedelta(hours=k * a.step_hours)
    tag = f"{k:012x}"
    for row in rows:
        out = list(row)
        try:
            t = datetime.strptime(row[0][:19], "%Y-%m-%d %H:%M:%S") + shift
            out[0] = t.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
            key = (t.year, t.month)
        except ValueError:
            key = (2019, 10)
        if len(out) >= 9 and out[8]:
            out[8] = out[8][:-12] + tag
        name = MONTHS.get(key, "2020-Mar")
        if name not in writers:
            handles[name] = open(os.path.join(a.out, f"{name}.csv"), "w", newline="")
            writers[name] = csv.writer(handles[name])
            writers[name].writerow(header)
        writers[name].writerow(out)
for h in handles.values():
    h.close()
print({n: os.path.getsize(os.path.join(a.out, f"{n}.csv")) for n in writers})
