"""G2 parity test: the same session through the BATCH path (ml/prefixes.py, Spark SQL) and through the STREAM
function (ml.features.SessionState, as used by Q2) must give the same tensors, bit for bit.

  python tests/test_features_parity.py                                  # built-in fixture (no data needed)
  python tests/test_features_parity.py --unified data/unified --real 200  # + 200 real sessions per shop

Stream side is fed like Kafka/Q2 would: micro-batches of random size, ~10% late (out-of-order) events, ~5%
duplicates, post-cut events (checkout/purchase and later), and the state goes through to_json/from_json between
micro-batches (Spark state store). Checks:
  - tensors: for every session and every prefix length k, stream tensor == batch tensor (x_cat, x_num, length, shop)
  - raw steps: stream steps == batch steps (exact ints / doubles)
  - emitted (seq_no, k) == 1..n_steps of batch, nothing at/after the cut, sessions without prefix emit nothing
  - session_tensors (training, vectorised) == window_to_tensor (stream) for every k
  - duplicates all detected, no gap skipped, emit="last" == last of emit="all", state JSON round trip
  - vocab from train only: categories seen only in val/test map to OOV
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from common.canonical import EVENT_SCHEMA, build_sessions, conform, get_spark, read_table  # noqa: E402
from ml import features as FT  # noqa: E402
from ml.prefixes import build_spec, build_steps, prefix_events  # noqa: E402

UTC = timezone.utc


# ---------------------------------------------------------------- fixture with known edge cases
def make_fixture(rng, n_sessions=240):
    rows, sid, eid = [], 0, 0
    shops = {
        "synthetic": dict(types=["page_view", "add_to_cart"], cut=["checkout", "purchase"],
                          splits=[datetime(2023, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC),
                                  datetime(2025, 6, 1, tzinfo=UTC)],
                          cats=[f"cat_{i}" for i in range(12)], brands=[None]),
        "rees46_cosmetics": dict(types=["page_view", "add_to_cart", "remove_from_cart"], cut=["purchase"],
                                 splits=[datetime(2019, 11, 10, tzinfo=UTC), datetime(2020, 1, 10, tzinfo=UTC),
                                         datetime(2020, 2, 10, tzinfo=UTC)],
                                 cats=[f"cid:{1000 + i}" for i in range(15)], brands=[None, "b1", "b2", "b3", "b4"]),
    }
    for shop, cfg in shops.items():
        for i in range(n_sessions):
            sid += 1
            split_i = i % 3                       # 0 train, 1 val, 2 test
            t = cfg["splits"][split_i] + timedelta(seconds=rng.randrange(20 * 86400))
            start = t
            kind = i % 12
            n = {0: 1, 1: 2, 2: 55, 3: 130}.get(kind, rng.randint(1, 30))
            cut_at = None
            if kind in (4, 5, 6):
                cut_at = rng.randint(1, n)        # purchase/checkout inside, more events after it
            elif kind == 7:
                cut_at = 1                        # cut at the first event: no prefix
            evs, prev_prod, seq = [], None, 0
            for j in range(n + (2 if cut_at else 0)):
                seq += 1
                if cut_at and seq == cut_at:
                    etype = rng.choice(cfg["cut"])
                elif cut_at and seq == cut_at + 1 and shop == "synthetic" and rng.random() < 0.5:
                    etype = "purchase"
                else:
                    etype = rng.choice(cfg["types"])
                gap = rng.choice([0, 0, 1, 3, 15, 60, 300, 1700])
                t = t + timedelta(seconds=gap, milliseconds=rng.choice([0, 0, 250]))
                if rng.random() < 0.03:
                    t = t - timedelta(seconds=5)          # non-monotonic time inside a session (dt clamped to 0)
                ev = dict(shop_id=shop, event_id=eid, session_id=sid, seq_no=seq, customer_id=10 * sid,
                          event_type=etype, event_ts=t, ts_raw=t, session_start_ts=start, ts_fixed=False)
                eid += 1
                if etype in FT.CUT_TYPES:
                    if etype == "checkout":
                        ev["cart_size"] = rng.randint(1, 4)
                    else:
                        ev.update(order_id=eid, amount_usd=12.5, items=[(1, "c", None, 12.5, 1)])
                else:
                    prod = prev_prod if (prev_prod and rng.random() < 0.3) else rng.randrange(1, 60)
                    cats = cfg["cats"] + (["only_in_test"] if split_i == 2 else [])
                    ev.update(product_id=prod, category=rng.choice(cats), brand=rng.choice(cfg["brands"]),
                              price=rng.choice([0.0, 1.99, 4.5, 19.0, 120.75]))
                    if etype in ("add_to_cart", "remove_from_cart"):
                        ev["qty"] = rng.choice([1, 1, 1, 2, 3])
                    prev_prod = prod
                evs.append(ev)
            rows += evs
    return rows


def fixture_df(spark, rows, tmp):
    """Rows -> canonical DataFrame through a JSON file (spark.createDataFrame needs Python workers, which
    PySpark 3.5 cannot start on Python 3.14). Timestamps travel as epoch ms."""
    import json
    path = os.path.join(tmp, "fixture_events.jsonl")
    ts_cols = ("event_ts", "ts_raw", "session_start_ts")
    item_keys = ("product_id", "category", "brand", "price", "qty")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            d = {k: v for k, v in r.items() if k not in ts_cols and k != "items"}
            d.update({f"{k}_ms": round(r[k].timestamp() * 1000) for k in ts_cols})
            if r.get("items"):
                d["items"] = [dict(zip(item_keys, it)) for it in r["items"]]
            f.write(json.dumps(d) + "\n")
    schema = ", ".join(f"{fl.name} {fl.dataType.simpleString()}" for fl in EVENT_SCHEMA.fields
                       if fl.name not in ts_cols + ("ym",)) + ", " + ", ".join(f"{k}_ms long" for k in ts_cols)
    df = spark.read.schema(schema).json(path)
    for k in ts_cols:
        df = df.withColumn(k, F.timestamp_millis(F.col(f"{k}_ms")))
    return conform(df)


# ---------------------------------------------------------------- stream simulation
def perturb(events, rng):
    """Out-of-order + duplicates + random micro-batches, like the simulator -> Kafka -> Q2 path."""
    evs = list(events)
    for i in range(len(evs)):                      # ~10% late: move an event up to 6 positions later
        if rng.random() < 0.10:
            j = min(len(evs) - 1, i + rng.randint(1, 6))
            evs.insert(j, evs.pop(i))
    n_dup = 0
    for i in range(len(evs) - 1, -1, -1):          # ~5% duplicates, re-delivered a bit later
        if rng.random() < 0.05:
            evs.insert(min(len(evs), i + rng.randint(1, 4)), evs[i])
            n_dup += 1
    batches, i = [], 0
    while i < len(evs):
        k = rng.randint(1, 7)
        batches.append(evs[i:i + k])
        i += k
    return batches, n_dup


def run_stream(shop, batches, spec, emit="all"):
    state_json, out = None, []
    for b in batches:
        st = FT.SessionState(shop) if state_json is None else FT.SessionState.from_json(state_json)
        out += st.update(b, spec, emit=emit)
        state_json = st.to_json()
        assert FT.SessionState.from_json(state_json).to_json() == state_json, "state JSON round trip"
    return out, FT.SessionState.from_json(state_json)


def all_steps_in_order(shop, events, spec):
    big = dict(spec, max_len=10 ** 9)              # window never truncates -> holds every raw step
    st = FT.SessionState(shop)
    st.update(sorted(events, key=lambda e: e["seq_no"]), big, emit="none")
    return st.window


def same(a, b):
    return all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in ("x_cat", "x_num"))


# ---------------------------------------------------------------- parity
def parity(spark, events, sessions, shops, min_count, rng, tmp, label):
    pre = prefix_events(events, sessions)
    spec = build_spec(pre, shops, min_count, 5000, 5000, {"test": label})
    path = os.path.join(tmp, f"prefix_{label}")
    build_steps(spark, pre, spec).write.mode("overwrite").parquet(path)       # Parquet round trip as in prod
    batch = {(r.shop_id, r.session_id): r for r in spark.read.parquet(path).collect()}
    stream_in = defaultdict(list)
    for r in FT.spark_event_input(events).collect():
        stream_in[(r.shop_id, r.session_id)].append(r.asDict())
    sess = {(r.shop_id, r.session_id): r for r in sessions.collect()}

    c = defaultdict(int)
    for key, evs in stream_in.items():
        shop = key[0]
        batches, n_dup = perturb(evs, rng)
        out, st = run_stream(shop, batches, spec)
        c["sessions"] += 1
        c["dup_missed"] += (st.n_dup != n_dup)
        c["skipped"] += st.n_skipped
        b = batch.get(key)
        s = sess[key]
        if b is None:                                  # no prefix (cut at seq 1) -> nothing emitted
            c["no_prefix_sessions"] += 1
            c["bad_no_prefix"] += bool(out) or s.cut_seq_no != 1
            continue
        steps = [tuple(x[f] for f in FT.STEP_FIELDS) for x in b.steps]
        c["bad_label_or_split"] += (b.label != s.label_purchase) or (b.split != s.split)
        c["bad_raw_steps"] += steps != all_steps_in_order(shop, evs, spec)
        c["bad_emitted_k"] += [(seq, k) for seq, k, _ in out] != [(x[0], i + 1) for i, x in enumerate(steps)]
        c["emit_at_or_after_cut"] += sum(1 for seq, _, _ in out if s.cut_seq_no and seq >= s.cut_seq_no)
        tens = FT.session_tensors(steps, b.shop_idx, max_len=spec["max_len"])
        for i, (seq, k, t) in enumerate(out[:len(steps)]):
            c["tensors_compared"] += 1
            bt = {f: tens[f][k - 1] for f in ("x_cat", "x_num", "length", "shop")}
            ok = same(t, bt) and t["length"] == bt["length"] and t["shop"] == bt["shop"]
            ok = ok and same(t, FT.window_to_tensor(steps[:k], b.shop_idx, spec["max_len"]))
            c["tensor_mismatch"] += not ok
        last, _ = run_stream(shop, batches, spec, emit="last")
        c["bad_emit_last"] += not all(same(t, FT.window_to_tensor(steps[:k], b.shop_idx, spec["max_len"]))
                                      for _, k, t in last) or not {s_ for s_, _, _ in last} <= {s_ for s_, _, _ in out}
        c["truncated_sessions"] += len(steps) > spec["max_len"]
        c["positives"] += b.label
    return c, spec, batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", help="also test real sessions sampled from this unified folder")
    ap.add_argument("--real", type=int, default=200, help="real sessions per shop (with --unified)")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    spark = get_spark("test-features-parity", cores=2, shuffle_partitions=8)
    tmp = tempfile.mkdtemp(prefix="parity_")
    checks, failed = {}, []
    try:
        # ---- unit checks
        nan_ev = {"seq_no": 1, "event_type": "page_view", "product_id": float("nan"), "category": None,
                  "brand": float("nan"), "price": float("nan"), "qty": float("nan"), "ts_ms": np.int64(5)}
        checks["nan_is_null"] = (FT._norm_event(nan_ev), ("page_view", None, None, None, None, None, 5))
        st = FT.SessionState("rees46_cosmetics")
        sp0 = FT.make_spec(["rees46_cosmetics"], [], [])
        st.update([dict(nan_ev, seq_no=s, ts_ms=s) for s in (1, 3, 4, 5)], sp0, max_pending=2)
        checks["gap_skipped_when_pending_full"] = (st.n_skipped, 1)

        # ---- fixture
        events = fixture_df(spark, make_fixture(rng), tmp).cache()
        shops = ["synthetic", "rees46_cosmetics"]
        sessions = None
        for shop in shops:
            s = build_sessions(events.filter(F.col("shop_id") == shop), shop)
            sessions = s if sessions is None else sessions.unionByName(s)
        sessions = sessions.cache()
        c, spec, batch = parity(spark, events, sessions, shops, 2, rng, tmp, "fixture")
        oov_test = sum(1 for (shop, _), r in batch.items() if r.split == "test"
                       for x in r.steps if x["cat_idx"] == FT.OOV_IDX)
        checks.update({
            "fixture: tensor_mismatch": (c["tensor_mismatch"], 0),
            "fixture: bad_raw_steps": (c["bad_raw_steps"], 0),
            "fixture: bad_emitted_k": (c["bad_emitted_k"], 0),
            "fixture: emit_at_or_after_cut": (c["emit_at_or_after_cut"], 0),
            "fixture: bad_no_prefix": (c["bad_no_prefix"], 0),
            "fixture: bad_label_or_split": (c["bad_label_or_split"], 0),
            "fixture: bad_emit_last": (c["bad_emit_last"], 0),
            "fixture: dup_missed": (c["dup_missed"], 0),
            "fixture: skipped": (c["skipped"], 0),
            "fixture: has no-prefix sessions": (c["no_prefix_sessions"] > 0, True),
            "fixture: has sessions > max_len": (c["truncated_sessions"] > 0, True),
            "fixture: has positives": (c["positives"] > 0, True),
            "fixture: test-only category not in vocab": (any("only_in_test" in k for k in spec["category"]), False),
            "fixture: test-only category -> OOV": (oov_test > 0, True),
        })
        print(f"fixture: {c['sessions']} sessions, {c['tensors_compared']} tensors compared")

        # ---- real sessions
        if a.unified:
            ev_all = read_table(spark, a.unified, "events")
            ss_all = read_table(spark, a.unified, "sessions")
            picks = None
            for shop in shops:
                s = ss_all.filter((F.col("shop_id") == shop) & (F.col("n_events") >= 2) & (F.col("n_events") <= 400))
                pre_len = F.coalesce(F.col("cut_seq_no") - 1, F.col("n_events"))
                long_ = s.filter(pre_len > 60).orderBy(F.xxhash64("session_id")).limit(max(1, a.real // 10))
                pos = s.filter("label_purchase = 1").orderBy(F.xxhash64("session_id", F.lit(1))).limit(a.real // 3)
                rnd = s.orderBy(F.xxhash64("session_id", F.lit(2))).limit(a.real)
                p = long_.unionByName(pos).unionByName(rnd).select("shop_id", "session_id").distinct()
                picks = p if picks is None else picks.unionByName(p)
            picks = picks.cache()
            r_events = ev_all.join(F.broadcast(picks), ["shop_id", "session_id"], "left_semi").cache()
            r_sess = ss_all.join(F.broadcast(picks), ["shop_id", "session_id"], "left_semi").cache()
            c, _, _ = parity(spark, r_events, r_sess, shops, 1, rng, tmp, "real")
            for k in ("tensor_mismatch", "bad_raw_steps", "bad_emitted_k", "emit_at_or_after_cut", "bad_no_prefix",
                      "bad_label_or_split", "bad_emit_last", "dup_missed", "skipped"):
                checks[f"real: {k}"] = (c[k], 0)
            checks["real: sessions tested"] = (c["sessions"] > 0, True)
            print(f"real: {c['sessions']} sessions ({c['positives']} purchasers, {c['truncated_sessions']} longer "
                  f"than {FT.MAX_LEN} steps), {c['tensors_compared']} tensors compared")
    finally:
        spark.stop()
        shutil.rmtree(tmp, ignore_errors=True)

    for k, (got, exp) in checks.items():
        ok = got == exp
        failed += [] if ok else [k]
        print(f"{'OK ' if ok else 'BAD'} {k:45s} got={got} expected={exp}")
    print("TEST", "PASS" if not failed else f"FAIL {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
