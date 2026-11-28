#!/usr/bin/env python3
"""Simulator B — replay canonical Parquet (3 shop) to Redpanda.

Ownership: B (Platform & Streaming). Contract: common/canonical.py (schema v3),
common/contract.py (CANONICAL_RULES / ORDER_EVENT_RULES). Never edits contract.

Features (G1):
  - replay 3 shop: synthetic / rees46_cosmetics / olist (events + order_events)
  - time-warp: scale event-time gaps by --speed-factor, cap actual sleep by --max-sleep
  - shift future timestamps: rows with event_ts in the past are shifted so the
    first event starts at --start-offset-s seconds ago; order_events
    estimated_delivery_ts is shifted by the same delta (per requirement)
  - chaos: --late-pct (emit out-of-order late), --dup-pct (duplicate send)
  - stress: --fold N duplicates the stream N times with shifted keys
    (approx N x throughput, for --fold 1/5/10/20 test)
  - throughput target ~1100 events/s at default settings (tune speed-factor)

Topics (7 total, created by redpanda-init in docker-compose.yml):
  events.raw, order_events.raw, agg.kpi, pred.intent, session.closed,
  ops.metrics, events.dlq

Examples:
  python simulator.py --shops synthetic,rees46_cosmetics,olist --speed-factor 50
  python simulator.py --mode orders --shops olist --speed-factor 100
  python simulator.py --fold 5 --speed-factor 200 --late-pct 0.03 --dup-pct 0.02
  python simulator.py --limit 5000 --dry-run   # no Kafka, just validate payloads
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

LOG = logging.getLogger("simulator")
SCHEMA_VERSION = 3
INTEGER_FIELDS = (
    "event_id", "session_id", "order_id", "customer_id", "seq_no",
    "product_id", "qty", "cart_size", "discount_pct", "n_items", "review_score",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay unified Parquet to Redpanda (B).")
    p.add_argument("--unified", default="data/unified")
    p.add_argument("--shops", default="synthetic,rees46_cosmetics,olist",
                   help="Comma list of shop_id to replay.")
    p.add_argument("--mode", default="both", choices=["events", "orders", "both"])
    p.add_argument("--ym", default="", help="Comma list of ym=YYYY-MM to replay; empty = all.")
    p.add_argument("--bootstrap-server", default="localhost:19092")
    p.add_argument("--events-topic", default="events.raw")
    p.add_argument("--orders-topic", default="order_events.raw")
    p.add_argument("--speed-factor", type=float, default=50.0)
    p.add_argument("--max-sleep", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--fold", type=int, default=1, help="Stress multiplier: duplicate each event N times.")
    p.add_argument("--late-pct", type=float, default=0.03, help="Fraction sent out-of-order (late).")
    p.add_argument("--dup-pct", type=float, default=0.02, help="Fraction duplicated.")
    p.add_argument("--start-offset-s", type=float, default=60.0,
                   help="First replayed event starts this many seconds in the past.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Validate + print stats, no Kafka send.")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    if not math.isfinite(a.speed_factor) or a.speed_factor <= 0:
        p.error("--speed-factor must be > 0")
    if a.fold < 1:
        p.error("--fold must be >= 1")
    if not (0 <= a.late_pct <= 1 and 0 <= a.dup_pct <= 1):
        p.error("--late-pct/--dup-pct must be in [0,1]")
    return a


def _glob_files(base: str) -> list[str]:
    m = sorted(glob.glob(base))
    if not m and Path(base).exists():
        m = [base]
    return m


def collect_inputs(unified: str, table: str, shops: list[str], yms: list[str]) -> list[str]:
    out: list[str] = []
    for shop in shops:
        pat = os.path.join(unified, table, f"shop_id={shop}", "ym=*", "*.parquet")
        for f in sorted(glob.glob(pat)):
            if yms and f.split("ym=")[-1][:7] not in yms:
                continue
            out.append(f)
    return out


def to_json_value(v: Any) -> Any:
    import pandas as pd  # local import: simulator container has pandas
    if v is None or v is pd.NA:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime().astimezone(timezone.utc).isoformat(timespec="microseconds")
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).isoformat(timespec="microseconds")
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "tolist"):
        return to_json_value(v.tolist())
    if hasattr(v, "item"):
        return to_json_value(v.item())
    if isinstance(v, (list, tuple)):
        return [to_json_value(i) for i in v]
    if isinstance(v, dict):
        return {str(k): to_json_value(i) for k, i in v.items()}
    return v


def shift_dt(v: Any, delta: timedelta) -> Any:
    import pandas as pd
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(v)
    if ts is pd.NaT or pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) + delta


def _scaled_gap(previous: datetime, current: datetime, speed_factor: float,
                max_sleep: float) -> tuple[float, float]:
    event_gap = max((current - previous).total_seconds(), 0.0) / speed_factor
    sleep_gap = min(event_gap, max_sleep) if max_sleep > 0 else event_gap
    return event_gap, sleep_gap


def load_sorted(files: list[str], ts_col="event_ts", id_col="event_id", row_limit: int = 0):
    import pandas as pd
    frames = []
    remaining = row_limit
    for f in files:
        df = pd.read_parquet(f)
        if not df.empty:
            if "shop_id" not in df.columns:
                partition = next(
                    (part for part in reversed(Path(f).parts) if part.startswith("shop_id=")),
                    None,
                )
                if partition is None:
                    raise ValueError(f"shop_id column or partition is required: {f}")
                df["shop_id"] = partition.split("=", 1)[1]
            if remaining > 0:
                df = df.head(remaining)
                remaining -= len(df)
            frames.append(df)
            if remaining == 0:
                break
    if not frames:
        raise ValueError("No rows in selected Parquet inputs.")
    ev = pd.concat(frames, ignore_index=True)
    ev[ts_col] = pd.to_datetime(ev[ts_col], utc=True, errors="raise")
    return ev.sort_values([ts_col, id_col], kind="mergesort").reset_index(drop=True)


def build_payload(row, replay_ts: datetime, delta: timedelta) -> dict:
    payload = {str(k): to_json_value(v) for k, v in row.items()}
    for field in INTEGER_FIELDS:
        value = payload.get(field)
        if value is not None:
            integer = int(value)
            if integer != value:
                raise ValueError(f"{field} must be an integer, got {value!r}")
            payload[field] = integer
    for item in payload.get("items") or []:
        if isinstance(item, dict):
            for field in ("product_id", "qty"):
                value = item.get(field)
                if value is not None:
                    integer = int(value)
                    if integer != value:
                        raise ValueError(f"items[].{field} must be an integer, got {value!r}")
                    item[field] = integer
    payload["schema_version"] = SCHEMA_VERSION
    payload["event_ts"] = replay_ts.astimezone(timezone.utc).isoformat(timespec="microseconds")
    for col in ("ts_raw", "session_start_ts", "estimated_delivery_ts"):
        if col in payload and payload[col] is not None:
            shifted = shift_dt(row.get(col), delta)
            payload[col] = to_json_value(shifted)
    payload["ym"] = replay_ts.strftime("%Y-%m")
    if "product_id" in payload:
        payload["product"] = payload.get("product_id")
    return payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = random.Random(args.seed)
    shops = [s.strip() for s in args.shops.split(",") if s.strip()]
    yms = [y.strip() for y in args.ym.split(",") if y.strip()] if args.ym else []

    jobs: list[tuple[str, str]] = []
    if args.mode in ("events", "both"):
        files = collect_inputs(args.unified, "events", [s for s in shops], yms)
        if files:
            jobs.append(("events", args.events_topic))
    if args.mode in ("orders", "both"):
        ofiles = collect_inputs(args.unified, "order_events", ["olist"] if "olist" in shops else [], yms)
        if ofiles:
            jobs.append(("orders", args.orders_topic))
    if not jobs:
        raise SystemExit("No input files matched --shops/--ym/--mode selection.")

    producer = None
    if not args.dry_run:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=[s.strip() for s in args.bootstrap_server.split(",") if s.strip()],
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: json.dumps(v, separators=(",", ":"),
                                                  ensure_ascii=False, allow_nan=False).encode("utf-8"),
            acks="all", retries=10, max_in_flight_requests_per_connection=1,
        )

    sent = 0
    late_buf: list[tuple[dict, str]] = []
    pending = []

    def flush_pending():
        if producer is None or not pending:
            return
        producer.flush(timeout=30)
        for future in pending:
            future.get(timeout=30)
        pending.clear()

    def publish(topic, key, payload):
        nonlocal sent
        if producer is None:
            sent += 1
            return
        pending.append(producer.send(topic, key=key, value=payload))
        sent += 1
        if len(pending) >= 1000:
            flush_pending()

    try:
        while True:
            for kind, topic in jobs:
                files = collect_inputs(args.unified, "events" if kind == "events" else "order_events",
                                       shops if kind == "events" else ["olist"], yms)
                if not files:
                    continue
                ev = load_sorted(files, row_limit=args.limit)
                t0_hist = ev["event_ts"].iloc[0].to_pydatetime()
                replay_start = datetime.now(timezone.utc) - timedelta(seconds=args.start_offset_s)
                delta = replay_start - t0_hist.astimezone(timezone.utc)
                prev_hist = None
                sched = None
                for _, row in ev.iterrows():
                    hist = row["event_ts"].to_pydatetime()
                    if prev_hist is None:
                        sched = replay_start
                    else:
                        event_gap, sleep_gap = _scaled_gap(
                            prev_hist, hist, args.speed_factor, args.max_sleep
                        )
                        time.sleep(sleep_gap)
                        sched = sched + timedelta(seconds=event_gap)
                    payload = build_payload(row, sched, delta)
                    n_copies = args.fold
                    for f in range(n_copies):
                        p = dict(payload)
                        key_base = f"{p.get('shop_id')}:{p.get('session_id', p.get('order_id'))}"
                        key = f"{key_base}#f{f}" if args.fold > 1 else key_base
                        r = rng.random()
                        if r < args.dup_pct and producer is not None:
                            publish(topic, key, p)
                        if rng.random() < args.late_pct:
                            late_buf.append((p, key))  # flush a few later -> late events
                            if len(late_buf) >= 50 and producer is not None:
                                lp, lk = late_buf.pop(0)
                                publish(topic, lk, lp)
                                if args.limit and sent >= args.limit:
                                    flush_pending()
                                    LOG.info("Reached limit %d", args.limit)
                                    return
                            continue
                        publish(topic, key, p)
                        if args.limit and sent >= args.limit:
                            flush_pending()
                            LOG.info("Reached limit %d", args.limit)
                            return
                        if sent % 2000 == 0:
                            LOG.info("Published %d events (topic=%s fold=%d)", sent, topic, args.fold)
                    prev_hist, sched_prev = hist, sched
                if producer is not None:
                    while late_buf:  # drain lates at end of pass
                        lp, lk = late_buf.pop(0)
                        publish(topic, lk, lp)
                        if args.limit and sent >= args.limit:
                            flush_pending()
                            LOG.info("Reached limit %d", args.limit)
                            return
                    flush_pending()
                LOG.info("Pass done kind=%s topic=%s total=%d", kind, topic, sent)
                if args.limit and sent >= args.limit:
                    return
            if not args.loop:
                break
    finally:
        if producer is not None:
            producer.close(timeout=30)
    LOG.info("Replay finished; published %d events.", sent)


if __name__ == "__main__":
    main()
