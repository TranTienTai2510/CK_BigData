"""Q2: Session State + Real-time Purchase Intent Inference via ONNX (A).

Reads canonical events (from events.clean or events.raw), maintains per-session state in strict
seq_no order using ml.features.SessionState, executes batched ONNX inference using models/intent_v1.onnx,
and emits:
  1. pred.intent: (shop_id, session_id, seq_no, event_ts, prob, model_ver)
  2. session.closed: (shop_id, session_id, last_seq_no, last_event_ts, label_purchase, revenue_usd)

Compatible with ClickHouse rt.intent, rt.session_closed, and rt.v_online_eval_1m.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from pyspark.sql import functions as F, types as T

from common.canonical import EVENT_SCHEMA
from ml.features import (
    EVENT_INPUT_COLS,
    SessionState,
    load_spec,
    shop_index,
)

MODEL_VER = "intent_v1"

# Target schemas matching ClickHouse & Kafka contracts
INTENT_SCHEMA = T.StructType([
    T.StructField("shop_id", T.StringType(), False),
    T.StructField("session_id", T.LongType(), False),
    T.StructField("seq_no", T.IntegerType(), False),
    T.StructField("event_ts", T.StringType(), False),
    T.StructField("prob", T.FloatType(), False),
    T.StructField("model_ver", T.StringType(), False),
])

CLOSED_SCHEMA = T.StructType([
    T.StructField("shop_id", T.StringType(), False),
    T.StructField("session_id", T.LongType(), False),
    T.StructField("last_seq_no", T.IntegerType(), False),
    T.StructField("last_event_ts", T.StringType(), False),
    T.StructField("label_purchase", T.IntegerType(), False),
    T.StructField("revenue_usd", T.FloatType(), False),
])


def resolve_path(relative_or_abs):
    """Find file in project root or container workdir."""
    candidates = [
        relative_or_abs,
        os.path.join(ROOT_DIR, relative_or_abs),
        os.path.join("/opt/project", relative_or_abs),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[1])


class ModelInferenceWorker:
    """Singleton per worker process to cache the ONNX runtime InferenceSession."""
    _instance = None
    _ort_session = None
    _model_path = None

    @classmethod
    def get_session(cls, model_path):
        if cls._ort_session is None or cls._model_path != model_path:
            import onnxruntime as ort
            real_path = resolve_path(model_path)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            cls._ort_session = ort.InferenceSession(real_path, opts, providers=["CPUExecutionProvider"])
            cls._model_path = model_path
        return cls._ort_session


class Q2SessionTracker:
    """Tracks state and lifecycle of one session:
    - Maintains ml.features.SessionState for sequence ordering and tensor extraction.
    - Tracks purchase flag, total revenue, and emits session.closed upon completion.
    """
    def __init__(self, shop_id, session_id):
        self.shop_id = shop_id
        self.session_id = session_id
        self.state = SessionState(shop_id)
        self.has_purchase = False
        self.revenue_usd = 0.0
        self.last_seq_no = 0
        self.last_event_ts = ""
        self.closed_emitted = False

    def process_events(self, events, spec):
        """Processes events in the micro-batch. Returns list of (seq_no, event_ts, tensor)."""
        new_items = []
        for ev in events:
            etype = ev.get("event_type")
            seq = int(ev.get("seq_no") or 0)
            ets = str(ev.get("event_ts") or "")

            if seq > self.last_seq_no:
                self.last_seq_no = seq
                self.last_event_ts = ets

            if etype == "purchase":
                self.has_purchase = True
                amt = ev.get("amount_usd")
                if amt is not None:
                    try:
                        self.revenue_usd += float(amt)
                    except (ValueError, TypeError):
                        pass

        # Update SessionState
        emitted_steps = self.state.update(events, spec, emit="all")
        # Map seq_no -> event_ts for emitted predictions
        ts_map = {int(e.get("seq_no") or 0): str(e.get("event_ts") or "") for e in events}

        for seq, pos, tensor in emitted_steps:
            event_ts = ts_map.get(seq, self.last_event_ts)
            new_items.append((seq, event_ts, tensor))

        return new_items

    def check_close(self, force=False):
        """Emits session.closed record if purchase occurred or session ended (force=True)."""
        if self.closed_emitted:
            return None

        if self.has_purchase or force:
            self.closed_emitted = True
            self.state.closed = True
            return {
                "shop_id": str(self.shop_id),
                "session_id": int(self.session_id),
                "last_seq_no": int(self.last_seq_no),
                "last_event_ts": str(self.last_event_ts),
                "label_purchase": int(1 if self.has_purchase else 0),
                "revenue_usd": float(round(self.revenue_usd, 4)),
            }
        return None


class Q2Processor:
    """Manages active session state store and performs batched ONNX inferences.
    Supports single model execution as well as Champion / Challenger A/B testing.
    """
    def __init__(self, spec_path="models/feature_spec.json", model_path="models/intent_v1.onnx",
                 model_ver=MODEL_VER, challenger_path=None, ab_test=False):
        self.spec_path = resolve_path(spec_path)
        self.model_path = resolve_path(model_path)
        self.model_ver = model_ver
        self.challenger_path = resolve_path(challenger_path) if challenger_path else None
        self.ab_test = ab_test
        self.spec = load_spec(self.spec_path)
        self.trackers = {}  # (shop_id, session_id) -> Q2SessionTracker

    def _route_model(self, shop_id, session_id):
        if not self.ab_test or not self.challenger_path:
            return self.model_path, self.model_ver
        # 50/50 split based on stable session hash
        h = abs(hash((str(shop_id), int(session_id))))
        if h % 2 == 0:
            return self.model_path, "intent_v1"
        else:
            return self.challenger_path, "intent_transformer_v1"

    def process_records(self, records, force_close=False):
        """Process an iterable of event dicts.
        Returns:
            intent_records: list of dicts for pred.intent
            closed_records: list of dicts for session.closed
        """
        # Group by (shop_id, session_id)
        grouped = {}
        for r in records:
            key = (str(r.get("shop_id")), int(r.get("session_id")))
            grouped.setdefault(key, []).append(r)

        pending_by_model = {}  # (model_path, model_ver) -> list of (shop_id, session_id, seq, ets, tensor)
        closed_records = []

        for (shop_id, session_id), evs in grouped.items():
            if (shop_id, session_id) not in self.trackers:
                self.trackers[(shop_id, session_id)] = Q2SessionTracker(shop_id, session_id)
            tracker = self.trackers[(shop_id, session_id)]

            # Extract tensors
            items = tracker.process_events(evs, self.spec)
            m_path, m_ver = self._route_model(shop_id, session_id)
            for seq, ets, tensor in items:
                pending_by_model.setdefault((m_path, m_ver), []).append((shop_id, session_id, seq, ets, tensor))

        # Check for closure: all trackers if force_close, or only touched trackers
        trackers_to_check = self.trackers.values() if force_close else [
            self.trackers[k] for k in grouped if k in self.trackers
        ]
        for tracker in trackers_to_check:
            cl = tracker.check_close(force=force_close)
            if cl is not None:
                closed_records.append(cl)

        intent_records = []
        for (m_path, m_ver), items in pending_by_model.items():
            if not items:
                continue
            ort_sess = ModelInferenceWorker.get_session(m_path)
            x_cat_batch = np.stack([it[4]["x_cat"] for it in items], axis=0)
            x_num_batch = np.stack([it[4]["x_num"] for it in items], axis=0)
            length_batch = np.asarray([it[4]["length"] for it in items], dtype=np.int64)
            shop_batch = np.asarray([it[4]["shop"] for it in items], dtype=np.int64)

            feed = {
                "x_cat": x_cat_batch,
                "x_num": x_num_batch,
                "length": length_batch,
                "shop": shop_batch,
            }

            probs = ort_sess.run(None, feed)[0]

            for (shop_id, session_id, seq, ets, _), prob in zip(items, probs):
                intent_records.append({
                    "shop_id": str(shop_id),
                    "session_id": int(session_id),
                    "seq_no": int(seq),
                    "event_ts": str(ets),
                    "prob": float(round(float(prob), 5)),
                    "model_ver": str(m_ver),
                })

        return intent_records, closed_records


def build_q2_offline(spark, events_df, spec_path="models/feature_spec.json",
                     model_path="models/intent_v1.onnx", force_close=True):
    """Offline / Batch execution helper for unit testing & parity checks.
    Takes a canonical events DataFrame and returns (intent_df, closed_df).
    """
    rows = events_df.select(
        "shop_id", "session_id", "seq_no", "event_type", "product_id",
        "category", "brand", "price", "qty", "amount_usd",
        F.unix_millis("event_ts").alias("ts_ms"),
        F.date_format("event_ts", "yyyy-MM-dd HH:mm:ss.SSS").alias("event_ts")
    ).collect()

    records = [r.asDict() for r in rows]
    processor = Q2Processor(spec_path=spec_path, model_path=model_path)
    intent_data, closed_data = processor.process_records(records, force_close=force_close)

    intent_df = spark.createDataFrame(
        intent_data if intent_data else [],
        schema=INTENT_SCHEMA
    ) if intent_data else spark.createDataFrame([], schema=INTENT_SCHEMA)

    closed_df = spark.createDataFrame(
        closed_data if closed_data else [],
        schema=CLOSED_SCHEMA
    ) if closed_data else spark.createDataFrame([], schema=CLOSED_SCHEMA)

    return intent_df, closed_df


def build(spark, bootstrap, in_topic="events.clean", pred_topic="pred.intent",
          closed_topic="session.closed", spec_path="models/feature_spec.json",
          model_path="models/intent_v1.onnx", model_ver=MODEL_VER,
          challenger_path=None, ab_test=False, starting="latest",
          checkpoint_dir="/opt/project/checkpoints/q2-intent"):
    """Builds the Spark Structured Streaming pipeline for Q2."""
    spec_path_abs = resolve_path(spec_path)
    model_path_abs = resolve_path(model_path)
    challenger_path_abs = resolve_path(challenger_path) if challenger_path else None

    # 1. Read input stream from Kafka
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", bootstrap)
           .option("subscribe", in_topic)
           .option("startingOffsets", starting)
           .load())

    val = F.from_json(F.col("value").cast("string"), EVENT_SCHEMA)
    df = raw.select(val.alias("e")).select("e.*")

    # Select normalized fields required for feature encoding & tracking
    prep_df = df.select(
        "shop_id", "session_id", "seq_no", "event_type", "product_id",
        "category", "brand", "price", "qty", "amount_usd",
        F.unix_millis("event_ts").alias("ts_ms"),
        F.date_format("event_ts", "yyyy-MM-dd HH:mm:ss.SSS").alias("event_ts")
    )

    processor = Q2Processor(spec_path=spec_path_abs, model_path=model_path_abs,
                            model_ver=model_ver, challenger_path=challenger_path_abs,
                            ab_test=ab_test)

    def write_to_kafka(micro_df, target_topic):
        (micro_df.select(F.to_json(F.struct(*micro_df.columns)).alias("value"))
         .selectExpr("CAST(value AS STRING)")
         .write.format("kafka")
         .option("kafka.bootstrap.servers", bootstrap)
         .option("topic", target_topic)
         .save())

    def foreach_batch_fn(batch_df, batch_id):
        if batch_df.isEmpty():
            return

        records = [r.asDict() for r in batch_df.collect()]
        intent_data, closed_data = processor.process_records(records, force_close=False)

        if intent_data:
            intent_df = spark.createDataFrame(intent_data, schema=INTENT_SCHEMA)
            write_to_kafka(intent_df, pred_topic)

        if closed_data:
            closed_df = spark.createDataFrame(closed_data, schema=CLOSED_SCHEMA)
            write_to_kafka(closed_df, closed_topic)

    query = (prep_df.writeStream
             .foreachBatch(foreach_batch_fn)
             .option("checkpointLocation", checkpoint_dir)
             .start())

    return [query]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q2: Session State + Real-time ONNX Intent Inference")
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    ap.add_argument("--in-topic", default="events.clean")
    ap.add_argument("--pred-topic", default="pred.intent")
    ap.add_argument("--closed-topic", default="session.closed")
    ap.add_argument("--spec", default="models/feature_spec.json")
    ap.add_argument("--model", default="models/intent_v1.onnx")
    ap.add_argument("--model-ver", default=MODEL_VER)
    ap.add_argument("--challenger", default="models/intent_transformer_full.onnx")
    ap.add_argument("--ab-test", action="store_true", help="Enable 50/50 Champion vs Challenger online testing")
    ap.add_argument("--starting", default="latest")
    ap.add_argument("--checkpoint", default="/opt/project/checkpoints/q2-intent")
    args = ap.parse_args()

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("q2-intent")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    queries = build(
        spark=spark,
        bootstrap=args.bootstrap,
        in_topic=args.in_topic,
        pred_topic=args.pred_topic,
        closed_topic=args.closed_topic,
        spec_path=args.spec,
        model_path=args.model,
        model_ver=args.model_ver,
        challenger_path=args.challenger if args.ab_test else None,
        ab_test=args.ab_test,
        starting=args.starting,
        checkpoint_dir=args.checkpoint,
    )
    for q in queries:
        q.awaitTermination()
