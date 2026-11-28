"""Q2: distributed session state and ONNX purchase-intent inference."""
import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.streaming.state import GroupStateTimeout

from common.canonical import EVENT_SCHEMA
from common.messages import Q2_OUTPUT_SCHEMA, Q2_STATE_SCHEMA
from ml.features import SessionState, load_spec

MODEL_VER = "intent_v1"
SESSION_INACTIVITY_MS = 30 * 60 * 1000


def resolve_path(relative_or_abs):
    candidates = [
        relative_or_abs,
        os.path.join(ROOT_DIR, relative_or_abs),
        os.path.join("/opt/project", relative_or_abs),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[1])


class ModelInferenceWorker:
    """Cache ONNX sessions in each Python worker process."""
    _sessions = {}

    @classmethod
    def get_session(cls, model_path):
        real_path = resolve_path(model_path)
        if real_path not in cls._sessions:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            cls._sessions[real_path] = ort.InferenceSession(
                real_path, options, providers=["CPUExecutionProvider"]
            )
        return cls._sessions[real_path]


def _predict(model_path, tensors):
    if not tensors:
        return []

    session = ModelInferenceWorker.get_session(model_path)
    feed = {
        "x_cat": np.stack([tensor["x_cat"] for tensor in tensors]),
        "x_num": np.stack([tensor["x_num"] for tensor in tensors]),
        "length": np.asarray([tensor["length"] for tensor in tensors], dtype=np.int64),
        "shop": np.asarray([tensor["shop"] for tensor in tensors], dtype=np.int64),
    }
    return session.run(["prob"], feed)[0].reshape(-1).tolist()


def _route_model(shop_id, session_id, model_path, model_ver, challenger_path, ab_test):
    if not ab_test or not challenger_path:
        return model_path, model_ver
    route_key = f"{shop_id}\x1f{session_id}".encode("utf-8")
    route = int.from_bytes(hashlib.blake2b(route_key, digest_size=8).digest(), "big") % 2
    if route == 0:
        return model_path, MODEL_VER
    return challenger_path, "intent_transformer_v1"


def _format_event_time(timestamp_ms):
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _output_frame(pd, records):
    frame = pd.DataFrame(records, columns=Q2_OUTPUT_SCHEMA.fieldNames())
    for column in ("seq_no", "last_pred_seq_no", "last_seq_no", "label_purchase"):
        frame[column] = pd.array(frame[column], dtype="Int32")
    frame["prob"] = pd.array(frame["prob"], dtype="Float32")
    for column in ("record_type", "shop_id", "event_ts", "model_ver", "last_event_ts"):
        frame[column] = pd.array(frame[column], dtype="string")
    return frame


def _stateful_function(spec_path, model_path, model_ver, challenger_path, ab_test, inference_fn):
    spec = load_spec(spec_path)

    def infer_and_format(shop_id, session_id, predictions):
        if not predictions:
            return []
        model_path_for_session, model_ver_for_session = _route_model(
            shop_id, session_id, model_path, model_ver, challenger_path, ab_test
        )
        tensors = [item[2] for item in predictions]
        scores = (inference_fn or _predict)(model_path_for_session, tensors)
        if len(scores) != len(predictions):
            raise RuntimeError(
                f"inference returned {len(scores)} scores for {len(predictions)} prefixes"
            )
        return [
            {
                "record_type": "intent",
                "shop_id": shop_id,
                "session_id": session_id,
                "seq_no": int(seq),
                "event_ts": _format_event_time(event_ts_ms),
                "prob": float(score),
                "model_ver": model_ver_for_session,
                "last_pred_seq_no": None,
                "last_seq_no": None,
                "last_event_ts": None,
                "label_purchase": None,
                "revenue_usd": None,
            }
            for (seq, event_ts_ms, _), score in zip(predictions, scores)
        ]

    def process_group(key, pdf_iterator, group_state):
        import pandas as pd

        shop_id, session_id = str(key[0]), int(key[1])

        if group_state.exists:
            session_json, has_purchase, purchase_seq_no, revenue_usd = group_state.get
            session_state = SessionState.from_json(session_json)
        else:
            session_state = SessionState(shop_id)
            has_purchase = False
            purchase_seq_no = None
            revenue_usd = 0.0

        output_records = []
        pending_predictions = []

        def append_predictions(predictions):
            pending_predictions.extend(predictions)

        def append_closed():
            output_records.append({
                "record_type": "closed",
                "shop_id": shop_id,
                "session_id": session_id,
                "seq_no": None,
                "event_ts": None,
                "prob": None,
                "model_ver": None,
                "last_pred_seq_no": session_state.last_pred_seq_no,
                "last_seq_no": session_state.last_seq_no,
                "last_event_ts": _format_event_time(session_state.last_event_ts_ms),
                "label_purchase": int(has_purchase),
                "revenue_usd": float(round(revenue_usd, 4)),
            })

        if group_state.hasTimedOut:
            pending_predictions = session_state.update([], spec, emit="all", max_pending=0)
            pending_predictions = [
                (seq, session_state.emitted_event_ts_ms[seq], tensor)
                for seq, _, tensor in pending_predictions
            ]
            append_closed()
            group_state.remove()
        else:
            saw_input = False
            for frame in pdf_iterator:
                saw_input = True
                records = frame.sort_values(
                    ["seq_no", "ts_ms"], kind="mergesort"
                ).to_dict(orient="records")
                for event in records:
                    seq_no = int(event["seq_no"])
                    already_seen = (
                        seq_no < session_state.next_seq or seq_no in session_state.pending
                    )
                    emitted = session_state.update([event], spec, emit="all")

                    if not already_seen and event["event_type"] == "purchase":
                        has_purchase = True
                        purchase_seq_no = (
                            seq_no if purchase_seq_no is None else min(purchase_seq_no, seq_no)
                        )
                        amount = event.get("amount_usd")
                        if amount is not None and not pd.isna(amount):
                            revenue_usd += float(amount)

                    append_predictions([
                        (emitted_seq, session_state.emitted_event_ts_ms[emitted_seq], tensor)
                        for emitted_seq, _, tensor in emitted
                    ])

            if not saw_input:
                return iter([])

            purchase_processed = (
                has_purchase
                and purchase_seq_no is not None
                and session_state.next_seq > purchase_seq_no
            )
            if purchase_processed:
                append_closed()
                group_state.remove()
            else:
                group_state.update((
                    session_state.to_json(),
                    bool(has_purchase),
                    purchase_seq_no,
                    float(revenue_usd),
                ))
                timeout_ms = (
                    (session_state.last_activity_ts_ms or 0) + SESSION_INACTIVITY_MS
                )
                watermark_ms = group_state.getCurrentWatermarkMs()
                if watermark_ms >= 0:
                    timeout_ms = max(timeout_ms, watermark_ms + 1)
                group_state.setTimeoutTimestamp(timeout_ms)

        output_records.extend(infer_and_format(shop_id, session_id, pending_predictions))
        if output_records:
            yield _output_frame(pd, output_records)

    return process_group


def apply_q2_stateful(events_df, spec_path="models/feature_spec.json",
                      model_path="models/intent_v1.onnx", model_ver=MODEL_VER,
                      challenger_path=None, ab_test=False, inference_fn=None):
    """Apply per-session event-time state to a streaming DataFrame."""
    spec_path_abs = resolve_path(spec_path)
    model_path_abs = model_path if inference_fn else resolve_path(model_path)
    challenger_path_abs = (
        challenger_path if inference_fn else resolve_path(challenger_path)
    ) if challenger_path else None
    events = events_df.withColumn("ts_ms", F.unix_millis("event_ts"))
    handler = _stateful_function(
        spec_path_abs, model_path_abs, model_ver, challenger_path_abs, ab_test, inference_fn
    )
    return (
        events.withWatermark("event_ts", "30 minutes")
        .groupBy("shop_id", "session_id")
        .applyInPandasWithState(
            func=handler,
            outputStructType=Q2_OUTPUT_SCHEMA,
            stateStructType=Q2_STATE_SCHEMA,
            outputMode="Append",
            timeoutConf=GroupStateTimeout.EventTimeTimeout,
        )
    )


def build(spark, bootstrap, in_topic="events.clean", pred_topic="pred.intent",
          closed_topic="session.closed", spec_path="models/feature_spec.json",
          model_path="models/intent_v1.onnx", model_ver=MODEL_VER,
          challenger_path=None, ab_test=False, starting="latest",
          checkpoint_dir="/opt/project/checkpoints/q2-intent"):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", in_topic)
        .option("startingOffsets", starting)
        .load()
    )
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("event")
    ).select("event.*")
    stateful = apply_q2_stateful(
        parsed,
        spec_path=spec_path,
        model_path=model_path,
        model_ver=model_ver,
        challenger_path=challenger_path,
        ab_test=ab_test,
    )

    intent_json = F.to_json(F.struct(
        "shop_id", "session_id", "seq_no", "event_ts", "prob", "model_ver"
    ))
    closed_json = F.to_json(F.struct(
        "shop_id", "session_id", "last_pred_seq_no", "last_seq_no", "last_event_ts",
        "label_purchase", "revenue_usd",
    ))
    routed = stateful.select(
        F.when(F.col("record_type") == "intent", F.lit(pred_topic))
         .otherwise(F.lit(closed_topic)).alias("topic"),
        F.concat_ws(":", "shop_id", F.col("session_id").cast("string")).alias("key"),
        F.when(F.col("record_type") == "intent", intent_json)
         .otherwise(closed_json).alias("value"),
    )
    query = (
        routed.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("checkpointLocation", checkpoint_dir)
        .outputMode("append")
        .start()
    )
    return [query]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2 distributed ONNX intent inference")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    parser.add_argument("--in-topic", default="events.clean")
    parser.add_argument("--pred-topic", default="pred.intent")
    parser.add_argument("--closed-topic", default="session.closed")
    parser.add_argument("--spec", default="models/feature_spec.json")
    parser.add_argument("--model", default="models/intent_v1.onnx")
    parser.add_argument("--model-ver", default=MODEL_VER)
    parser.add_argument("--challenger", default="models/intent_transformer_full.onnx")
    parser.add_argument("--ab-test", action="store_true")
    parser.add_argument("--starting", default="latest")
    parser.add_argument("--checkpoint", default="/opt/project/checkpoints/q2-intent")
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("q2-intent")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
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
    for query in queries:
        query.awaitTermination()
