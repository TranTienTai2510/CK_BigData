"""Tests for distributed Q2 session lifecycle and batch/stream feature parity."""
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from pyspark.sql import types as T

from common.canonical import get_spark
from ml.features import SessionState, load_spec, session_tensors
from streaming.q2_intent import apply_q2_stateful

SPEC_PATH = os.path.join(ROOT_DIR, "models", "feature_spec.json")


def _test_inference(_model_path, tensors):
    assert _model_path == "test-model"
    return [0.5 + 0.01 * min(int(tensor["length"]), 10) for tensor in tensors]


def _event(shop_id, session_id, seq_no, event_type, ts, **extra):
    event = {
        "shop_id": shop_id,
        "session_id": session_id,
        "seq_no": seq_no,
        "event_type": event_type,
        "product_id": None,
        "category": None,
        "brand": None,
        "price": None,
        "qty": None,
        "amount_usd": None,
        "event_ts": ts.isoformat().replace("+00:00", "Z"),
    }
    event.update(extra)
    return event


def _test_input_schema():
    return T.StructType([
        T.StructField("shop_id", T.StringType(), False),
        T.StructField("session_id", T.LongType(), False),
        T.StructField("seq_no", T.IntegerType(), False),
        T.StructField("event_type", T.StringType(), False),
        T.StructField("product_id", T.LongType(), True),
        T.StructField("category", T.StringType(), True),
        T.StructField("brand", T.StringType(), True),
        T.StructField("price", T.DoubleType(), True),
        T.StructField("qty", T.IntegerType(), True),
        T.StructField("amount_usd", T.DoubleType(), True),
        T.StructField("event_ts", T.TimestampType(), False),
    ])


def test_tensor_batch_stream_parity_is_exact():
    spec = load_spec(SPEC_PATH)
    shop_id = "synthetic"
    events = [
        {"seq_no": 1, "event_type": "page_view", "product_id": 11, "category": "c1",
         "brand": "b1", "price": 12.5, "qty": None, "ts_ms": 1000},
        {"seq_no": 2, "event_type": "add_to_cart", "product_id": 11, "category": "c1",
         "brand": "b1", "price": 12.5, "qty": 2, "ts_ms": 2000},
        {"seq_no": 3, "event_type": "page_view", "product_id": 12, "category": "c2",
         "brand": "b2", "price": 8.0, "qty": None, "ts_ms": 5000},
        {"seq_no": 4, "event_type": "purchase", "product_id": None, "category": None,
         "brand": None, "price": None, "qty": None, "amount_usd": 20.5, "ts_ms": 6000},
    ]

    state = SessionState(shop_id)
    emitted = state.update(events, spec, emit="all")
    shop_index = spec["shops"][shop_id]
    batch = session_tensors(state.window, shop_index, max_len=spec["max_len"])

    assert [seq for seq, _, _ in emitted] == [1, 2, 3]
    assert state.last_pred_seq_no == 3
    assert state.last_seq_no == 4
    assert state.last_event_ts_ms == 6000
    for index, (_, _, streamed) in enumerate(emitted):
        assert np.array_equal(streamed["x_cat"], batch["x_cat"][index])
        assert np.array_equal(streamed["x_num"], batch["x_num"][index])
        assert streamed["length"] == batch["length"][index]
        assert streamed["shop"] == batch["shop"][index]

    restored = SessionState.from_json(state.to_json())
    assert restored.last_pred_seq_no == state.last_pred_seq_no
    assert restored.last_seq_no == state.last_seq_no
    assert restored.last_event_ts_ms == state.last_event_ts_ms


def test_apply_in_pandas_stateful_lifecycle_and_online_join():
    spark = get_spark(
        "test-q2-stateful", driver_memory="2g", cores=2, shuffle_partitions=2
    )
    spark.conf.set("spark.sql.streaming.noDataMicroBatches.enabled", "true")
    spark.sparkContext.setLogLevel("ERROR")

    root = tempfile.mkdtemp(prefix="q2_stateful_")
    input_dir = os.path.join(root, "input")
    checkpoint_dir = os.path.join(root, "checkpoint")
    os.makedirs(input_dir)
    query_name = f"q2_test_{uuid.uuid4().hex}"
    query = None

    def write_events(file_name, events):
        temp_path = os.path.join(input_dir, f".{file_name}.tmp")
        final_path = os.path.join(input_dir, file_name)
        with open(temp_path, "w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event) + "\n")
        os.replace(temp_path, final_path)

    try:
        source = (
            spark.readStream.schema(_test_input_schema())
            .option("maxFilesPerTrigger", 1)
            .json(input_dir)
        )
        output = apply_q2_stateful(
            source,
            spec_path=SPEC_PATH,
            model_path="test-model",
            inference_fn=_test_inference,
        )
        query = (
            output.writeStream.format("memory")
            .queryName(query_name)
            .outputMode("append")
            .option("checkpointLocation", checkpoint_dir)
            .trigger(processingTime="500 milliseconds")
            .start()
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        positive = [
            _event("synthetic", 101, 1, "page_view", start,
                   product_id=501, category="cat1", brand="brand1", price=25.0),
            _event("synthetic", 101, 2, "purchase", start + timedelta(seconds=1),
                   amount_usd=25.0),
        ]
        negative = [
            _event("rees46_cosmetics", 202, 1, "page_view", start,
                   product_id=701, category="cream", brand="loreal", price=12.5),
            _event("rees46_cosmetics", 202, 2, "page_view", start + timedelta(minutes=3),
                   product_id=702, category="shampoo", brand="dove", price=8.0),
        ]
        write_events("batch_1.json", positive + negative)
        query.processAllAvailable()

        advance = start + timedelta(minutes=64)
        write_events("batch_2.json", [
            _event("synthetic", 303, 1, "page_view", advance,
                   product_id=801, category="clock", brand="test", price=1.0)
        ])
        query.processAllAvailable()
        write_events("batch_3.json", [
            _event("synthetic", 304, 1, "page_view", advance + timedelta(minutes=1),
                   product_id=802, category="clock", brand="test", price=1.0)
        ])
        query.processAllAvailable()

        deadline = time.time() + 15
        rows = []
        while time.time() < deadline:
            query.processAllAvailable()
            rows = spark.sql(f"SELECT * FROM {query_name}").collect()
            closed = [row for row in rows if row.record_type == "closed"]
            if any(row.session_id == 202 for row in closed):
                break
            time.sleep(0.25)

        intents = [row for row in rows if row.record_type == "intent"]
        closed = [row for row in rows if row.record_type == "closed"]
        positive_intents = [
            row for row in intents
            if row.shop_id == "synthetic" and row.session_id == 101
        ]
        positive_closed = [
            row for row in closed
            if row.shop_id == "synthetic" and row.session_id == 101
        ]
        negative_closed = [
            row for row in closed
            if row.shop_id == "rees46_cosmetics" and row.session_id == 202
        ]

        assert len(positive_closed) == 1
        assert positive_closed[0].label_purchase == 1
        assert positive_closed[0].last_seq_no == 2
        assert positive_closed[0].last_pred_seq_no == positive_intents[-1].seq_no

        assert len(negative_closed) == 1
        assert negative_closed[0].label_purchase == 0
        assert negative_closed[0].last_seq_no == 2
        assert negative_closed[0].last_pred_seq_no == 2
        assert any(
            row.shop_id == negative_closed[0].shop_id
            and row.session_id == negative_closed[0].session_id
            and row.seq_no == negative_closed[0].last_pred_seq_no
            for row in intents
        )
    finally:
        if query is not None:
            query.stop()
        spark.stop()
        shutil.rmtree(root, ignore_errors=True)


def main():
    test_tensor_batch_stream_parity_is_exact()
    test_apply_in_pandas_stateful_lifecycle_and_online_join()
    print("ALL Q2 STATEFUL TESTS PASS")


if __name__ == "__main__":
    main()
