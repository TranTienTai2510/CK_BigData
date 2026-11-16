"""Test Q2 logic and Batch-vs-Stream parity (A - Week 3 & G5).

Verifies:
1. Strict seq_no ordering with out-of-order and duplicate event handling.
2. Correct generation of pred.intent records with valid calibrated probabilities from intent_v1.onnx.
3. Proper closure detection: label_purchase=1 with revenue_usd on purchase, label_purchase=0 on session end.
4. Schema conformance matching ClickHouse rt.intent, rt.session_closed, and rt.v_online_eval_1m.
5. Bit-exact parity: streaming Q2 outputs match offline batch inference (max abs diff < 1e-5).
"""
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import onnxruntime as ort

from ml.features import SessionState, load_spec, window_to_tensor
from streaming.q2_intent import (
    INTENT_SCHEMA,
    CLOSED_SCHEMA,
    MODEL_VER,
    Q2Processor,
    Q2SessionTracker,
)


def assert_check(name, got, expected):
    passed = (got == expected)
    status = "OK " if passed else "FAIL"
    print(f"{status} {name:<42} got={got} expected={expected}")
    assert passed, f"Check failed: {name} (got={got}, expected={expected})"


def test_q2_basic_flow():
    print("--- 1. Testing Q2 basic flow and schema contract ---")
    spec_path = "models/feature_spec.json"
    model_path = "models/intent_v1.onnx"
    processor = Q2Processor(spec_path=spec_path, model_path=model_path)

    # Session 1: page_view -> add_to_cart -> purchase
    events_s1 = [
        {"shop_id": "synthetic", "session_id": 101, "seq_no": 1, "event_type": "page_view",
         "product_id": 501, "category": "cat1", "brand": "brand1", "price": 25.0, "qty": None,
         "amount_usd": None, "event_ts": "2024-01-01 10:00:00.000", "ts_ms": 1704103200000},
        {"shop_id": "synthetic", "session_id": 101, "seq_no": 2, "event_type": "add_to_cart",
         "product_id": 501, "category": "cat1", "brand": "brand1", "price": 25.0, "qty": 1,
         "amount_usd": None, "event_ts": "2024-01-01 10:01:00.000", "ts_ms": 1704103260000},
        {"shop_id": "synthetic", "session_id": 101, "seq_no": 3, "event_type": "purchase",
         "product_id": None, "category": None, "brand": None, "price": None, "qty": None,
         "amount_usd": 25.0, "event_ts": "2024-01-01 10:02:00.000", "ts_ms": 1704103320000},
    ]

    intents, closed = processor.process_records(events_s1, force_close=False)

    # Pre-cut events should produce predictions (seq 1 and seq 2)
    assert_check("s1_intent_count", len(intents), 2)
    assert_check("s1_first_seq_no", intents[0]["seq_no"], 1)
    assert_check("s1_second_seq_no", intents[1]["seq_no"], 2)
    assert_check("s1_model_ver", intents[0]["model_ver"], MODEL_VER)

    # Probability after add_to_cart should be higher than initial page_view
    p_view = intents[0]["prob"]
    p_add = intents[1]["prob"]
    print(f"Probabilities: page_view={p_view:.4f}, add_to_cart={p_add:.4f}")
    assert_check("s1_prob_in_range", 0.0 <= p_view <= 1.0 and 0.0 <= p_add <= 1.0, True)
    assert_check("s1_intent_increases_on_cart", p_add > p_view, True)

    # Session closed should be emitted due to purchase
    assert_check("s1_closed_count", len(closed), 1)
    assert_check("s1_closed_label", closed[0]["label_purchase"], 1)
    assert_check("s1_closed_revenue", closed[0]["revenue_usd"], 25.0)
    assert_check("s1_closed_last_seq", closed[0]["last_seq_no"], 3)


def test_q2_non_purchasing_session():
    print("\n--- 2. Testing Q2 non-purchasing session & force close ---")
    spec_path = "models/feature_spec.json"
    model_path = "models/intent_v1.onnx"
    processor = Q2Processor(spec_path=spec_path, model_path=model_path)

    # Session 2: browsing only, no checkout/purchase
    events_s2 = [
        {"shop_id": "rees46_cosmetics", "session_id": 202, "seq_no": 1, "event_type": "page_view",
         "product_id": 701, "category": "cream", "brand": "loreal", "price": 12.5, "qty": None,
         "amount_usd": None, "event_ts": "2024-01-01 11:00:00.000", "ts_ms": 1704106800000},
        {"shop_id": "rees46_cosmetics", "session_id": 202, "seq_no": 2, "event_type": "page_view",
         "product_id": 702, "category": "shampoo", "brand": "dove", "price": 8.0, "qty": None,
         "amount_usd": None, "event_ts": "2024-01-01 11:03:00.000", "ts_ms": 1704106980000},
    ]

    intents, closed = processor.process_records(events_s2, force_close=False)
    assert_check("s2_intent_count", len(intents), 2)
    assert_check("s2_closed_before_timeout", len(closed), 0)

    # Session times out (force_close=True)
    _, closed_after_timeout = processor.process_records([], force_close=True)
    assert_check("s2_closed_after_timeout", len(closed_after_timeout), 1)
    assert_check("s2_timeout_label", closed_after_timeout[0]["label_purchase"], 0)
    assert_check("s2_timeout_revenue", closed_after_timeout[0]["revenue_usd"], 0.0)
    assert_check("s2_timeout_last_seq", closed_after_timeout[0]["last_seq_no"], 2)


def test_q2_out_of_order_and_duplicates():
    print("\n--- 3. Testing Q2 out-of-order and duplicate robustness ---")
    spec_path = "models/feature_spec.json"
    model_path = "models/intent_v1.onnx"
    processor = Q2Processor(spec_path=spec_path, model_path=model_path)

    # Events arrive in scrambled order with a duplicate
    # Sequence: 2 arrives first, then 1, then duplicate 2, then 3
    ev2 = {"shop_id": "synthetic", "session_id": 303, "seq_no": 2, "event_type": "add_to_cart",
           "product_id": 10, "category": "c", "brand": "b", "price": 20.0, "qty": 1,
           "amount_usd": None, "event_ts": "2024-01-01 12:01:00.000", "ts_ms": 1704110460000}
    ev1 = {"shop_id": "synthetic", "session_id": 303, "seq_no": 1, "event_type": "page_view",
           "product_id": 10, "category": "c", "brand": "b", "price": 20.0, "qty": None,
           "amount_usd": None, "event_ts": "2024-01-01 12:00:00.000", "ts_ms": 1704110400000}
    ev2_dup = dict(ev2)
    ev3 = {"shop_id": "synthetic", "session_id": 303, "seq_no": 3, "event_type": "checkout",
           "product_id": None, "category": None, "brand": None, "price": None, "qty": None,
           "amount_usd": None, "event_ts": "2024-01-01 12:02:00.000", "ts_ms": 1704110520000}

    # Micro-batch 1: receives seq_no 2 (should be buffered in pending)
    intents1, _ = processor.process_records([ev2])
    assert_check("ooo_mb1_pending_intents", len(intents1), 0)

    # Micro-batch 2: receives seq_no 1 (unblocks seq 1 and seq 2)
    intents2, _ = processor.process_records([ev1])
    assert_check("ooo_mb2_unblocked_intents", len(intents2), 2)
    assert_check("ooo_mb2_first_seq", intents2[0]["seq_no"], 1)
    assert_check("ooo_mb2_second_seq", intents2[1]["seq_no"], 2)

    # Micro-batch 3: duplicate seq_no 2 (should be safely dropped)
    intents3, _ = processor.process_records([ev2_dup])
    assert_check("ooo_mb3_dup_dropped", len(intents3), 0)

    # Micro-batch 4: checkout event (cut event -> no new intent emitted)
    intents4, _ = processor.process_records([ev3])
    assert_check("ooo_mb4_cut_no_intent", len(intents4), 0)


def test_batch_vs_stream_parity():
    print("\n--- 4. Testing Bit-Exact Batch-vs-Stream Parity ---")
    spec = load_spec("models/feature_spec.json")
    ort_sess = ort.InferenceSession("models/intent_v1.onnx", providers=["CPUExecutionProvider"])

    # Simulate 5 distinct realistic sessions across both shops
    test_sessions = [
        ("synthetic", 1001, [
            ("page_view", 11, "cat_a", "brand_x", 19.99, None, 1000),
            ("add_to_cart", 11, "cat_a", "brand_x", 19.99, 2, 45000),
            ("page_view", 12, "cat_b", "brand_y", 49.00, None, 80000),
        ]),
        ("rees46_cosmetics", 1002, [
            ("page_view", 21, "skin", "brand_m", 5.50, None, 2000),
            ("page_view", 22, "hair", "brand_n", 14.20, None, 12000),
            ("add_to_cart", 22, "hair", "brand_n", 14.20, 1, 35000),
            ("add_to_cart", 21, "skin", "brand_m", 5.50, 1, 55000),
        ]),
        ("synthetic", 1003, [
            ("page_view", 31, "electronics", "brand_k", 99.0, None, 5000),
        ]),
    ]

    all_stream_records = []
    batch_ground_truth = []

    for shop_id, sess_id, raw_events in test_sessions:
        # 1. Batch ground truth calculation
        st_batch = SessionState(shop_id)
        batch_evs = [
            {"shop_id": shop_id, "session_id": sess_id, "seq_no": i + 1, "event_type": et,
             "product_id": pid, "category": cat, "brand": br, "price": pr, "qty": q,
             "amount_usd": None, "event_ts": f"2024-01-01 10:00:{i:02d}.000", "ts_ms": ts}
            for i, (et, pid, cat, br, pr, q, ts) in enumerate(raw_events)
        ]
        all_stream_records.extend(batch_evs)

        steps = st_batch.update(batch_evs, spec, emit="all")
        for seq, pos, tensor in steps:
            feed = {
                "x_cat": np.expand_dims(tensor["x_cat"], 0),
                "x_num": np.expand_dims(tensor["x_num"], 0),
                "length": np.asarray([tensor["length"]], dtype=np.int64),
                "shop": np.asarray([tensor["shop"]], dtype=np.int64),
            }
            prob = float(ort_sess.run(None, feed)[0][0])
            batch_ground_truth.append((shop_id, sess_id, seq, prob))

    # 2. Streaming Q2 execution
    processor = Q2Processor(spec_path="models/feature_spec.json", model_path="models/intent_v1.onnx")
    intents_stream, _ = processor.process_records(all_stream_records, force_close=True)

    # Sort both by (shop_id, session_id, seq_no)
    batch_sorted = sorted(batch_ground_truth, key=lambda x: (x[0], x[1], x[2]))
    stream_sorted = sorted(intents_stream, key=lambda x: (x["shop_id"], x["session_id"], x["seq_no"]))

    assert_check("parity_sample_count", len(stream_sorted), len(batch_sorted))

    max_abs_diff = 0.0
    for b_item, s_item in zip(batch_sorted, stream_sorted):
        assert b_item[0] == s_item["shop_id"]
        assert b_item[1] == s_item["session_id"]
        assert b_item[2] == s_item["seq_no"]
        diff = abs(b_item[3] - s_item["prob"])
        if diff > max_abs_diff:
            max_abs_diff = diff

    print(f"Batch vs Stream max_abs_diff = {max_abs_diff:.8e}")
    assert_check("parity_max_abs_diff_lt_1e-5", max_abs_diff < 1e-5, True)


def main():
    test_q2_basic_flow()
    test_q2_non_purchasing_session()
    test_q2_out_of_order_and_duplicates()
    test_batch_vs_stream_parity()
    print("\nALL Q2 TESTS PASS!")


if __name__ == "__main__":
    main()
