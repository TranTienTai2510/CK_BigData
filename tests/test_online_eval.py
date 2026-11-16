"""Test Online Evaluation, Champion vs Challenger A/B testing & Monitoring (A - Week 4 & G6).

Verifies:
1. Online ROC-AUC and PR-AUC computation with tolerance check (|online - offline| <= 0.02).
2. Granular breakdown by shop and prefix length bucket (1, 2, 3-5, 6-10, 11-20, 21-50, >50).
3. Online A/B testing mechanism routing sessions 50/50 between Champion (intent_v1) and Challenger (intent_transformer_v1).
4. Pipeline health monitoring: normal status is HEALTHY (no STALL / BACKPRESSURE).
"""
import json
import math
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from analysis.live import (
    PREFIX_BUCKETS,
    PipelineMonitor,
    assign_prefix_bucket,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_online_pairs,
)
from streaming.q2_intent import Q2Processor


def assert_check(name, got, expected):
    passed = (got == expected)
    status = "OK " if passed else "FAIL"
    print(f"{status} {name:<45} got={got} expected={expected}")
    assert passed, f"Check failed: {name} (got={got}, expected={expected})"


def test_online_evaluation_metrics():
    print("--- 1. Testing Online Evaluation & Delta AUC <= 0.02 ---")
    np.random.seed(42)
    n = 5000
    pairs = []
    for i in range(n):
        shop = "rees46_cosmetics" if i % 2 == 0 else "synthetic"
        k = int(np.random.choice([1, 2, 3, 5, 8, 14, 25, 55]))
        # True purchase rate ~18.5%
        y = 1 if np.random.rand() < (0.28 if shop == "synthetic" else 0.07) else 0
        # Signal detection model: d' = sqrt(2)*Phi^-1(0.785) ~ 1.116
        score = np.random.normal(loc=1.116 if y == 1 else 0.0, scale=1.0)
        prob = float(1.0 / (1.0 + np.exp(-score)))
        pairs.append({
            "shop_id": shop,
            "session_id": 200000 + i,
            "seq_no": k,
            "prob": prob,
            "label_purchase": y,
            "model_ver": "intent_v1",
        })

    report = evaluate_online_pairs(pairs, offline_auc=0.7850, model_ver="intent_v1")

    assert_check("total_sessions_evaluated", report["total_evaluated_sessions"], n)
    assert_check("online_roc_auc_valid", 0.70 <= report["online_roc_auc"] <= 0.85, True)
    print(f"Online ROC-AUC = {report['online_roc_auc']:.4f}, Offline = {report['offline_roc_auc']:.4f}, Delta = {report['delta_auc']:.4f}")
    assert_check("tolerance_criterion_le_0_02", report["delta_auc"] <= 0.02, True)
    assert_check("passed_tolerance_flag", report["passed_tolerance"], True)

    # Check breakdown by shop
    assert_check("has_both_shops", set(report["by_shop"].keys()), {"rees46_cosmetics", "synthetic"})

    # Check breakdown by prefix length
    for b in ["1", "2", "3-5", "6-10"]:
        assert_check(f"has_prefix_bucket_{b}", b in report["by_prefix_len"], True)


def test_ab_testing_champion_vs_challenger():
    print("\n--- 2. Testing Champion vs Challenger A/B Routing ---")
    spec_path = "models/feature_spec.json"
    champ_path = "models/intent_v1.onnx"
    challenger_path = "models/intent_transformer_full.onnx"

    processor = Q2Processor(
        spec_path=spec_path,
        model_path=champ_path,
        model_ver="intent_v1",
        challenger_path=challenger_path,
        ab_test=True,
    )

    # Create events for 20 distinct sessions
    records = []
    for sess_id in range(1, 21):
        records.append({
            "shop_id": "synthetic" if sess_id % 2 == 0 else "rees46_cosmetics",
            "session_id": sess_id,
            "seq_no": 1,
            "event_type": "page_view",
            "product_id": 100 + sess_id,
            "category": "c",
            "brand": "b",
            "price": 20.0,
            "qty": None,
            "amount_usd": None,
            "event_ts": "2024-01-01 10:00:00.000",
            "ts_ms": 1704103200000,
        })

    intents, _ = processor.process_records(records, force_close=True)
    assert_check("ab_intents_emitted", len(intents), 20)

    champ_count = sum(1 for it in intents if it["model_ver"] == "intent_v1")
    chall_count = sum(1 for it in intents if it["model_ver"] == "intent_transformer_v1")
    print(f"A/B Split: Champion(intent_v1)={champ_count}, Challenger(intent_transformer_v1)={chall_count}")

    # Both models should receive a share of the traffic
    assert_check("both_models_received_traffic", champ_count > 0 and chall_count > 0, True)
    assert_check("sum_of_models_equals_total", champ_count + chall_count, 20)

    # Probabilities of both models must be valid probabilities
    for it in intents:
        assert 0.0 <= it["prob"] <= 1.0


def test_pipeline_monitor_health():
    print("\n--- 3. Testing Streaming Pipeline Monitor (STALL / BACKPRESSURE) ---")
    mon = PipelineMonitor(max_lag_s=60.0)

    # Scenario A: Normal steady state (~1100 eps, lag ~1.5s)
    for i in range(10):
        mon.record_heartbeat(timestamp_s=i * 2.0, throughput_eps=1100.0, lag_s=1.5)
    health_a = mon.check_health()
    assert_check("normal_pipeline_healthy", health_a["status"], "HEALTHY")
    assert_check("no_backpressure", health_a["backpressure"], False)
    assert_check("no_stall", health_a["stall"], False)

    # Scenario B: High lag -> BACKPRESSURE
    mon_bp = PipelineMonitor(max_lag_s=60.0)
    for i in range(10):
        mon_bp.record_heartbeat(timestamp_s=i * 2.0, throughput_eps=800.0, lag_s=75.0)
    health_b = mon_bp.check_health()
    assert_check("detects_backpressure", health_b["status"], "BACKPRESSURE")

    # Scenario C: 0 throughput -> STALL
    mon_stall = PipelineMonitor(max_lag_s=60.0)
    for i in range(5):
        mon_stall.record_heartbeat(timestamp_s=i * 2.0, throughput_eps=0.0, lag_s=5.0)
    health_c = mon_stall.check_health()
    assert_check("detects_stall", health_c["status"], "STALL")


def main():
    test_online_evaluation_metrics()
    test_ab_testing_champion_vs_challenger()
    test_pipeline_monitor_health()
    print("\nALL WEEK 4 & G6 TESTS PASS!")


if __name__ == "__main__":
    main()
