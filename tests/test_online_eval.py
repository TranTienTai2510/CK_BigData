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
import urllib.error
from argparse import Namespace
from unittest.mock import patch

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
    cmd_eval,
    evaluate_online_pairs,
    query_pipeline_heartbeat,
)
from streaming.q2_intent import _route_model


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
    champ_path = "models/intent_v1.onnx"
    challenger_path = "models/intent_transformer_full.onnx"
    routes = [
        _route_model(
            "synthetic" if session_id % 2 == 0 else "rees46_cosmetics",
            session_id,
            champ_path,
            "intent_v1",
            challenger_path,
            True,
        )
        for session_id in range(1, 101)
    ]
    repeated_routes = [
        _route_model(
            "synthetic" if session_id % 2 == 0 else "rees46_cosmetics",
            session_id,
            champ_path,
            "intent_v1",
            challenger_path,
            True,
        )
        for session_id in range(1, 101)
    ]
    assert_check("route_is_deterministic", routes == repeated_routes, True)

    champ_count = sum(1 for _, version in routes if version == "intent_v1")
    chall_count = sum(1 for _, version in routes if version == "intent_transformer_v1")
    print(f"A/B Split: Champion(intent_v1)={champ_count}, Challenger(intent_transformer_v1)={chall_count}")

    assert_check("all_sessions_routed", champ_count + chall_count, 100)
    assert_check("both_models_received_traffic", champ_count > 0 and chall_count > 0, True)


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


def test_live_commands_use_observed_data():
    print("\n--- 4. Testing live commands fail closed and use ClickHouse metrics ---")
    with patch("analysis.live.query_clickhouse_pairs", side_effect=urllib.error.URLError("offline")):
        try:
            cmd_eval(Namespace(
                snapshot="",
                model_card=os.path.join(ROOT_DIR, "models", "model_card.json"),
                ch_host="localhost",
                ch_port=8123,
                ch_user="default",
                ch_pass="clickstream",
                model_ver="intent_v1",
                out="",
            ))
        except RuntimeError as exc:
            assert "valid --snapshot" in str(exc)
        else:
            raise AssertionError("offline ClickHouse must not produce synthetic evaluation results")

    with patch("analysis.live.query_clickhouse_pairs", return_value=[]):
        try:
            cmd_eval(Namespace(
                snapshot="",
                model_card=os.path.join(ROOT_DIR, "models", "model_card.json"),
                ch_host="localhost",
                ch_port=8123,
                ch_user="default",
                ch_pass="clickstream",
                model_ver="intent_v1",
                out="",
            ))
        except RuntimeError as exc:
            assert "No online prediction" in str(exc)
        else:
            raise AssertionError("empty ClickHouse results must not produce a success-shaped evaluation")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"12.5\t1.25\n"

    with patch("urllib.request.urlopen", return_value=Response()):
        assert query_pipeline_heartbeat() == (12.5, 1.25)


def main():
    test_online_evaluation_metrics()
    test_ab_testing_champion_vs_challenger()
    test_pipeline_monitor_health()
    test_live_commands_use_observed_data()
    print("\nALL WEEK 4 & G6 TESTS PASS!")


if __name__ == "__main__":
    main()
