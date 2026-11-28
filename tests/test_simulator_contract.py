"""B: simulator payload matches canonical contract (offline, no Kafka)."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from datetime import datetime, timezone, timedelta
import pandas as pd
from simulator import _scaled_gap, build_payload, load_sorted, SCHEMA_VERSION


def test_build_payload_keys():
    row = pd.Series({"shop_id":"synthetic","session_id":1,"event_id":5,"seq_no":1,"customer_id":9,
        "event_type":"page_view","event_ts":pd.Timestamp("2024-01-01",tz="UTC"),"ts_raw":pd.Timestamp("2024-01-01",tz="UTC"),
        "product_id":7.0,"category":"c","brand":"b","price":10.0,"qty":None,"ym":"2024-01"})
    now = datetime(2026,1,1,tzinfo=timezone.utc)
    p = build_payload(row, now, now - row["event_ts"].to_pydatetime())
    assert p["schema_version"]==SCHEMA_VERSION and p["ym"]=="2026-01" and p["product"]==7
    assert isinstance(p["product_id"], int)
    print("SIM_CONTRACT_OK")


def test_load_sorted_restores_partition_shop_id():
    with tempfile.TemporaryDirectory() as root:
        partition = Path(root) / "shop_id=synthetic" / "ym=2024-01"
        partition.mkdir(parents=True)
        path = partition / "part-00000.parquet"
        pd.DataFrame({
            "event_id": [5],
            "event_ts": [pd.Timestamp("2024-01-01", tz="UTC")],
        }).to_parquet(path, index=False)
        events = load_sorted([str(path)])
        assert events["shop_id"].tolist() == ["synthetic"]
    print("SIM_PARTITION_SHOP_ID_OK")


def test_sleep_cap_does_not_cap_event_time():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    event_gap, sleep_gap = _scaled_gap(
        start, start + timedelta(hours=1), speed_factor=3600, max_sleep=0.01
    )
    assert event_gap == 1
    assert sleep_gap == 0.01
    print("SIM_TIME_WARP_OK")


if __name__=="__main__":
    test_build_payload_keys()
    test_load_sorted_restores_partition_shop_id()
    test_sleep_cap_does_not_cap_event_time()
