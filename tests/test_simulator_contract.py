"""B: simulator payload matches canonical contract (offline, no Kafka)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from datetime import datetime, timezone, timedelta
import pandas as pd
from simulator import build_payload, SCHEMA_VERSION
def test_build_payload_keys():
    row = pd.Series({"shop_id":"synthetic","session_id":1,"event_id":5,"seq_no":1,"customer_id":9,
        "event_type":"page_view","event_ts":pd.Timestamp("2024-01-01",tz="UTC"),"ts_raw":pd.Timestamp("2024-01-01",tz="UTC"),
        "product_id":7,"category":"c","brand":"b","price":10.0,"qty":None,"ym":"2024-01"})
    now = datetime(2026,1,1,tzinfo=timezone.utc)
    p = build_payload(row, now, now - row["event_ts"].to_pydatetime())
    assert p["schema_version"]==SCHEMA_VERSION and p["ym"]=="2026-01" and p["product"]==7
    print("SIM_CONTRACT_OK")
if __name__=="__main__":
    test_build_payload_keys()
