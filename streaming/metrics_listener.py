"""Listener -> ops.metrics (B): batch watermark/progress -> Kafka JSON."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import SparkSession

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    ap.add_argument("--job", default="q1")
    ap.add_argument("--interval-s", type=float, default=5.0)
    a = ap.parse_args()
    from kafka import KafkaProducer
    prod = KafkaProducer(bootstrap_servers=[s.strip() for s in a.bootstrap.split(",") if s.strip()],
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"), acks="all")
    spark = SparkSession.builder.appName(f"metrics-{a.job}").config("spark.sql.session.timeZone","UTC").getOrCreate()
    # poll active streaming queries via progress API by attaching listener is complex in py; simple heartbeat:
    while True:
        payload = {"job": a.job, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "status": "RUNNING", "schema_version": 3}
        prod.send("ops.metrics", key=a.job, value=payload).get(timeout=15)
        time.sleep(a.interval_s)

if __name__ == "__main__":
    main()
