"""G3 check: sample job reading events.raw (submitted by A to verify the image).

Usage (inside spark-master container):
    /opt/project/streaming/submit_example.sh
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyspark.sql import SparkSession, functions as F
from common.messages import TOPIC_EVENTS_RAW


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    ap.add_argument("--topic", default=TOPIC_EVENTS_RAW)
    ap.add_argument("--timeout-s", type=int, default=60)
    a = ap.parse_args()

    spark = (
        SparkSession.builder.appName("sample-raw")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", a.bootstrap)
        .option("subscribe", a.topic)
        .option("startingOffsets", "earliest")
        .load()
    )
    q = (
        df.groupBy(F.col("timestamp").cast("string").alias("ts"))
        .count()
        .writeStream.outputMode("complete")
        .format("console")
        .start()
    )
    q.awaitTermination(a.timeout_s)
    q.stop()
    print("SAMPLE_OK")


if __name__ == "__main__":
    main()
