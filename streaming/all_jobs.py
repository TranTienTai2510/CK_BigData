"""Run Q0, Q1, and Q2 in one Spark application for constrained demo hosts."""
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pyspark.sql import SparkSession

from streaming.q0_validate import build as build_q0
from streaming.q1_kpi import build as build_q1
from streaming.q2_intent import build as build_q2


def main():
    parser = argparse.ArgumentParser(description="Run all clickstream streaming queries.")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    parser.add_argument("--starting", default="latest")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("clickstream-q0-q1-q2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    queries = []
    try:
        queries.extend(build_q0(spark, args.bootstrap, starting=args.starting))
        queries.extend(
            build_q1(
                spark,
                args.bootstrap,
                watermark="10 seconds",
                starting=args.starting,
                checkpoint_dir="/opt/project/checkpoints/q1-kpi-e2e",
            )
        )
        queries.extend(
            build_q2(
                spark,
                args.bootstrap,
                starting=args.starting,
                checkpoint_dir="/opt/project/checkpoints/q2-intent-e2e",
            )
        )
        spark.streams.awaitAnyTermination()
    finally:
        for query in queries:
            if query.isActive:
                query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
