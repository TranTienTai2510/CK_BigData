"""Q1: 10s KPI windows from events.clean -> agg.kpi (B). Watermark + dedup."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import functions as F, types as T
from common.canonical import EVENT_SCHEMA

def build(spark, bootstrap, in_topic="events.clean", out_topic="agg.kpi",
          watermark="10 minutes", starting="latest"):
    src = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", bootstrap)
           .option("subscribe", in_topic)
           .option("startingOffsets", starting).load())
    df = src.select(F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("e")).select("e.*")
    df = df.withWatermark("event_ts", watermark).dropDuplicates(["shop_id", "event_id"])
    kpi = (df.groupBy(F.window("event_ts", "10 seconds"), "shop_id").agg(
        F.count("*").alias("n_events"),
        F.countDistinct("session_id").alias("n_sessions"),
        F.sum((F.col("event_type") == "purchase").cast("int")).alias("n_purchase"),
        F.sum("amount_usd").alias("revenue_usd"),
        F.sum((F.col("event_type") == "add_to_cart").cast("int")).alias("n_add"),
        F.sum((F.col("event_type") == "checkout").cast("int")).alias("n_checkout"),
    ).select(F.col("window.start").alias("window_start"),
             F.col("window.end").alias("window_end"), "shop_id",
             "n_events", "n_sessions", "n_purchase", "revenue_usd", "n_add", "n_checkout"))
    out = kpi.select(F.to_json(F.struct(*kpi.columns)).alias("value"))
    q = (out.writeStream.format("kafka").option("kafka.bootstrap.servers", bootstrap)
         .option("topic", out_topic)
         .option("checkpointLocation", "/opt/project/checkpoints/q1-kpi")
         .outputMode("append").start())
    return [q]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    ap.add_argument("--watermark", default="10 minutes")
    ap.add_argument("--starting", default="latest")
    a = ap.parse_args()
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("q1-kpi")
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    for q in build(spark, a.bootstrap, watermark=a.watermark, starting=a.starting):
        q.awaitTermination()
