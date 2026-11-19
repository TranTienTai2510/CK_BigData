"""Q0: validate events.raw against canonical contract -> events.clean / events.dlq (B)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import functions as F, types as T
from common.canonical import EVENT_SCHEMA
from common.contract import CANONICAL_RULES, failed_rules_col

def build(spark, bootstrap, raw_topic="events.raw", clean_topic="events.clean",
          dlq_topic="events.dlq", starting="latest"):
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", bootstrap)
           .option("subscribe", raw_topic)
           .option("startingOffsets", starting).load())
    val = F.from_json(F.col("value").cast("string"), EVENT_SCHEMA)
    df = raw.select(val.alias("e"), F.col("timestamp").alias("kafka_ts"))
    df = df.select("kafka_ts", "e.*")
    # schema_version may exist in payload but not in EVENT_SCHEMA -> keep lenient: extra ignored by from_json
    df = df.withColumn("failed_rules", failed_rules_col(CANONICAL_RULES))
    df = df.withColumn("is_bad", F.size("failed_rules") > 0)
    clean = df.filter(~F.col("is_bad")).drop("failed_rules", "is_bad")
    dlq = (df.filter(F.col("is_bad"))
           .select("shop_id", "session_id", "event_id", "seq_no", "event_type",
                   "event_ts", "failed_rules", "kafka_ts"))
    def to_kafka(d, topic):
        return (d.select(F.to_json(F.struct(*d.columns)).alias("value"))
                 .selectExpr("CAST(value AS STRING)"))
    cq = (to_kafka(clean, clean_topic).writeStream.format("kafka")
          .option("kafka.bootstrap.servers", bootstrap).option("topic", clean_topic)
          .option("checkpointLocation", "/opt/project/checkpoints/q0-clean")
          .outputMode("append").start())
    dq = (to_kafka(dlq, dlq_topic).writeStream.format("kafka")
          .option("kafka.bootstrap.servers", bootstrap).option("topic", dlq_topic)
          .option("checkpointLocation", "/opt/project/checkpoints/q0-dlq")
          .outputMode("append").start())
    return [cq, dq]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"))
    ap.add_argument("--starting", default="latest")
    a = ap.parse_args()
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("q0-validate")
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    qs = build(spark, a.bootstrap, starting=a.starting)
    for q in qs: q.awaitTermination()
