"""B: Q0/Q1 logic offline with local Spark (no Kafka)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.canonical import get_spark, EVENT_SCHEMA
from common.contract import CANONICAL_RULES, failed_rules_col
from pyspark.sql import functions as F
def main():
    spark = get_spark("test-stream", driver_memory="2g", cores=2, shuffle_partitions=4)
    rows=[("synthetic",1,1,1,1,"page_view",7,"c","b",10.0,None,None,None,None,None,None,None,None,None,"2024-01-01 00:00:00","2024-01-01 00:00:00","2024-01-01 00:00:00",False,"2024-01"),
          ("synthetic",2,1,2,1,"weird",None,None,None,None,None,None,None,None,None,None,None,None,None,"2024-01-01 00:00:01","2024-01-01 00:00:01","2024-01-01 00:00:01",False,"2024-01")]
    cols=[f.name for f in EVENT_SCHEMA.fields]
    df=spark.createDataFrame(rows, schema=EVENT_SCHEMA)
    df=df.withColumn("failed_rules", failed_rules_col(CANONICAL_RULES))
    bad=df.filter(F.size("failed_rules")>0).count(); good=df.filter(F.size("failed_rules")==0).count()
    assert good==1 and bad==1, (good,bad)
    kpi=(df.withWatermark("event_ts","10 minutes").dropDuplicates(["shop_id","event_id"])
         .groupBy(F.window("event_ts","10 seconds"),"shop_id").agg(F.count("*").alias("n_events")).collect())
    assert sum(r.n_events for r in kpi)==2
    print("STREAM_CONTRACT_OK"); spark.stop()
if __name__=="__main__": main()
