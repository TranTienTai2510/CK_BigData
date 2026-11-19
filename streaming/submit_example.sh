#!/bin/bash
# G3 check: A submits a sample job reading events.raw
set -e
MASTER=${SPARK_MASTER:-spark://spark-master:7077}
PKG=org.apache.spark:spark-sql-kafka-0-10_2:3.5.5
/opt/bitnami/spark/bin/spark-submit --master $MASTER --packages $PKG \
  --conf spark.sql.session.timeZone=UTC \
  -c "
from pyspark.sql import SparkSession, functions as F
spark=SparkSession.builder.appName('sample-raw').getOrCreate()
df=spark.readStream.format('kafka').option('kafka.bootstrap.servers','redpanda:9092').option('subscribe','events.raw').option('startingOffsets','earliest').load()
q=df.groupBy(F.col('timestamp').cast('string')).count().writeStream.outputMode('complete').format('console').start()
q.awaitTermination(60)
"
echo SAMPLE_OK
