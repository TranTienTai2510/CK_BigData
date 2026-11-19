#!/bin/bash
# G3 check: A submits a sample job reading events.raw
# Run from host: docker exec -it <spark-master> /opt/project/streaming/submit_example.sh
set -e
MASTER=${SPARK_MASTER:-spark://spark-master:7077}
PKG=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5
APP=/opt/project/streaming
# spark-submit ships inside the pip pyspark package
SUBMIT="$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), "bin", "spark-submit"))')"
"$SUBMIT" --master $MASTER --packages $PKG \
  --conf spark.sql.session.timeZone=UTC \
  $APP/sample_raw.py \
  --bootstrap "${KAFKA_BOOTSTRAP:-redpanda:9092}" \
  --topic events.raw \
  --timeout-s 60
echo SAMPLE_OK
