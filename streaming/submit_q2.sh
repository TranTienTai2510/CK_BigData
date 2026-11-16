#!/bin/bash
# A: submit Q2 intent inference streaming query to Spark standalone.
# Usage: ./streaming/submit_q2.sh
# Run from host: docker exec -it <spark-master> /opt/project/streaming/submit_q2.sh
set -e
MASTER=${SPARK_MASTER:-spark://spark-master:7077}
PKG=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5
APP=/opt/project/streaming
SUBMIT="$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), "bin", "spark-submit"))')"

"$SUBMIT" --master $MASTER --packages $PKG \
  --conf spark.sql.session.timeZone=UTC \
  $APP/q2_intent.py \
  --bootstrap ${KAFKA_BOOTSTRAP:-redpanda:9092} \
  --in-topic events.clean \
  --pred-topic pred.intent \
  --closed-topic session.closed \
  --model models/intent_v1.onnx \
  --spec models/feature_spec.json
