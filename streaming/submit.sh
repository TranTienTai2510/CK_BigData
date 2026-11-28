#!/bin/bash
# B: submit streaming queries to Spark standalone. Usage: ./streaming/submit.sh q0|q1|all
# Run from host: docker exec -it <spark-master> /opt/project/streaming/submit.sh all
set -e
MASTER=${SPARK_MASTER:-spark://spark-master:7077}
PKG=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5
COMMON_ARGS=(--master "$MASTER" --repositories https://repo.maven.apache.org/maven2 \
  --packages "$PKG" \
  --exclude-packages org.apache.hadoop:hadoop-client-api,org.apache.hadoop:hadoop-client-runtime)
APP=/opt/project/streaming
# spark-submit ships inside the pip pyspark package
SUBMIT="$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), "bin", "spark-submit"))')"
case "${1:-all}" in
  q0) "$SUBMIT" "${COMMON_ARGS[@]}" $APP/q0_validate.py ;;
  q1) "$SUBMIT" "${COMMON_ARGS[@]}" $APP/q1_kpi.py ;;
  all) "$SUBMIT" "${COMMON_ARGS[@]}" $APP/q0_validate.py &
       "$SUBMIT" "${COMMON_ARGS[@]}" $APP/q1_kpi.py & wait ;;
esac
