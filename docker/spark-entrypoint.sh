#!/bin/bash
# Entrypoint: SPARK_MODE=master | worker (B).
# The pip pyspark distribution includes spark-class, but not sbin/start-*.sh.
set -e
export SPARK_HOME="${SPARK_HOME:-$(python -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')}"
MODE="${SPARK_MODE:-master}"
if [ "$MODE" = "master" ]; then
  exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
    --host "${SPARK_MASTER_HOST:-spark-master}" \
    --port "${SPARK_MASTER_PORT:-7077}" --webui-port "${SPARK_MASTER_WEBUI_PORT:-8080}"
fi
if [ "$MODE" = "worker" ]; then
  exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
    "${SPARK_MASTER_URL:-spark://spark-master:7077}" \
    --memory "${SPARK_WORKER_MEMORY:-2g}" --cores "${SPARK_WORKER_CORES:-2}"
fi
echo "Unknown SPARK_MODE=$MODE (want master|worker)" >&2
exit 1
