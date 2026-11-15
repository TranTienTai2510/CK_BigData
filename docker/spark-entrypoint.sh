#!/bin/bash
# Entrypoint: SPARK_MODE=master | worker (B).
# SPARK_HOME is resolved from the pip pyspark install (ships bin/ + sbin/ + jars/).
set -e
export SPARK_HOME="${SPARK_HOME:-$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), "..", "pyspark"))' 2>/dev/null || true)}"
if [ -z "$SPARK_HOME" ] || [ ! -d "$SPARK_HOME" ]; then
  PYDIR="$(python -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')"
  SPARK_HOME="$(cd "$PYDIR" && pwd)"
  export SPARK_HOME
fi
MODE="${SPARK_MODE:-master}"
if [ "$MODE" = "master" ]; then
  exec "$SPARK_HOME/sbin/start-master.sh" --host "${SPARK_MASTER_HOST:-spark-master}" \
    --port "${SPARK_MASTER_PORT:-7077}" --webui-port "${SPARK_MASTER_WEBUI_PORT:-8080}"
fi
if [ "$MODE" = "worker" ]; then
  exec "$SPARK_HOME/sbin/start-worker.sh" "${SPARK_MASTER_URL:-spark://spark-master:7077}" \
    --memory "${SPARK_WORKER_MEMORY:-2g}" --cores "${SPARK_WORKER_CORES:-2}"
fi
echo "Unknown SPARK_MODE=$MODE (want master|worker)" >&2
exit 1
