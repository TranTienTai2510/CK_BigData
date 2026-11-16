#!/bin/bash
# B: kill spark-worker mid-stream, restart, verify no loss via checkpoints
set -e
echo "start stack"; docker compose up -d redpanda clickhouse spark-master spark-worker
sleep 10
python simulator.py --speed-factor 200 --limit 5000 &
SIM=$!
sleep 15
echo "killing worker..."; docker kill ck_td1-spark-worker-1 || docker kill $(docker ps -q --filter name=spark-worker | head -1)
sleep 5
docker compose up -d spark-worker
wait $SIM || true
echo "RECOVERY_DONE: compare rt.events count vs sent; checkpoints at spark-checkpoints volume"
