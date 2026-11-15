#!/bin/bash
# B: stress matrix --fold 1/5/10/20, record throughput/p95/RAM
set -e
for f in 1 5 10 20; do
  echo "=== fold=$f ==="
  START=$(date +%s)
  timeout 120 python simulator.py --fold $f --speed-factor 500 --max-sleep 0.1 --limit 20000 || true
  END=$(date +%s); echo "fold=$f elapsed=$((END-START))s events=20000" | tee -a tools/stress_results.csv
  docker stats --no-stream --format "{{.Name}} {{.MemUsage}}" | tee -a tools/stress_results.csv
done
