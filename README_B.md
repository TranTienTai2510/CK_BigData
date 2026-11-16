# B — Platform & Streaming (runbook)

## 0. Chuẩn môi trường
`python -m common.runtime_env` đã PASS (Python 3.10, PySpark 3.5.5, JDK 17, HADOOP_HOME `D:\hadoop`). WSL2 `.wslconfig` 10GB đã có. `SPARK_HOME=C:\spark-3.5.8-bin-hadoop3 (will be ignored)` là bình thường — `common/runtime_env.py` tự bỏ qua để dùng đúng jar của pip `pyspark==3.5.5`.

Cài thêm thư viện còn thiếu cho batch + simulator + ML (chạy 1 lần trong `.venv`):
```powershell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install "pyspark==3.5.5" pandas==2.2.3 pyarrow==17.0.0 kafka-python==2.0.2 numpy scikit-learn scipy torch onnx "onnxruntime==1.20.1" kagglehub
python -c "import pyspark,pandas,pyarrow,kafka,numpy,sklearn,scipy,torch,onnx,onnxruntime,kagglehub; print('DEPS_OK')"
```

## 1. Build + up (G1)
```powershell
docker build -f Dockerfile.spark -t clickstream-spark:3.5.5 .
docker compose up -d
# Tìm đúng tên container redpanda (compose tự đặt tên, vd ck_bigdata-redpanda-1):
docker ps --format "{{.Names}}" | Select-String "redpanda"
# Liệt kê topic — thay <ten-container> bằng tên tìm được ở trên, KHÔNG giữ dấu <>:
docker exec -it <ten-container-redpanda> rpk topic list -X brokers=redpanda:9092
# đủ 8 topic (raw/clean/orders/kpi/pred/closed/metrics/dlq)
```
Nạp DDL ClickHouse (chạy trong PowerShell, dùng pipe thay cho `<` vì PowerShell không hỗ trợ `<`):
```powershell
$ch = (docker ps --format "{{.Names}}" | Select-String "clickhouse" | Select-Object -First 1).ToString().Trim()
Get-Content clickhouse/schema.sql -Raw | docker exec -i $ch clickhouse-client -u default --password clickstream --multiquery
Get-Content clickhouse/kafka_ingest.sql -Raw | docker exec -i $ch clickhouse-client -u default --password clickstream --multiquery
Get-Content clickhouse/views.sql -Raw | docker exec -i $ch clickhouse-client -u default --password clickstream --multiquery
```

## 2. Simulator (G1: ~1100 ev/s, 3 shop, fold stress)
```bash
pip install kafka-python pandas pyarrow --break-system-packages
python simulator.py --shops synthetic,rees46_cosmetics,olist --speed-factor 50
python simulator.py --mode orders --shops olist --speed-factor 100
python simulator.py --fold 5 --speed-factor 500 --late-pct 0.03 --dup-pct 0.02
python simulator.py --limit 2000 --dry-run
# check: SELECT count() FROM rt.v_funnel_1m
```

## 3. Spark Q0/Q1 (G3 image + submit)
```bash
./streaming/submit.sh all
# G3 check cho A:
./streaming/submit_example.sh
# checkpoints: volume spark-checkpoints (/opt/project/checkpoints/q0-*, q1-*)
```

## 4. Grafana
Mở `http://localhost:3000` (admin/admin), datasource ClickHouse tự provision, 3 dashboard: KPI, SLA Olist, Model online (fake-data tolerant tới G5).

## 5. Stress + recovery (Tuần 5)
```bash
./tools/stress.sh            # fold 1/5/10/20 -> tools/stress_results.csv (throughput, p95, RAM)
./tools/recovery_test.sh     # kill worker -> restart -> checkpoint, không mất/không trùng
```

## 6. Test
```bash
python tests/test_simulator_contract.py
python tests/test_streaming_contract.py   # cần Spark local
```

## Bàn giao
- G1: compose up + 8 topic + simulator chạy + `rt.v_funnel_1m` có số.
- G3: image `clickstream-spark:3.5.5` (pandas/pyarrow/onnxruntime) + `streaming/submit*.sh` + volume checkpoint.
- G6: `analysis/live.py monitor` (của A) 60 phút không STALL/BACKPRESSURE.
