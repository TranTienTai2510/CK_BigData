# B — Platform & Streaming (runbook)

## 0. Chuẩn môi trường
Môi trường Spark local trên Windows cần Python 3.10, PySpark 3.5.5, JDK 17, `winutils.exe` + `hadoop.dll` trong `D:\hadoop\bin`. Chạy `python -m common.runtime_env` để xác nhận. Spark trong Docker dùng JDK/PySpark được đóng gói trong image; master và worker chạy foreground bằng `spark-class`.

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
Get-Content clickhouse/schema.sql -Raw | docker exec -i $ch clickhouse client --user default --password clickstream --multiquery
Get-Content clickhouse/kafka_ingest.sql -Raw | docker exec -i $ch clickhouse client --user default --password clickstream --multiquery
Get-Content clickhouse/views.sql -Raw | docker exec -i $ch clickhouse client --user default --password clickstream --multiquery
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
```powershell
$spark = (docker compose ps -q spark-master).Trim()
$sparkSubmit = "/usr/local/lib/python3.10/site-packages/pyspark/bin/spark-submit"
docker exec -d $spark $sparkSubmit --master spark://spark-master:7077 --driver-memory 512m --executor-memory 512m --total-executor-cores 2 --repositories https://repo.maven.apache.org/maven2 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 --exclude-packages org.apache.hadoop:hadoop-client-api,org.apache.hadoop:hadoop-client-runtime /opt/project/streaming/all_jobs.py
# G3 check cho A:
docker exec -it $spark /opt/project/streaming/submit_example.sh
# checkpoints: volume spark-checkpoints (/opt/project/checkpoints/q0-*, q1-*)
```

`docker-compose.yml` mount các thư mục `common/`, `ml/`, `streaming/`, `models/` vào master/worker. `all_jobs.py` gom Q0/Q1/Q2 vào một Spark application để giảm RAM trên máy demo; Spark UI hiển thị app `clickstream-q0-q1-q2`. Trước khi chạy Q2, cần có `models/intent_v1.onnx` khớp với `models/feature_spec.json`; nếu chưa có model, huấn luyện/xuất bằng `python ml/train_dl.py`.

Trong `all_jobs.py`, watermark KPI là 10 giây để cửa sổ 10 giây xuất hiện trong lượt demo ngắn; các checkpoint Q1/Q2 tích hợp là `q1-kpi-e2e` và `q2-intent-e2e`. Job Q1 chạy độc lập vẫn giữ watermark mặc định 10 phút và checkpoint riêng. Simulator tăng event-time theo `--speed-factor`; `--max-sleep` chỉ giới hạn thời gian chờ thực tế.

## 4. Grafana
Mở `http://localhost:3000` (admin/admin), datasource ClickHouse và dashboard JSON được provision từ `grafana/provisioning/` và `grafana/dashboards/`. ClickHouse HTTP: `http://localhost:8123` (user `default`, password `clickstream`).

`python analysis/live.py eval --out data/prefix/_online_eval.json` và `python analysis/live.py monitor` truy vấn ClickHouse thật; nếu ClickHouse không truy cập được, lệnh eval báo lỗi thay vì sinh dữ liệu đánh giá giả.

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
