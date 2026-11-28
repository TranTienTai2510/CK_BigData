# Huong dan chay du an

Tai lieu nay huong dan chay day du du an tren Windows bang PowerShell. Du an gom lop batch xu ly du lieu, he thong streaming Redpanda + Spark, ClickHouse, Grafana va mo hinh du doan y dinh mua.

## 1. Yeu cau

- Windows 10/11, 16 GB RAM tro len
- Docker Desktop dang chay
- Python 3.10
- JDK 17
- Git
- Khoang 10-20 GB dung luong trong

Kiem tra cac cong cu:

```powershell
docker --version
docker compose version
python --version
java -version
git --version
```

Neu Python 3.10 chua co trong PATH, tao moi truong ao:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pyspark==3.5.5 pandas==2.2.3 pyarrow==17.0.0 kafka-python==2.0.2
pip install numpy scikit-learn scipy torch onnx onnxruntime==1.20.1 kagglehub
```

Moi script Python tren Windows se tu cau hinh JDK va Hadoop. Kiem tra:

```powershell
python -m common.runtime_env
```

## 2. Cau hinh du lieu

Doi voi che do demo streaming, can co du lieu trong `data/unified/`. Du lieu lon khong duoc commit tren GitHub, vi du muc `data/` bi ignore.

Tai du lieu Olist neu can:

```powershell
python tools\download_olist.py
```

Neu da co cac du lieu raw tren may:

```text
data/raw/
data/raw_rees46/
data/raw_olist/
```

Chay lop batch:

```powershell
python batch\preprocess.py --raw data/raw --out data/curated --strict
python batch\adapters\synthetic.py --curated data/curated --out data/unified --strict
python batch\adapters\rees46.py --raw data/raw_rees46 --out data/unified --strict --driver-memory 4g --cores 4
python batch\adapters\olist.py --raw data/raw_olist --out data/unified --strict
python batch\check_unified.py --unified data/unified --strict
```

Tao tap tin dac trung va huan luyen mo hinh:

```powershell
python ml\prefixes.py --unified data/unified --out data/prefix
python ml\baselines.py --prefix data/prefix --out data/prefix\_baselines.json
python ml\train_dl.py --prefix data/prefix --out-dir models --report data/prefix\_dl_ablation.json
```

Mo hinh trien khai streaming can cac file:

```text
models/intent_v1.onnx
models/model_card.json
models/feature_spec.json
```

Dung Step 3 cho demo neu may da co san cac file tren.

## 3. Khoi dong thanh phan Docker

Dung PowerShell tai thu muc goc cua du an.

```powershell
docker build -f Dockerfile.spark -t clickstream-spark:3.5.5 .
docker compose up -d
docker compose ps
```

Chu dao cua stack gom:

- Redpanda: Kafka-compatible broker
- Spark master va worker
- ClickHouse
- Grafana
- Q0 validation
- Q1 KPI
- Q2 purchase-intent prediction

Kiem tra container:

```powershell
docker ps --format "table {{.Names}}`t{{.Status}}"
```

Kiem tra cac topic duoc tao:

```powershell
$redpanda = docker ps --format "{{.Names}}" | Select-String "redpanda" | Select-Object -First 1
$redpanda = $redpanda.ToString().Trim()
docker exec $redpanda rpk topic list -X brokers=redpanda:9092
```

Can thay 8 topic: `events.raw`, `events.clean`, `order_events.raw`, `agg.kpi`, `pred.intent`, `session.closed`, `ops.metrics` va `events.dlq`.

## 4. Nap schema ClickHouse

PowerShell can dung pipe thay vi dau `<`.

```powershell
$clickhouse = docker ps --format "{{.Names}}" | Select-String "clickhouse" | Select-Object -First 1
$clickhouse = $clickhouse.ToString().Trim()

Get-Content clickhouse\schema.sql -Raw | docker exec -i $clickhouse clickhouse client --user default --password clickstream --multiquery
Get-Content clickhouse\kafka_ingest.sql -Raw | docker exec -i $clickhouse clickhouse client --user default --password clickstream --multiquery
Get-Content clickhouse\views.sql -Raw | docker exec -i $clickhouse clickhouse client --user default --password clickstream --multiquery
```

## 5. Khoi dong streaming jobs

Sau khi cac container on dinh, truyen cac job Spark. Co the dung mot application gom Q0, Q1 va Q2:

```powershell
$spark = docker compose ps -q spark-master
$submit = "/usr/local/lib/python3.10/site-packages/pyspark/bin/spark-submit"
docker exec -d $spark $submit --master spark://spark-master:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 --conf spark.cores.max=3 --conf spark.executor.cores=1 --conf spark.executor.memory=768m --conf spark.ui.enabled=false /opt/project/streaming/all_jobs.py --bootstrap redpanda:9092 --starting latest
```

Hoac chay tung job:

```powershell
$spark = docker compose ps -q spark-master
docker exec -d $spark /opt/project/streaming/submit.sh all
```

Theo doi trang thai:

```powershell
docker compose logs --tail=100 q0-validate q1-kpi q2-intent
docker compose ps
```

Spark UI: `http://localhost:4040` hoac `http://localhost:8080`.

## 6. Giai phong su kien mau

Gui mot luong su kien demo den Redpanda:

```powershell
python simulator.py --shops synthetic,rees46_cosmetics,olist --speed-factor 50
```

Cac kịch ban thuong dung:

```powershell
# Chi gui mot gioi han luong moi khi test
python simulator.py --limit 2000 --speed-factor 50

# Luu lai du lieu order cua Olist
python simulator.py --mode orders --shops olist --speed-factor 100

# Kich thuoc stress co late event va duplicate event
python simulator.py --fold 5 --speed-factor 500 --late-pct 0.03 --dup-pct 0.02

# Chi kiem tra du lieu, khong gui Kafka
python simulator.py --limit 2000 --dry-run
```

Trong khi simulator chay, kiem tra du lieu trong ClickHouse:

```powershell
$clickhouse = docker ps --format "{{.Names}}" | Select-String "clickhouse" | Select-Object -First 1
$clickhouse = $clickhouse.ToString().Trim()
docker exec -i $clickhouse clickhouse client --user default --password clickstream --query "SELECT count() AS so_su_kien FROM rt.events"
docker exec -i $clickhouse clickhouse client --user default --password clickstream --query "SELECT * FROM rt.v_funnel_1m LIMIT 20"
```

## 7. Mo dashboard Grafana

Mo trinh duyet:

```text
http://localhost:3000
```

Tai khoan mac dinh:

```text
Ten dang nhap: admin
Mat khau: admin
```

Datasource ClickHouse da duoc provision tu `grafana/provisioning/datasources/clickhouse.yml`. ClickHouse HTTP:

```text
http://localhost:8123
```

## 8. Chay kiem thu

Cac test khong can du lieu lon hoac Docker:

```powershell
python tests\test_simulator_contract.py
python tests\test_features_parity.py
python tests\test_q2_parity.py
python tests\test_online_eval.py
python tests\test_clickhouse_schema.py
```

Test cap module voi mo hinh va du lieu da san co:

```powershell
python tests\test_onnx_parity.py
python tests\test_streaming_contract.py
```

Kiem tra thuc hien pipeline va live evaluation:

```powershell
python analysis\live.py monitor --dry-run
python analysis\live.py eval --out data/prefix/_online_eval.json
python analysis\live.py tail --threshold 0.5
```

## 9. Dung va khoi dong lai

Dung toan bo stack:

```powershell
docker compose down
```

Giu lai checkpoint va volume khi chi khoi dong lai:

```powershell
docker compose down
docker compose up -d
```

Xoa toan bo du lieu Docker neu muon demo tu dau:

```powershell
docker compose down -v
```

Lenh tren se xoa volume Redpanda, ClickHouse, Grafana va checkpoint Spark.

## 10. Loi thuong gap

### Docker compose khong khoi dong duoc

```powershell
docker compose down
docker builder prune
docker compose build --no-cache spark-master
docker compose up -d
```

Dam bao Docker Desktop dang chay va phan bo du RAM cho Docker.

### Spark bao khong tim thay Java

```powershell
python -m common.runtime_env
```

Co the chi dinh ro JDK:

```powershell
$env:CLICKSTREAM_JAVA_HOME = "C:\Program Files\Java\jdk-17"
python -m common.runtime_env
```

### Khong co du lieu trong ClickHouse

- Xac nhan simulator dang gui den `localhost:19092`.
- Xac nhan container Redpanda, Q0 va Q1 dang running.
- Xac nhan da nap cac file schema ClickHouse.
- Cho khoang 10-30 giay sau khi bat dau simulator, sau do truy van lai.

### Q2 khong du doan duoc

Kiem tra ba file model trong `models/`. Neu thieu `intent_v1.onnx`, chay lai:

```powershell
python ml\train_dl.py --prefix data/prefix --out-dir models --report data/prefix\_dl_ablation.json
```
