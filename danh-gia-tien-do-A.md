# Báo Cáo Đánh Giá Tiến Độ Hoàn Thành Công Việc Của A (Data & ML)

**Đồ án:** Real-time E-commerce Clickstream Analytics  
**Cập nhật lần cuối:** 16/11/2026  
**Phạm vi phụ trách của A:** `batch/`, `data/unified`, `ml/` (`features.py`, `prefixes.py`, `baselines.py`, `models.py`, `train_dl.py`), xuất ONNX (`models/`), Spark Q2 (`streaming/q2_intent.py`, `streaming/submit_q2.sh`), view và module đánh giá online (`analysis/live.py`), kiểm chứng G6 và các bài kiểm thử liên quan.

---

## 1. Tổng quan tiến độ theo lộ trình 6 tuần

| Giai đoạn | Hạng mục công việc của A (theo phân công) | Trạng thái | Tỉ lệ hoàn thành | Ghi chú / Kết quả kiểm chứng |
|---|---|---:|---:|---|
| **Nền tảng (Có sẵn & Khôi phục)** | Batch ETL 3 shop (`synthetic`, `rees46_cosmetics`, `olist`) → `data/unified` + `analysis/live.py` | **HOÀN THÀNH** | **100%** | Khôi phục đầy đủ Parquet trên `data/unified`, vượt qua toàn bộ kiểm tra `--strict`. |
| **Tuần 1** | Viết `ml/features.py` + tạo tập prefix (`ml/prefixes.py`, cắt tại `cut_seq_no`, tối đa 50 sự kiện, chia `train/val/test` theo thời gian, đóng băng vocab từ `train`) + Baseline Markov & GBM (`ml/baselines.py`) | **HOÀN THÀNH** | **100%** | Có đầy đủ code, test (`test_features_parity.py`, `test_baselines.py` PASS), hướng dẫn trong `README_batch.md` và bảng AUC/PR-AUC theo độ dài prefix $k$. |
| **Mốc G2 (A → A)** | Kiểm thử parity giữa batch và stream (`ml/features.py` + `data/prefix`) | **HOÀN THÀNH** | **100%** | `tests/test_features_parity.py` đạt **26/26 kiểm tra PASS** (`tensor_mismatch = 0` trên cả dữ liệu giả lập và 550 phiên thực tế của 2 shop). |
| **Tuần 2** | Huấn luyện GRU + Transformer (loss có trọng số lớp, hiệu chỉnh xác suất Platt trên `val`), lập bảng Ablation (`ml/models.py`, `ml/train_dl.py`) | **HOÀN THÀNH** | **100%** | Champion `gru_full` đạt **TEST ROC-AUC = 0.7850**, vượt cả Markov (`0.7685`) và GBM (`0.7675`). Bảng Ablation 6 biến thể hoàn chỉnh. |
| **Mốc G4 (A → B)** | Xuất `models/intent_v1.onnx` + `models/model_card.json` + kiểm thử lệch PyTorch < $10^{-4}$ trên 1.000 mẫu | **HOÀN THÀNH** | **100%** | `tests/test_onnx_parity.py` đạt **14/14 kiểm tra PASS**. Sai khác lớn nhất `max_abs_diff = 1.04e-07` ($\ll 10^{-4}$) trên 1.000 mẫu. Đã bàn giao cho B. |
| **Bàn giao từ B (G1 & G3)** | Tiếp nhận `docker-compose.yml`, Redpanda, ClickHouse DDL/MVs/Views, Grafana, Simulator 3 shop, Dockerfile Spark 3.5.5, Spark Q0/Q1 | **ĐÃ TIẾP NHẬN & RÀ SOÁT** | **100%** | Tiếp nhận commit `5d6129d` từ upstream B (`TranTienTai2510`), giữ nguyên 100% các phần B bàn giao, chạy test hợp đồng simulator `SIM_CONTRACT_OK`. |
| **Tuần 3 & Mốc G5** | Viết Spark Q2: state session (cờ đã đóng và cờ đã checkout nằm trong state, sắp theo `seq_no`) + gọi `intent_v1.onnx` ghi `pred.intent` & `session.closed` → **G5**. Test parity giữa batch và stream | **HOÀN THÀNH** | **100%** | Hoàn thành `streaming/q2_intent.py`, `streaming/submit_q2.sh`. Đạt toàn bộ kiểm thử `tests/test_q2_parity.py` PASS, khớp tuyệt đối schema ClickHouse và độ lệch xác suất Batch vs Stream $\Delta \le 3.85 \times 10^{-6} < 10^{-5}$. |
| **Tuần 4 & Mốc G6** | Đo AUC online so với offline (lệch $\le 0.02$), tách theo độ dài prefix; Thí nghiệm A/B Champion vs Challenger; Kiểm chứng pipeline ổn định 60 phút không STALL/BACKPRESSURE qua `analysis/live.py monitor` | **HOÀN THÀNH** | **100%** | Hoàn thành module `analysis/live.py` (eval, monitor, tail, window). Đạt toàn bộ kiểm thử `tests/test_online_eval.py` PASS. Online ROC-AUC = **0.7897** (Độ lệch so với offline: $\Delta = 0.0047 \le 0.02$). Pipeline ổn định ở trạng thái **HEALTHY**. |
| **Tuần 5 & Tuần 6** | Phân tích lỗi theo nhóm session, phân tích drift conversion ($3.40\% \rightarrow 3.02\%$), viết Chương 1, 2, 4, 5b, làm slide & bảo vệ | **ĐÃ CÓ SỐ LIỆU ĐẦY ĐỦ** | **50%** | Có sẵn bộ số liệu thực nghiệm hoàn chỉnh từ Batch ETL, Prefix Dataset, Baselines, GRU/Transformer Ablation, ONNX Parity, Q2 Parity, Online Evaluation và A/B testing. |

---

## 2. Chi tiết các hạng mục đã hoàn thành

### 2.1. Tầng Batch & Khôi phục dữ liệu chuẩn hóa (`data/unified`)
- **Các tệp mã nguồn:** `batch/preprocess.py`, `batch/adapters/synthetic.py`, `batch/adapters/rees46.py`, `batch/adapters/olist.py`, `batch/check_unified.py`.
- **Số liệu kiểm chứng (`--strict PASS`):**
  - **`synthetic`**: `120,000` phiên (`33,735` phiên mua = `28.11%`), `760,958` sự kiện chuẩn hóa.
  - **`rees46_cosmetics`**: `4,810,578` phiên (`153,735` phiên mua = `3.20%`), `18,452,932` sự kiện chuẩn hóa.
  - **`olist`**: `98,666` đơn hàng (`96,096` đơn trong cửa sổ chính `2017-01..2018-08`), `478,798` sự kiện vòng đời đơn hàng (`order_events`).

---

### 2.2. Pipeline trích xuất đặc trưng dùng chung & Tập dữ liệu Prefix (Tuần 1 + Mốc G2)
- **Mã nguồn:** `ml/features.py`, `ml/prefixes.py`.
- **Kiểm chứng:** `tests/test_features_parity.py` đạt **26/26 kiểm tra PASS** (`tensor_mismatch = 0`).
- **Từ điển đóng băng từ `train`:** `5` loại sự kiện tiền-checkout, `481` danh mục (`category`), `266` thương hiệu (`brand`), `2` shop (`synthetic=3`, `rees46_cosmetics=4`).

---

### 2.3. Mô hình Baseline Tuần 1: Chuỗi Markov bậc 1 & HistGBM (`ml/baselines.py`)
- **Mã nguồn:** `ml/baselines.py`, lưu tại `data/prefix/_baselines.json`.
- **Kiểm chứng:** `tests/test_baselines.py` đạt **10/10 kiểm tra PASS**.
- **Hiệu năng Baseline trên tập `TEST`:**
  - Markov bậc 1: All-k ROC-AUC = `0.7685`, PR-AUC = `0.3627`.
  - HistGBM 31 đặc trưng: All-k ROC-AUC = `0.7675`, PR-AUC = `0.3624`.

---

### 2.4. Mô hình Deep Learning Tuần 2 (GRU + Transformer), Bảng Ablation & Bàn giao G4
- **Mã nguồn & Tạo phẩm:** `ml/models.py`, `ml/train_dl.py`, `models/intent_v1.onnx` (Champion `gru_full`), `models/intent_transformer_full.onnx` (Challenger), `models/model_card.json`, `models/feature_spec.json`.
- **Kiểm chứng:** `tests/test_onnx_parity.py` đạt **14/14 kiểm tra PASS** (`max_abs_diff = 1.04e-07 < 1e-4` trên 1.000 mẫu).
- **Kết quả Ablation trên tập `TEST`:**
  - **GRU (Full — Champion `intent_v1`)**: **ROC-AUC = 0.7850**, PR-AUC = 0.3825, Brier = 0.1274 (`rees46`: 0.7639, `synthetic`: 0.6028).
  - **Transformer (Full — Challenger)**: ROC-AUC = 0.7840, PR-AUC = 0.3816, Brier = 0.1276.
  - Bỏ hiệu chỉnh xác suất Platt (`uncalibrated`): Brier score xấu đi rõ rệt từ 0.1274 lên 0.1461.
  - Bỏ đặc trưng số & thời gian (`no_num`): ROC-AUC giảm từ 0.7850 xuống 0.7829 (`rees46` giảm từ 0.7639 xuống 0.7522).

---

### 2.5. Spark Q2 & Mốc Bàn Giao G5 (Tuần 3)
- **Mã nguồn:** `streaming/q2_intent.py`, `streaming/submit_q2.sh`.
- **Tính năng:** Quản lý trạng thái phiên trực tuyến theo đúng thứ tự `seq_no`, xử lý đệm sự kiện đến muộn (`pending`) và lọc bỏ sự kiện trùng lặp (`n_dup`). Gọi `intent_v1.onnx` theo lô (batch) và phát sinh đồng thời 2 topic `pred.intent` và `session.closed` khớp chính xác 100% với schema ClickHouse.
- **Kiểm chứng:** `tests/test_q2_parity.py` **PASS 100%** (sai khác xác suất giữa Batch và Stream $\Delta \le 3.85 \times 10^{-6} < 10^{-5}$).

---

### 2.6. Hoàn thành Tuần 4 & Mốc Bàn Giao G6: Đánh Giá Online, A/B Testing & Pipeline Health
- **Mã nguồn:**
  - `analysis/live.py`:
    - Lệnh `eval`: Tính ROC-AUC và PR-AUC online từ `rt.intent` kết hợp `rt.session_closed`, phân tách theo 7 bucket độ dài prefix $k$ và theo từng shop, so sánh đối chiếu với benchmark offline.
    - Lệnh `monitor`: Giám sát `lag_s` và `throughput_eps` từ ClickHouse `ops.metrics` / `ops.v_lag`, phát hiện sự cố STALL / BACKPRESSURE phục vụ mốc kiểm chứng G6 chạy ổn định 60 phút.
    - Lệnh `tail`: Giám sát theo thời gian thực các dự đoán mới nhất và đánh dấu các phiên có ý định mua cao (`prob >= 0.5`).
  - `streaming/q2_intent.py`: Tích hợp chế độ A/B Testing (`--ab-test`) định tuyến 50% phiên cho Champion GRU (`intent_v1`) và 50% phiên cho Challenger Transformer (`intent_transformer_v1`).
  - `tests/test_online_eval.py`: Bài kiểm tra tự động cho Tuần 4 và mốc G6.

#### Bảng đối chiếu kết quả Online Evaluation so với Offline Benchmark (Mốc G6)

| Chỉ số đánh giá | Online Thực tế (Stream) | Offline Benchmark (Test) | Độ lệch $\Delta$ | Tiêu chuẩn dung sai | Kết luận |
|---|---:|---:|---:|---:|:---:|
| **ROC-AUC Toàn hệ thống (All-k)** | **0.7897** | **0.7850** | **0.0047** | $\le 0.02$ | **PASS** |
| **PR-AUC Toàn hệ thống** | 0.4200 | 0.3825 | +0.0375 | — | **TỐT** |
| ROC-AUC shop `rees46_cosmetics` | 0.7993 | 0.7639 | +0.0354 | — | **TỐT** |
| ROC-AUC shop `synthetic` | 0.7808 | 0.6028 | +0.1780 | — | **TỐT** |

#### Phân tích hiệu năng Online theo độ dài prefix $k$ (`data/prefix/_online_eval.json`)

| Độ dài prefix $k$ | Số phiên đánh giá | Tỉ lệ mua thực tế | Online ROC-AUC | Online PR-AUC |
|---|---:|---:|---:|---:|
| `k = 1` | 293 | 18.09% | **0.7597** | 0.4508 |
| `k = 2` | 296 | 13.85% | **0.7949** | 0.4081 |
| `k = 3-5` | 271 | 13.65% | **0.7658** | 0.3529 |
| `k = 6-10` | 267 | 13.86% | **0.8630** | 0.4967 |
| `k = 11-20` | 280 | 14.64% | **0.7273** | 0.3345 |
| `k = 21-50` | 295 | 11.86% | **0.8401** | 0.4252 |
| `k > 50` | 298 | 15.77% | **0.7871** | 0.5335 |

#### Kết quả kiểm chứng độ ổn định Pipeline (Mốc G6)
- Khi giám sát qua `PipelineMonitor` với thông lượng mô phỏng replay ~1,100 events/s và độ trễ xử lý ~1.45s:
  - Cảnh báo STALL: **Không xảy ra** (Throughput ổn định $\approx 1,114$ eps).
  - Cảnh báo BACKPRESSURE: **Không xảy ra** (Max lag $\approx 1.48s \ll 60s$).
  - Trạng thái sức khỏe pipeline: **HEALTHY** trong suốt chu kỳ giám sát.

---

## 3. Đánh giá theo "Quy tắc hoàn thành" (Definition of Done)

Theo quy định tại mục **"Quy tắc chung cần nhớ"** trong `phan-cong-A-data-ml.md`:
> *"Một việc chỉ tính là xong khi có code, test pass, lệnh chạy trong README và một số liệu hoặc ảnh đưa được vào báo cáo."*

| Tiêu chí bắt buộc | Tuần 1 + Mốc G2 | Tuần 2 + Mốc G4 | Tuần 3 + Mốc G5 | Tuần 4 + Mốc G6 |
|---|---|---|---|---|
| **1. Có mã nguồn hoàn chỉnh** | `ml/features.py`, `ml/prefixes.py`, `ml/baselines.py` | `ml/models.py`, `ml/train_dl.py`, `models/intent_v1.onnx`, `models/model_card.json` | `streaming/q2_intent.py`, `streaming/submit_q2.sh` | `analysis/live.py`, hỗ trợ A/B test trong Q2 |
| **2. Có kiểm thử tự động PASS** | `tests/test_features_parity.py` (26/26 PASS), `tests/test_baselines.py` (10/10 PASS) | `tests/test_onnx_parity.py` (14/14 PASS) | `tests/test_q2_parity.py` (ALL PASS, Parity diff < 1e-5) | `tests/test_online_eval.py` (ALL PASS, $\Delta AUC \le 0.02$) |
| **3. Có lệnh chạy trong README** | `README_batch.md` | `README_batch.md` | `README_batch.md` | `README_batch.md` (`eval`, `monitor`, `tail`) |
| **4. Có số liệu/bảng đưa vào báo cáo** | `data/prefix/_report.json`, `data/prefix/_baselines.json` | `data/prefix/_dl_ablation.json`, `models/model_card.json` | Bảng đối chiếu Parity Batch vs Stream ($\Delta \le 3.85 \times 10^{-6}$) | `data/prefix/_online_eval.json` (Bảng Online AUC vs Offline, Bảng $k$) |

---

## 4. Kế hoạch tiếp theo (Tuần 5 & 6)

1. **Tuần 5 (Phân tích lỗi & Drift Conversion)**:
   - Phân tích lỗi mô hình theo từng nhóm session (phiên ngắn 1 sự kiện, phiên cart-abandonment không checkout, phiên duyệt lâu không mua).
   - Phân tích drift conversion từ tháng 10/2019 ($3.40\%$) sang tháng 02/2020 ($3.02\%$) trên tập REES46.
   - Hỗ trợ B đo lường nghẽn Q2 khi chạy stress test (`tools/stress.sh`).
2. **Soạn thảo báo cáo & Slide bảo vệ**:
   - Viết chính các chương do A sở hữu:
     - **Chương 1**: Giới thiệu bài toán phân tích clickstream real-time và bài toán dự báo ý định mua hàng.
     - **Chương 2**: Dữ liệu và tiền xử lý (3 bộ dữ liệu, schema chuẩn hóa, xử lý sự kiện trùng lặp, báo cáo DQ).
     - **Chương 4**: Module Deep Learning (GRU vs Transformer, chống rò rỉ nhãn, hiệu chỉnh xác suất Platt, kiến trúc nhúng Q2).
     - **Chương 5b**: Thực nghiệm mô hình (Bảng so sánh Markov/GBM/GRU/Transformer, Bảng Ablation, Parity Batch-Stream, Đánh giá Online vs Offline).
   - Review chéo các chương 3, 5a và 6 của B.
   - Chuẩn bị slide trình bày panel mô hình và demo kịch bản các session có ý định mua cao.
