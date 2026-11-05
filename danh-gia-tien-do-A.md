# Báo Cáo Đánh Giá Tiến Độ Hoàn Thành Công Việc Của A (Data & ML)

**Đồ án:** Real-time E-commerce Clickstream Analytics  
**Cập nhật lần cuối:** 05/11/2026  
**Phạm vi phụ trách của A:** `batch/`, `data/unified`, `ml/` (`features.py`, `prefixes.py`, `baselines.py`, `models.py`, `train_dl.py`), xuất ONNX (`models/`), Spark Q2, view đánh giá online (`rt.v_online_eval_1m`), `analysis/live.py` và các bài kiểm thử liên quan.

---

## 1. Tổng quan tiến độ theo lộ trình 6 tuần

| Giai đoạn | Hạng mục công việc của A (theo phân công) | Trạng thái | Tỉ lệ hoàn thành | Ghi chú / Kết quả kiểm chứng |
|---|---|---:|---:|---|
| **Nền tảng (Có sẵn & Khôi phục)** | Batch ETL 3 shop (`synthetic`, `rees46_cosmetics`, `olist`) → `data/unified` + `analysis/live.py` | **HOÀN THÀNH** | **100%** | Khôi phục đầy đủ Parquet trên `data/unified`, vượt qua toàn bộ kiểm tra `--strict`. |
| **Tuần 1** | Viết `ml/features.py` + tạo tập prefix (`ml/prefixes.py`, cắt tại `cut_seq_no`, tối đa 50 sự kiện, chia `train/val/test` theo thời gian, đóng băng vocab từ `train`) + Baseline Markov & GBM (`ml/baselines.py`) | **HOÀN THÀNH** | **100%** | Có đầy đủ code, test (`test_features_parity.py`, `test_baselines.py` PASS), hướng dẫn trong `README_batch.md` và bảng AUC/PR-AUC theo độ dài prefix $k$. |
| **Mốc G2 (A → A)** | Kiểm thử parity giữa batch và stream (`ml/features.py` + `data/prefix`) | **HOÀN THÀNH** | **100%** | `tests/test_features_parity.py` đạt **26/26 kiểm tra PASS** (`tensor_mismatch = 0` trên cả dữ liệu giả lập và 550 phiên thực tế của 2 shop). |
| **Tuần 2** | Huấn luyện GRU + Transformer (loss có trọng số lớp, hiệu chỉnh xác suất Platt trên `val`), lập bảng Ablation (`ml/models.py`, `ml/train_dl.py`) | **HOÀN THÀNH** | **100%** | Champion `gru_full` đạt **TEST ROC-AUC = 0.7850**, vượt cả Markov (`0.7685`) và GBM (`0.7675`). Bảng Ablation 6 biến thể hoàn chỉnh. |
| **Mốc G4 (A → B)** | Xuất `models/intent_v1.onnx` + `models/model_card.json` + kiểm thử lệch PyTorch < $10^{-4}$ trên 1.000 mẫu | **HOÀN THÀNH** | **100%** | `tests/test_onnx_parity.py` đạt **14/14 kiểm tra PASS**. Sai khác lớn nhất `max_abs_diff = 1.04e-07` ($\ll 10^{-4}$) trên 1.000 mẫu. |
| **Tuần 3 & Mốc G5** | Viết Spark Q2 (state session + `mapInPandas` gọi `intent_v1.onnx` ghi `pred.intent` & `session.closed`) + test parity end-to-end | **CHỜ G1 & G3 TỪ B** | **0%** *(Sẵn sàng)* | Đã chuẩn bị sẵn `SessionState` trong `ml/features.py` và kiểm thử luồng `SessionState -> ONNX` trong `tests/test_onnx_parity.py`. Cần chờ B bàn giao **G1** (Docker/Redpanda/ClickHouse) và **G3** (Image Spark 3.5.5 + script submit). |
| **Tuần 4 & Mốc G6** | Đo AUC online (`rt.v_online_eval_1m`) so với offline (lệch $\le 0.02$), thí nghiệm Champion/Challenger, chạy ổn định 60 phút | **CHƯA BẮT ĐẦU** | **0%** | Đã xuất sẵn Challenger `models/intent_transformer_full.onnx` để dùng ngay ở Tuần 4. Phụ thuộc hạ tầng streaming của B. |
| **Tuần 5 & Tuần 6** | Phân tích lỗi theo nhóm session, phân tích drift conversion ($3.40\% \rightarrow 3.02\%$), viết Chương 1, 2, 4, 5b, làm slide & bảo vệ | **ĐÃ CÓ SỐ LIỆU CHƯƠNG 2, 4, 5b** | **30%** | Đã có trọn bộ số liệu thực nghiệm Batch ETL, Prefix Dataset, Baselines, GRU/Transformer, Ablation & ONNX Parity để đưa vào Chương 2, 4 và 5b. |

---

## 2. Chi tiết các hạng mục đã hoàn thành (Tuần 1, Tuần 2, Mốc G2 & G4)

### 2.1. Tầng Batch & Khôi phục dữ liệu chuẩn hóa (`data/unified`)
- **Các tệp mã nguồn:**
  - `batch/preprocess.py`
  - `batch/adapters/synthetic.py`
  - `batch/adapters/rees46.py`
  - `batch/adapters/olist.py`
  - `batch/check_unified.py`
- **Số liệu thực tế đã kiểm chứng (`--strict PASS`):**
  - **`synthetic`**: `120,000` phiên (`33,735` phiên mua = `28.11%`), `760,958` sự kiện chuẩn hóa.
  - **`rees46_cosmetics`**: `4,810,578` phiên (`153,735` phiên mua = `3.20%`), `18,452,932` sự kiện chuẩn hóa (`2,237,330` dòng gộp giỏ hàng/mua trùng, `32,127` dòng giá $\le 0$ đưa vào quarantine).
  - **`olist`**: `98,666` đơn hàng (`96,096` đơn trong cửa sổ chính `2017-01..2018-08`), `478,798` sự kiện vòng đời đơn hàng (`order_events`).

---

### 2.2. Pipeline trích xuất đặc trưng dùng chung & Tập dữ liệu Prefix (Tuần 1 + Mốc G2)
- **Các tệp mã nguồn & kiểm thử:**
  - `ml/features.py`: Định nghĩa kiến trúc 2 tầng dùng chung tuyệt đối giữa Batch và Streaming Q2:
    1. Tầng bước thô (`raw_Event` $\rightarrow$ `StepRaw`) & đối tượng giữ trạng thái phiên trực tuyến `SessionState`.
    2. Tầng chuyển cửa sổ tối đa $L = 50$ sự kiện thành tensor (`window_to_tensor` / `session_tensors`) với left-padding, `x_cat` (`int64[50, 3]`), `x_num` (`float32[50, 10]`), `length` (`int64`), `shop` (`int64`) và vector 31 đặc trưng phẳng (`window_to_flat_features`) cho GBM.
  - `ml/prefixes.py`: Xây dựng từ điển `data/prefix/feature_spec.json` (**chỉ từ tập `train`** của các sự kiện tiền-checkout `seq_no <= cut_seq_no` nhằm chống rò rỉ nhãn) và sinh tập dữ liệu prefix tại `data/prefix/sessions/`.
  - `tests/test_features_parity.py`: Kiểm thử độc lập chứng minh cùng một phiên đi qua Spark Batch (`ml/prefixes.py`) và đi qua luồng sự kiện tuần tự (`SessionState` trong `ml/features.py`) tạo ra cùng một tensor (**26/26 kiểm tra PASS**, `tensor_mismatch = 0`).
- **Thống kê từ điển & tập dữ liệu (`data/prefix/_report.json`):**
  - Từ điển đóng băng từ `train`: `5` loại sự kiện tiền-checkout, `481` danh mục (`category`), `266` thương hiệu (`brand`), `2` clickstream shop (`synthetic=3`, `rees46_cosmetics=4`).
  - Quy mô phiên theo tập chia thời gian (`train` / `val` / `test` = `70% / 15% / 15%`):
    - `rees46_cosmetics`: `3,365,672` train | `722,107` val | `722,799` test.
    - `synthetic`: `83,991` train | `18,001` val | `18,008` test.

---

### 2.3. Mô hình Baseline Tuần 1: Chuỗi Markov bậc 1 & HistGBM (`ml/baselines.py`)
- **Các tệp mã nguồn & kiểm thử:**
  - `ml/baselines.py`: Huấn luyện Baseline Markov (ma trận chuyển trạng thái hấp thụ + hiệu chỉnh độ dài prefix) và HistGBM (trên 31 đặc trưng bảng tổng hợp + Out-of-Fold Target Encoding cho `category`/`brand`).
  - `tests/test_baselines.py`: **10/10 kiểm tra PASS**.
  - Kết quả lưu tại: `data/prefix/_baselines.json`.

---

### 2.4. Mô hình Deep Learning Tuần 2 (GRU + Transformer), Bảng Ablation & Bàn giao G4
- **Các tệp mã nguồn, tạo phẩm & kiểm thử:**
  - `ml/models.py`: Định nghĩa `StepEmbedding`, `CausalSelfAttentionBlock`, `IntentModel` (hỗ trợ cả `gru` và `transformer`, tích hợp sẵn tham số hiệu chỉnh Platt `calib_a`, `calib_b` theo từng shop vào trong đồ thị mô hình).
  - `ml/train_dl.py`: Huấn luyện với `BCEWithLogitsLoss(pos_weight=2.1157)`, khớp Platt calibration trên `val`, chạy 6 cấu hình Ablation, xuất ONNX và `model_card.json`.
  - Tạo phẩm bàn giao **G4 (`models/`)**:
    - `models/intent_v1.onnx` (`101,732 bytes` — Champion `gru_full`).
    - `models/intent_transformer_full.onnx` (`133,908 bytes` — Challenger `transformer_full`).
    - `models/model_card.json` & `models/feature_spec.json`.
  - `tests/test_onnx_parity.py`: **14/14 kiểm tra PASS** (`max_abs_diff = 1.04e-07 < 1e-4` trên 1.000 mẫu, kiểm증 dynamic batch size và luồng `SessionState -> ONNX`).

#### Bảng tổng hợp kết quả thực nghiệm (Tuần 1 Baselines vs. Tuần 2 GRU/Transformer & Ablation)

| Mô hình / Cấu hình | `VAL` All-k ROC-AUC | `VAL` All-k PR-AUC | `VAL` Brier | `TEST` All-k ROC-AUC | `TEST` All-k PR-AUC | `TEST` Brier | `TEST rees46` ROC-AUC | `TEST rees46` PR-AUC | `TEST synthetic` ROC-AUC | `TEST synthetic` PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Markov bậc 1** *(Baseline Tuần 1)* | 0.7610 | 0.3602 | — | 0.7685 | 0.3627 | — | 0.7211 | 0.1460 | 0.5835 | 0.3887 |
| **HistGBM 31 đặc trưng** *(Baseline Tuần 1)* | 0.7573 | 0.3596 | — | 0.7675 | 0.3624 | — | 0.7245 | 0.1714 | 0.5951 | 0.3812 |
| **GRU (Full — Champion `intent_v1`)** | **0.7780** | **0.3808** | **0.1298** | **0.7850** | **0.3825** | **0.1274** | **0.7639** | **0.1825** | **0.6028** | **0.4057** |
| **Transformer (Full — Challenger)** | 0.7776 | 0.3801 | 0.1298 | 0.7840 | 0.3816 | 0.1276 | 0.7616 | 0.1783 | 0.6006 | 0.4048 |
| *Ablation 1*: GRU không trọng số lớp (`unweighted`) | 0.7769 | 0.3805 | 0.1299 | 0.7845 | 0.3833 | 0.1276 | 0.7633 | 0.1770 | 0.6020 | 0.4072 |
| *Ablation 2*: GRU không hiệu chỉnh xác suất (`uncalibrated`) | 0.7780 | 0.3807 | 0.1478 | 0.7853 | 0.3827 | 0.1461 | 0.7639 | 0.1825 | 0.6028 | 0.4057 |
| *Ablation 3*: GRU bỏ embedding `category` + `brand` | 0.7769 | 0.3794 | 0.1298 | 0.7844 | 0.3838 | 0.1275 | 0.7606 | 0.1768 | 0.6027 | 0.4077 |
| *Ablation 4*: GRU bỏ đặc trưng số & thời gian (`no_num`) | 0.7761 | 0.3775 | 0.1299 | 0.7829 | 0.3805 | 0.1277 | 0.7522 | 0.1660 | 0.6016 | 0.4041 |

#### Bảng kết quả Champion `GRU (Full)` theo độ dài prefix $k$ trên tập `TEST`

| Độ dài prefix $k$ | Số mẫu (`TEST`) | Tỉ lệ mua | All-shop ROC-AUC | All-shop PR-AUC | `rees46` ROC-AUC | `rees46` PR-AUC | `synthetic` ROC-AUC | `synthetic` PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | 48,000 | 9.12% | **0.8269** | **0.2718** | 0.6669 | 0.0774 | 0.5162 | 0.2925 |
| `2` | 22,952 | 18.82% | **0.7456** | **0.3580** | 0.6788 | 0.1282 | 0.5804 | 0.3809 |
| `3-5` | 34,599 | 24.10% | **0.7316** | **0.3951** | 0.7148 | 0.1626 | 0.6085 | 0.4083 |
| `6-10` | 22,723 | 29.64% | **0.6855** | **0.4106** | 0.6754 | 0.2075 | 0.5483 | 0.4187 |
| `11-20` | 6,425 | 19.64% | **0.6885** | **0.3451** | 0.6306 | 0.2335 | 0.4900 | 0.4366 |
| `21-50` | 3,261 | 16.50% | **0.6361** | **0.2610** | 0.6361 | 0.2610 | — | — |
| `>50` | 703 | 13.94% | **0.6978** | **0.2698** | 0.6978 | 0.2698 | — | — |

---

## 3. Đánh giá theo "Quy tắc hoàn thành" (Definition of Done)

Theo quy định tại mục **"Quy tắc chung cần nhớ"** trong `phan-cong-A-data-ml.md`:
> *"Một việc chỉ tính là xong khi có code, test pass, lệnh chạy trong README và một số liệu hoặc ảnh đưa được vào báo cáo."*

| Tiêu chí bắt buộc | Tuần 1 + Mốc G2 | Tuần 2 + Mốc G4 |
|---|---|---|
| **1. Có mã nguồn hoàn chỉnh** | `ml/features.py`, `ml/prefixes.py`, `ml/baselines.py` | `ml/models.py`, `ml/train_dl.py`, `models/intent_v1.onnx`, `models/model_card.json` |
| **2. Có kiểm thử tự động PASS** | `tests/test_features_parity.py` (26/26 PASS), `tests/test_baselines.py` (10/10 PASS) | `tests/test_onnx_parity.py` (14/14 PASS) |
| **3. Có lệnh chạy trong `README_batch.md`** | Đã cập nhật đầy đủ lệnh chạy và mô tả trong `README_batch.md` | Đã cập nhật đầy đủ lệnh chạy và mô tả trong `README_batch.md` |
| **4. Có số liệu/bảng đưa vào báo cáo** | `data/prefix/_report.json`, `data/prefix/_baselines.json` | `data/prefix/_dl_ablation.json`, `models/model_card.json` |

---

## 4. Các công việc tiếp theo & Điểm phụ thuộc vào B

1. **Điểm bàn giao từ A sang B hiện tại (`G4`)**:
   - A đã hoàn thành sớm toàn bộ tạo phẩm `G4` (`models/intent_v1.onnx`, `models/model_card.json`, `models/feature_spec.json`). B có thể sử dụng ngay cấu trúc của `model_card.json` để chuẩn bị volume mount cho container Spark.
2. **Điểm cần chờ B bàn giao (`G1` và `G3`) để A triển khai Tuần 3 (`Spark Q2` → `G5`)**:
   - **G1 (`B → A`)**: `docker-compose.yml` hoàn chỉnh, 7 topic trên Redpanda, các bảng/view ClickHouse (theo `common/contract.py`) và Simulator phát sự kiện 3 shop.
   - **G3 (`B → A`)**: Image Spark 3.5.5 tích hợp `onnxruntime`, script submit mẫu đọc từ `events.raw` và volume checkpoint.
3. **Phần việc A có thể làm độc lập ngay cả khi đang chờ B**:
   - Viết trước module xử lý trạng thái phiên của **Spark Q2** (`streaming/q2_intent.py`: hàm `applyInPandasWithState` / `mapInPandas` nạp `models/intent_v1.onnx` và gọi `SessionState` từ `ml/features.py`) kèm unit test cục bộ trên PySpark trước khi nối vào Redpanda/ClickHouse của B.
   - Soạn thảo trước nội dung **Chương 1, Chương 2, Chương 4 và Chương 5b** của báo cáo đồ án từ bộ số liệu đã thu thập.
