# BÁO CÁO CHI TIẾT PHẦN VIỆC CỦA A (DATA & ML)
## Đồ án: Real-time E-commerce Clickstream Analytics
**Người thực hiện:** Thành viên A (Data & ML)  
**Ngày hoàn thiện:** 16/11/2026  
**Mã nguồn lưu trữ:** [CK_BigData](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData)

---

# MỤC LỤC
- [CHƯƠNG 1. GIỚI THIỆU VÀ ĐỊNH NGHĨA BÀI TOÁN](#chương-1-giới-thiệu-và-định-nghĩa-bài-toán)
  - [1.1. Bối cảnh và Tầm quan trọng](#11-bối-cảnh-và-tầm-quan-trọng)
  - [1.2. Định nghĩa Bài toán Dự đoán Ý định Mua hàng (Purchase Intent)](#12-định-nghĩa-bài-toán-dự-đoán-ý-định-mua-hàng-purchase-intent)
  - [1.3. Phạm vi phụ trách và Đóng góp của Thành viên A](#13-phạm-vi-phụ-trách-và-đóng-góp-của-thành-viên-a)
- [CHƯƠNG 2. DỮ LIỆU, TIỀN XỬ LÝ VÀ CHUẨN HÓA BATCH](#chương-2-dữ-liệu-tiền-xử-lý-và-chuẩn-hóa-batch)
  - [2.1. Nguồn dữ liệu đa cửa hàng (3 Shops)](#21-nguồn-dữ-liệu-đa-cửa-hàng-3-shops)
  - [2.2. Các dị thường dữ liệu thô và Giải pháp tiền xử lý](#22-các-dị-thường-dữ-liệu-thô-và-giải-pháp-tiền-xử-lý)
  - [2.3. Lược đồ chuẩn hóa Hợp nhất (Canonical Schema v3)](#23-lược-đồ-chuẩn-hóa-hợp-nhất-canonical-schema-v3)
  - [2.4. Kết quả Thực thi Tầng Batch ETL và Kiểm định Chất lượng](#24-kết-quả-thực-thi-tầng-batch-etl-và-kiểm-định-chất-lượng)
- [CHƯƠNG 4. MODULE DEEP LEARNING VÀ XỬ LÝ TRẠNG THÁI PHIÊN (Q2)](#chương-4-module-deep-learning-và-xử-lý-trạng-thái-phiên-q2)
  - [4.1. Kiến trúc Trích xuất Đặc trưng 2 Tầng (`ml/features.py`)](#41-kiến-trúc-trích-xuất-đặc-trưng-2-tầng-mlfeaturespy)
  - [4.2. Nguyên tắc Chống rò rỉ Nhãn và Xây dựng Tập Prefix (`ml/prefixes.py`)](#42-nguyên-tắc-chống-rò-rỉ-nhãn-và-xây-dựng-tập-prefix-mlprefixespy)
  - [4.3. Kiến trúc Mô hình Chuỗi: GRU vs. Causal Transformer (`ml/models.py`)](#43-kiến-trúc-mô-hình-chuỗi-gru-vs-causal-transformer-mlmodelspy)
  - [4.4. Hàm Mất mát Trọng số Lớp và Hiệu chỉnh Xác suất Đa cửa hàng (Platt Scaling)](#44-hàm-mất-mát-trọng-số-lớp-và-hiệu-chỉnh-xác-suất-đa-cửa-hàng-platt-scaling)
  - [4.5. Đóng gói ONNX Runtime và Nhúng vào Spark Streaming Q2 (`streaming/q2_intent.py`)](#45-đóng-gói-onnx-runtime-và-nhúng-vào-spark-streaming-q2-streamingq2_intentpy)
- [CHƯƠNG 5B. THỰC NGHIỆM VÀ ĐÁNH GIÁ MÔ HÌNH](#chương-5b-thực-nghiệm-và-đánh-giá-mô-hình)
  - [5.1. Thiết lập Môi trường và Dữ liệu Thực nghiệm](#51-thiết-lập-môi-trường-và-dữ-liệu-thực-nghiệm)
  - [5.2. So sánh Mô hình Chuỗi với Các Baselines Tuần 1 (Markov & HistGBM)](#52-so-sánh-mô-hình-chuỗi-với-các-baselines-tuần-1-markov--histgbm)
  - [5.3. Bảng Nghiên cứu Ablation (Đánh giá Đóng góp của Từng Thành phần)](#53-bảng-nghiên-cứu-ablation-đánh-giá-đóng-góp-của-từng-thành-phần)
  - [5.4. Đánh giá Hiệu năng Chi tiết theo Độ dài Prefix $k$](#54-đánh-giá-hiệu-năng-chi-tiết-theo-độ-dài-prefix-k)
  - [5.5. Kiểm chứng Tính Nhất quán (Parity Checks): PyTorch ↔ ONNX và Batch ↔ Stream](#55-kiểm-chứng-tính-nhất-quán-parity-checks-pytorch--onnx-và-batch--stream)
  - [5.6. Đánh giá Trực tuyến (Online Evaluation) và Thí nghiệm A/B Testing](#56-đánh-giá-trực-tuyến-online-evaluation-và-thí-nghiệm-ab-testing)
  - [5.7. Phân tích Lỗi và Sự Dịch chuyển Phân phối (Concept Drift)](#57-phân-tích-lỗi-và-sự-dịch-chuyển-phân-phối-concept-drift)
- [BỘ CÂU HỎI ĐÁP CHUYÊN SÂU BẢO VỆ PHẦN B](#bộ-câu-hỏi-đáp-chuyên-sâu-bảo-vệ-phần-b)

---

# CHƯƠNG 1. GIỚI THIỆU VÀ ĐỊNH NGHĨA BÀI TOÁN

## 1.1. Bối cảnh và Tầm quan trọng
Trong thương mại điện tử hiện đại, phần lớn người dùng duyệt web mà không có ý định mua hàng ngay từ đầu. Tỉ lệ chuyển đổi trung bình của các sàn thương mại điện tử chỉ dao động từ 2% đến 4%. Việc phát hiện sớm ý định mua hàng (purchase intent) của người dùng ngay trong phiên duyệt web đang diễn ra (in-session clickstream) đóng vai trò quyết định để:
- Tối ưu hóa chuyển đổi: Kích hoạt phiếu giảm giá (voucher), đề xuất sản phẩm phù hợp đúng thời điểm trước khi người dùng thoát phiên.
- Tiết kiệm chi phí tiếp thị: Tránh tặng voucher lãng phí cho những khách hàng vốn dĩ chắc chắn sẽ mua hàng, hoặc những khách chỉ lướt xem ngẫu nhiên.
- Cá nhân hóa trải nghiệm khách hàng theo thời gian thực (Real-time Personalization).

## 1.2. Định nghĩa Bài toán Dự đoán Ý định Mua hàng (Purchase Intent)
- **Đầu vào:** Một phiên duyệt web $S = (e_1, e_2, \dots, e_k)$ gồm chuỗi $k$ sự kiện liên tiếp của người dùng trước khi thực hiện hành động thanh toán. Mỗi sự kiện $e_t$ bao gồm: loại hành động (`page_view`, `add_to_cart`, `remove_from_cart`), thời điểm diễn ra, định danh sản phẩm, danh mục, thương hiệu, giá niêm yết, số lượng.
- **Đầu ra:** Xác suất phiên duyệt web đó sẽ kết thúc bằng một giao dịch mua hàng thành công:
  $$P(\text{session ends with purchase} \mid e_1, e_2, \dots, e_k) \in [0.0, 1.0]$$
- **Ràng buộc thời gian thực và chống rò rỉ thông tin:**
  - Cửa sổ ngữ cảnh giới hạn tối đa $L = 50$ sự kiện gần nhất (cắt theo prefix $k$).
  - Chỉ được sử dụng các sự kiện diễn ra **trước thời điểm checkout/purchase đầu tiên** ($seq\_no < cut\_seq\_no$).
  - Mô hình phải suy luận cực nhanh ($< 10$ ms) để phục vụ luồng streaming.

## 1.3. Phạm vi phụ trách và Đóng góp của Thành viên A
Thành viên A chịu trách nhiệm toàn bộ vòng đời Dữ liệu và Trí tuệ nhân tạo (Data & ML Layer), bao gồm:
1. Thiết kế và vận hành tầng tiền xử lý Batch ETL, làm sạch và chuẩn hóa dữ liệu từ 3 sàn thương mại điện tử thực tế về lược đồ Canonical Schema v3.
2. Xây dựng module trích xuất đặc trưng dùng chung (`ml/features.py`) đảm bảo đồng nhất bit-level giữa lúc huấn luyện và khi phục vụ luồng trực tuyến (Training-Serving Skew = 0).
3. Huấn luyện các mô hình Baseline (Markov Chain bậc 1, HistGradientBoosting) và các mô hình Học sâu chuỗi (GRU, Causal Transformer), xử lý mất cân bằng lớp và hiệu chỉnh xác suất Platt scaling.
4. Đóng gói mô hình chuẩn ONNX (`intent_v1.onnx`), tạo thẻ mô hình `model_card.json`, kiểm chứng sai lệch PyTorch vs ONNX $< 10^{-4}$.
5. Phát triển Spark Streaming Query Q2 (`streaming/q2_intent.py`) duy trì trạng thái phiên và suy luận intent theo thời gian thực, ghi nhận 2 topic `pred.intent` và `session.closed`.
6. Xây dựng bộ công cụ đánh giá trực tuyến và giám sát pipeline (`analysis/live.py`).

---

# CHƯƠNG 2. DỮ LIỆU, TIỀN XỬ LÝ VÀ CHUẨN HÓA BATCH

## 2.1. Nguồn dữ liệu đa cửa hàng (3 Shops)
Dự án tích hợp 3 bộ dữ liệu thương mại điện tử lớn với các đặc thù khác nhau:
1. **`synthetic` (Kaggle e-commerce clickstream):** 120,000 phiên, 760,958 sự kiện. Tỉ lệ mua hàng cao (~28.11%), dữ liệu có cấu trúc phiên rõ ràng, tiền tệ USD.
2. **`rees46_cosmetics` (Kaggle REES46 Cosmetics Shop):** Bộ dữ liệu thực tế khổng lồ gồm 5 tháng (10/2019 - 02/2020), 4,810,578 phiên, 18,452,932 sự kiện. Tỉ lệ chuyển đổi mua hàng thấp đặc trưng của bán lẻ (~3.20%), có nhiều danh mục và thương hiệu thực tế.
3. **`olist` (Kaggle Brazilian E-commerce):** 98,666 đơn hàng với 478,798 sự kiện vòng đời đơn hàng (`order_events`: order_placed $\rightarrow$ payment_approved $\rightarrow$ handed_to_carrier $\rightarrow$ delivered $\rightarrow$ review_created). Đơn vị tiền tệ Real Brazil (BRL) được quy đổi sang USD theo tỷ giá hàng ngày của Cục Dự trữ Liên bang Mỹ (FRED).

## 2.2. Các dị thường dữ liệu thô và Giải pháp tiền xử lý
Trong quá trình xử lý dữ liệu thô, Thành viên A đã phát hiện và xử lý triệt để các lỗi dị thường:
- **Lỗi chia phiên (Sessionization Gap):** Trong REES46, người dùng giữ nguyên một `user_session` nhưng có những khoảng nghỉ giữa 2 sự kiện kéo dài nhiều ngày. Giải pháp: Sử dụng Spark Window Function chia lại phiên tại các điểm gián đoạn $> 30$ phút.
- **Sự kiện giỏ hàng và mua hàng trùng lặp:** Trong cùng 1 giây, người dùng click mua nhiều sản phẩm tạo ra nhiều dòng trùng `event_time`. Giải pháp: Nhóm theo `(user_session, user_id, event_time, product_id)` và cộng dồn số lượng `qty` tương ứng.
- **Lỗi giá bán âm hoặc bằng 0:** Phát hiện 32,127 dòng trong REES46 có giá $\le 0$. Giải pháp: Cách ly vào thư mục `data/unified/quarantine/` để đảm bảo chất lượng mô hình.
- **Bảo toàn tính nhất quán thời gian:** Đảm bảo bất biến $event\_ts \le ts\_raw$ trên clickstream và $event\_ts \ge ts\_raw$ trên vòng đời đơn hàng Olist.

## 2.3. Lược đồ chuẩn hóa Hợp nhất (Canonical Schema v3)
Tất cả sự kiện của 3 shop sau khi xử lý đều được quy đổi về lược đồ chung `common/canonical.py`:
- Khóa chính: `(shop_id, session_id)`, `(shop_id, event_id)` sử dụng xxhash64 chuyển thành số nguyên có dấu `Int64`.
- Phân vùng lưu trữ: `data/unified/events/shop_id=<shop>/ym=<yyyy-MM>/` lưu trữ dưới định dạng Apache Parquet nén Snappy.

## 2.4. Kết quả Thực thi Tầng Batch ETL và Kiểm định Chất lượng
Toàn bộ mã nguồn ETL ([`batch/preprocess.py`](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/batch/preprocess.py), [`batch/adapters/rees46.py`](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/batch/adapters/rees46.py), [`batch/adapters/olist.py`](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/batch/adapters/olist.py)) đã được kiểm tra nghiêm ngặt qua [`batch/check_unified.py --strict`](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/batch/check_unified.py):
- `synthetic`: **120,000 phiên, 760,958 sự kiện** (`PASS`).
- `rees46_cosmetics`: **4,810,578 phiên, 18,452,932 sự kiện** (`PASS`).
- `olist`: **98,666 đơn hàng, 478,798 sự kiện** (`PASS`).

---

# CHƯƠNG 4. MODULE DEEP LEARNING VÀ XỬ LÝ TRẠNG THÁI PHIÊN (Q2)

## 4.1. Kiến trúc Trích xuất Đặc trưng 2 Tầng (`ml/features.py`)
Để triệt tiêu hoàn toàn độ lệch giữa lúc huấn luyện mô hình (Batch) và lúc phục vụ trực tuyến (Streaming Q2), Thành viên A thiết kế kiến trúc trích xuất 2 tầng:
1. **Tầng 1 (Raw Step - Bước thô):** Trích xuất các giá trị nguyên bản theo thứ tự $seq\_no$:
   - Đặc trưng định danh (Phân loại): `type_idx`, `cat_idx`, `brand_idx`.
   - Đặc trưng liên tục (Số học/Thời gian): `price`, `qty`, `dt_prev_ms` (khoảng cách tới sự kiện trước), `t_first_ms` (thời gian từ đầu phiên), `same_prod` (cùng sản phẩm trước đó), bộ đếm tích lũy `n_view`, `n_add`, `n_remove`, `cart_units`, `pos`.
2. **Tầng 2 (Tensor Window):** Chuyển cửa sổ tối đa 50 sự kiện thành Tensor với chuẩn hóa hàm toán học đơn biến $log(1 + x)$ và $slog1p(x) = \text{sign}(x) \cdot \log(1 + |x|)$. Tensor được left-padded (đệm trái với 0) để sự kiện mới nhất luôn nằm ở vị trí chỉ mục cuối cùng ($49$).

## 4.2. Nguyên tắc Chống rò rỉ Nhãn và Xây dựng Tập Prefix (`ml/prefixes.py`)
- **Đóng băng từ điển:** Toàn bộ từ điển loại sự kiện, danh mục (481 categories) và thương hiệu (266 brands) chỉ được học trên tập `TRAIN`. Các giá trị mới xuất hiện ở tập `VAL` hoặc `TEST` đều được ánh xạ về chỉ mục `OOV_IDX = 2`.
- **Cắt phiên tại `cut_seq_no`:** Nhãn mua hàng được xác định dựa trên sự kiện `checkout` hoặc `purchase`. Do đó, tập prefix chỉ trích xuất các sự kiện diễn ra **trước** điểm checkout ($seq\_no \le cut\_seq\_no$).
- **Chia tập theo thời gian (Time-based split):** Chia theo tỷ lệ 70% Train, 15% Val, 15% Test theo mốc thời gian diễn ra phiên để phản ánh đúng thực tế vận hành.

## 4.3. Kiến trúc Mô hình Chuỗi: GRU vs. Causal Transformer (`ml/models.py`)
- **Lớp nhúng bước (StepEmbedding):** Kết hợp nhúng phân loại qua các bảng Embedding chiều $d_{cat} = 8$ cho loại sự kiện, danh mục, thương hiệu với lớp chiếu tuyến tính biến đổi 10 đặc trưng số $\rightarrow d_{num} = 8$. Tổng hợp qua LayerNorm tạo vector đặc trưng $d_{model} = 32$.
- **Nhánh GRU (Champion):** Sử dụng 1 lớp Gated Recurrent Unit với ẩn số $d_{model} = 32$, hỗ trợ đầu vào độ dài thay đổi qua mặt nạ left-padding.
- **Nhánh Transformer (Challenger):** Sử dụng 1 khối Pre-LN Causal Self-Attention với 4 attention heads, kết hợp mặt nạ tam giác nhân quả (Causal Mask) đảm bảo bước $t$ không thấy thông tin bước $t+1$.
- **Kết nối tắt (Skip-connections):** Nối trực tiếp vector đặc trưng số tại bước cuối cùng (`x_num_last`) và trung bình cửa sổ (`x_num_mean`) vào lớp phân loại cuối cùng, giúp gradient lan truyền mạnh mẽ và ổn định.

## 4.4. Hàm Mất mát Trọng số Lớp và Hiệu chỉnh Xác suất Đa cửa hàng (Platt Scaling)
- **Class-weighted Loss:** Do tỉ lệ mua hàng trung bình chỉ chiếm ~18.5% trên dữ liệu gộp (~3.2% trên REES46), mô hình sử dụng hàm mất mát BCE có trọng số:
  $$\text{Loss} = \text{BCEWithLogitsLoss}(pos\_weight = 2.1157)$$
- **Platt Scaling trên tập Validation:** Trọng số $pos\_weight > 1$ đẩy giá trị logit thô lên cao, làm sai lệch xác suất thực tế. Thành viên A đã khớp một mô hình hồi quy logistic đơn biến theo từng shop trên tập `val`:
  $$\text{logit}_{\text{calibrated}} = a_{\text{shop}} \cdot \text{logit}_{\text{raw}} + b_{\text{shop}}$$
  Các tham số hiệu chỉnh ($a_3 = 0.9948, b_3 = -0.7869$ cho Synthetic; $a_4 = 1.0333, b_4 = -0.7027$ cho REES46) được nhúng trực tiếp dưới dạng tensor buffer bên trong đồ thị ONNX, giúp mô hình tự động xuất xác suất chuẩn xác mà không cần hậu xử lý bên ngoài.

## 4.5. Đóng gói ONNX Runtime và Nhúng vào Spark Streaming Q2 (`streaming/q2_intent.py`)
- **Xuất ONNX opset 17:** Mô hình Champion được xuất ra `models/intent_v1.onnx` (`101,732 bytes`), hỗ trợ kích thước batch linh hoạt (dynamic batching).
- **Spark Q2 Processing:**
  - Nhận luồng sự kiện sạch từ topic `events.clean`.
  - Quản lý trạng thái phiên qua `Q2SessionTracker` bao bọc `SessionState`: tự động đệm các sự kiện đến trễ và loại bỏ sự kiện trùng lặp.
  - Sử dụng cơ chế suy luận theo lô (batched inference) qua ONNX Runtime Singleton worker.
  - Phát đồng thời 2 luồng: `pred.intent` (dự đoán từng bước) và `session.closed` (khi phiên kết thúc hoặc timeout).

---

# CHƯƠNG 5B. THỰC NGHIỆM VÀ ĐÁNH GIÁ MÔ HÌNH

## 5.1. Thiết lập Môi trường và Dữ liệu Thực nghiệm
- Môi trường: PyTorch CPU, ONNX Runtime 1.20+, Python 3.10/3.14.
- Dữ liệu đánh giá: Tập `TEST` độc lập gồm **138,663 phiên** (79,439 phiên REES46 và 59,224 phiên Synthetic) hoàn toàn không rò rỉ trong quá trình huấn luyện.

## 5.2. So sánh Mô hình Chuỗi với Các Baselines Tuần 1 (Markov & HistGBM)

| Mô hình | All-k ROC-AUC | All-k PR-AUC | Brier Score | REES46 ROC-AUC | Synthetic ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Baseline 1: Chuỗi Markov bậc 1** | 0.7685 | 0.3627 | — | 0.7211 | 0.5835 |
| **Baseline 2: HistGBM (31 đặc trưng bảng)** | 0.7675 | 0.3624 | — | 0.7245 | 0.5951 |
| **GRU (Full: Weighted + Calibrated) — Champion** | **0.7850** | **0.3825** | **0.1274** | **0.7639** | **0.6028** |
| **Transformer (Full: Weighted + Calibrated) — Challenger** | 0.7840 | 0.3816 | 0.1276 | 0.7616 | 0.6006 |

**Nhận xét:**
- GRU vượt trội so với mô hình GBM truyền thống: ROC-AUC toàn hệ thống tăng từ `0.7675` lên **`0.7850`** (+1.75 điểm %), đặc biệt trên sàn thực tế REES46 ROC-AUC tăng mạnh từ `0.7245` lên **`0.7639`** (+3.94 điểm %).
- GRU và Transformer cho hiệu năng tương đương, nhưng GRU có số lượng tham số ít hơn (22,525 vs. 26,397) và tốc độ suy luận nhanh gấp 3 lần, nên được chọn làm **Champion**.

## 5.3. Bảng Nghiên cứu Ablation (Đánh giá Đóng góp của Từng Thành phần)

| Cấu hình Ablation | All-k ROC-AUC | All-k PR-AUC | Brier Score | REES46 ROC-AUC |
|---|---:|---:|---:|---:|
| **GRU Full (Đầy đủ tính năng)** | **0.7850** | **0.3825** | **0.1274** | **0.7639** |
| *Ablation 1:* Bỏ loss có trọng số (`unweighted`) | 0.7845 | 0.3833 | 0.1276 | 0.7633 |
| *Ablation 2:* Bỏ hiệu chỉnh xác suất (`uncalibrated`) | 0.7853 | 0.3827 | **0.1461** | 0.7639 |
| *Ablation 3:* Bỏ Embedding danh mục & thương hiệu | 0.7844 | 0.3838 | 0.1275 | 0.7606 |
| *Ablation 4:* Bỏ đặc trưng số và thời gian (`no_num`) | 0.7829 | 0.3805 | 0.1277 | 0.7522 |

**Phân tích đóng góp:**
1. **Hiệu chỉnh xác suất (Platt Calibration):** Đóng vai trò then chốt trong việc đưa ra xác suất thực tế. Khi bỏ hiệu chỉnh, điểm Brier Score xấu đi nghiêm trọng từ `0.1274` lên `0.1461`.
2. **Đặc trưng số và thời gian:** Nhịp độ tương tác giữa các sự kiện (`dt_prev_s`), thời gian duyệt từ đầu phiên (`t_first_s`) và bộ đếm giỏ hàng là nguồn thông tin giá trị nhất; việc loại bỏ chúng khiến ROC-AUC của REES46 sụt giảm nghiêm trọng từ `0.7639` xuống `0.7522`.

## 5.4. Đánh giá Hiệu năng Chi tiết theo Độ dài Prefix $k$

| Độ dài prefix $k$ | Số mẫu (`TEST`) | Tỉ lệ mua thực tế | All-shop ROC-AUC | All-shop PR-AUC |
|---|---:|---:|---:|---:|
| `k = 1` (Sự kiện đầu tiên) | 48,000 | 9.12% | **0.8269** | 0.2718 |
| `k = 2` | 22,952 | 18.82% | **0.7456** | 0.3580 |
| `k = 3-5` | 34,599 | 24.10% | **0.7316** | 0.3951 |
| `k = 6-10` | 22,723 | 29.64% | **0.6855** | 0.4106 |
| `k = 11-20` | 6,425 | 19.64% | **0.6885** | 0.3451 |
| `k = 21-50` | 3,261 | 16.50% | **0.6361** | 0.2610 |
| `k > 50` | 703 | 13.94% | **0.6978** | 0.2698 |

## 5.5. Kiểm chứng Tính Nhất quán (Parity Checks): PyTorch ↔ ONNX và Batch ↔ Stream
- **PyTorch vs. ONNX Runtime Parity (`tests/test_onnx_parity.py`):**
  - Trên 1,000 mẫu ngẫu nhiên: Sai khác lớn nhất **$\max |y_{\text{onnx}} - y_{\text{torch}}| = 1.04 \times 10^{-7} \ll 10^{-4}$**.
  - Đạt tiêu chuẩn chất lượng công nghiệp bàn giao **G4**.
- **Batch vs. Streaming Q2 Parity (`tests/test_q2_parity.py`):**
  - Chạy so sánh cùng một chuỗi phiên qua pipeline trích xuất batch và qua module streaming trực tuyến Q2:
  - Sai khác tuyệt đối lớn nhất: **$\Delta = 3.85 \times 10^{-6} < 10^{-5}$** (hoàn toàn là sai số làm tròn dấu phẩy động 32-bit). Đạt mốc bàn giao **G5**.

## 5.6. Đánh giá Trực tuyến (Online Evaluation) và Thí nghiệm A/B Testing
- **Kết quả đánh giá trực tuyến (`analysis/live.py eval`):**
  - Online ROC-AUC: **`0.7897`** so với Offline Benchmark **`0.7850`**.
  - Độ lệch tuyệt đối: **$\Delta = 0.0047 \le 0.02$**, đáp ứng xuất sắc tiêu chí dung sai của mốc **G6**.
- **Thí nghiệm A/B Testing trực tuyến:** Module Q2 hỗ trợ định tuyến 50/50 giữa `intent_v1` và `intent_transformer_v1`, ghi nhận đồng thời cả 2 mô hình vào ClickHouse phục vụ bảng điều khiển so sánh trên Grafana.

## 5.7. Phân tích Lỗi và Sự Dịch chuyển Phân phối (Concept Drift)
- **Phân tích nhóm lỗi chính:**
  1. *Lỗi False Positive ở các phiên duyệt dài:* Những người dùng xem hơn 20 sản phẩm nhưng cuối cùng không mua, mô hình có xu hướng dự đoán xác suất cao do số lượng tương tác lớn.
  2. *Lỗi False Negative ở các phiên mua chớp nhoáng:* Người dùng vào thẳng trang sản phẩm quen thuộc và mua ngay chỉ sau 1 lượt xem, mô hình ban đầu dự đoán xác suất thấp ($~0.04$) nhưng cập nhật nhảy vọt lên ngay khi có `add_to_cart` ($~0.36$).
- **Phân tích Concept Drift:** Tỉ lệ mua hàng trên tập REES46 biến thiên từ $3.40\%$ (tháng 10/2019) xuống $3.02\%$ (tháng 02/2020) do tính mùa vụ sau kỳ nghỉ lễ. Nhờ cơ chế Platt calibration độc lập, mô hình vẫn duy trì ROC-AUC ổn định mà không bị suy giảm hiệu năng.

---

# BỘ CÂU HỎI ĐÁP CHUYÊN SÂU BẢO VỆ PHẦN B
*(Dành cho Thành viên A chuẩn bị để trả lời các câu hỏi về phần Hạ tầng & Streaming của B)*

### Câu 1: Tại sao hệ thống lại chọn ClickHouse thay vì Cassandra, HBase hay Elasticsearch để lưu trữ và phân tích dữ liệu Clickstream?
- **Trả lời:**
  1. ClickHouse là cơ sở dữ liệu cột (Column-oriented DBMS) tối ưu hóa vượt bậc cho các truy vấn phân tích tổng hợp (OLAP) trên dữ liệu chuỗi thời gian khổng lồ (hàng trăm triệu dòng). Tốc độ quét và nén dữ liệu theo cột của ClickHouse cao gấp 5-10 lần so với Cassandra hay HBase.
  2. ClickHouse hỗ trợ sẵn công cụ Kafka Engine và Materialized Views, cho phép tự động tiêu thụ trực tiếp (ingest) dữ liệu từ các Kafka/Redpanda topic vào bảng lưu trữ mà không cần viết thêm các dịch vụ trung gian phức tạp.
  3. Hỗ trợ các hàm phân tích xác suất và định lượng mạnh mẽ (`quantiles`, `avgIf`, `toStartOfMinute`) giúp Grafana truy vấn dashboard thời gian thực với độ trễ dưới 50ms.

### Câu 2: Cơ chế Watermark trong Spark Streaming Q1 được cấu hình như thế nào và có ý nghĩa gì đối với việc xử lý dữ liệu trễ?
- **Trả lời:**
  - Trong Q1 ([`streaming/q1_kpi.py`](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/streaming/q1_kpi.py)), Watermark được thiết lập với ngưỡng `event_ts` trễ tối đa 10 phút (`watermark="10 minutes"`).
  - Ý nghĩa: Trong môi trường mạng thực tế, dữ liệu có thể bị trễ do nghẽn mạng từ thiết bị di động (Simulator mô phỏng 3% sự kiện trễ). Watermark định nghĩa giới hạn thời gian mà Spark chấp nhận chờ sự kiện muộn trước khi chốt kết quả cửa sổ 10 giây và giải phóng trạng thái (state) khỏi bộ nhớ RAM, ngăn chặn rò rỉ bộ nhớ (Out-Of-Memory).

### Câu 3: Làm thế nào hệ thống đảm bảo tính toàn vẹn dữ liệu khi Worker của Spark bị sập đột ngột (Stress test & Recovery)?
- **Trả lời:**
  - Hệ thống áp dụng cơ chế Checkpoint kết hợp Write-Ahead Logs (WAL) lưu trữ trên volume bền vững (`spark-checkpoints`).
  - Khi một Worker bị dừng hoặc sập (`tools/recovery_test.sh`), Spark Master sẽ tái lập tác vụ trên Worker mới, đọc lại offset đã xác nhận từ checkpoint và tiếp tục tiêu thụ từ Redpanda đúng vị trí dừng mà không làm mất mát hoặc nhân bản dữ liệu (Exactly-once / At-least-once with idempotent ClickHouse table engine).

### Câu 4: Vì sao dữ liệu huấn luyện mô hình (ML) không được lấy trực tiếp từ luồng phát Simulator Replay?
- **Trả lời:**
  - Simulator thực hiện phát lại (replay) dữ liệu với hệ số tua nhanh thời gian (Time-warp speed factor từ 50x đến 500x) và cố tình chèn thêm 3% sự kiện trễ và 2% sự kiện trùng lặp nhằm mục đích kiểm thử khả năng chịu tải của hạ tầng streaming.
  - Nếu huấn luyện trên dữ liệu replay, mô hình sẽ bị sai lệch hoàn toàn về các đặc trưng khoảng cách thời gian (`dt_prev_s`, `t_first_s`) và bị rò rỉ nhãn do sự kiện lộn xộn, dẫn đến suy giảm nghiêm trọng độ chính xác trong thực tế.

### Câu 5: Vai trò của topic `events.dlq` (Dead Letter Queue) trong Spark Q0 là gì và các tiêu chuẩn kiểm tra hợp đồng?
- **Trả lời:**
  - Spark Q0 ([`streaming/q0_validate.py`](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/streaming/q0_validate.py)) đóng vai trò cổng kiểm soát chất lượng (Quality Gate) đầu vào.
  - Mọi sự kiện từ `events.raw` vi phạm các quy tắc hợp đồng trong `common/contract.py` (như thiếu khóa chính, loại sự kiện không hợp lệ, giá âm, sự kiện giỏ hàng nhưng thiếu thông tin sản phẩm) sẽ bị tách khỏi luồng chính và đẩy vào topic `events.dlq`.
  - Giúp bảo vệ các truy vấn downstream (Q1, Q2) không bị sập do dữ liệu rác, đồng thời ghi nhận vào bảng ClickHouse `rt.dlq` để đội ngũ kỹ thuật giám sát lỗi dữ liệu theo thời gian thực.
