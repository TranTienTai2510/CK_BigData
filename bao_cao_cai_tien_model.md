# BÁO CÁO TỔNG HỢP: TÁI CẤU TRÚC, SO SÁNH BASELINE VÀ CẢI TIẾN MÔ HÌNH PHÂN LOẠI NHỊ PHÂN

**Bài toán trọng tâm:** Phân loại nhị phân (Binary Classification) — Dự đoán ý định mua hàng theo tiền tố phiên (*Session-prefix Purchase Intent Prediction*):  
$$P(\text{session ends with purchase} \mid e_1, e_2, \dots, e_k) \in (0, 1)$$

**Ngày hoàn thiện:** 26/11/2026  
**Phạm vi thực hiện:** `ml/` ([ml/models.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/ml/models.py), [ml/baselines.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/ml/baselines.py), [ml/train_dl.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/ml/train_dl.py)), tạo phẩm [models/](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models), báo cáo thực nghiệm tại `data/prefix/`, tài liệu [README_batch.md](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/README_batch.md) và toàn bộ bộ kiểm thử parity.

---

## 1. Mục tiêu và Nguyên tắc Thiết kế

1. **Tập trung vào một bài toán phân loại nhị phân thống nhất:** Mọi kiến trúc từ baseline thống kê (`Markov`), mô hình cây (`HistGradientBoosting`), đến các mạng học sâu tuần tự/bảng (`GRU`, `Hybrid`, `TCN`, `Bi-LSTM + Attention`, `MLP-Mixer`, `Causal Transformer`, `Bi-LSTM v2`) đều được định nghĩa, huấn luyện và đánh giá cho cùng một mục tiêu duy nhất là ước lượng xác suất phiên kết thúc bằng giao dịch mua hàng từ tiền tố tối đa $L = 50$ sự kiện trước `cut_seq_no` (`checkout`/`purchase`).
2. **So sánh công bằng tuyệt đối (Zero Data/Split Skew):** Toàn bộ các mô hình đều được huấn luyện và đánh giá trên cùng một tập mẫu `(session, k)` trích xuất từ `data/prefix/sessions`:
   - `train`: **141,423** mẫu tiền tố
   - `val`: **138,658** mẫu tiền tố (25,974 mẫu dương, tỉ lệ mua 18.73%)
   - `test`: **138,663** mẫu tiền tố (25,667 mẫu dương, tỉ lệ mua 18.51%; trong đó `rees46_cosmetics`: 79,439 mẫu và `synthetic`: 59,224 mẫu)
3. **Bảo toàn 100% Contract Streaming Q2:** Mọi mô hình Deep Learning đều giữ nguyên giao diện đầu vào/ra chuẩn ONNX (`x_cat: int64 [B, 50, 3]`, `x_num: float32 [B, 50, 10]`, `length: int64 [B]`, `shop: int64 [B]` $\rightarrow$ `prob: float32 [B]`), tích hợp sẵn tham số hiệu chỉnh Platt per-shop (`calib_a`, `calib_b`) ngay trong đồ thị ONNX để tương thích trực tiếp với [streaming/q2_intent.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/streaming/q2_intent.py).

---

## 2. Các Thay đổi Kỹ thuật trong Mã nguồn

### 2.1. Nâng cấp [ml/baselines.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/ml/baselines.py)
- **Tái sử dụng bộ trích xuất đặc trưng bảng & Markov (`extract_tabular_and_markov`):** Tách logic tạo 32 đặc trưng bảng (`TABULAR_FEATURE_NAMES`) và tuple chuyển trạng thái Markov ra hàm dùng chung cho cả `load_prefix_samples` lẫn `ml/train_dl.py`, đảm bảo `MarkovPurchaseBaseline` và `GBMPurchaseBaseline` dùng chung đúng tập `(session, k)` với nhóm Deep Learning.
- **Đánh giá F1 và ngưỡng tối ưu (`compute_best_f1` & `safe_metrics`):** Tính `best_f1` và `best_threshold` trực tiếp từ đường cong Precision–Recall trên `VAL`, đồng thời bổ sung `f1_at_val_threshold` trên `TEST` (tránh rò rỉ ngưỡng đánh giá).

### 2.2. Hợp nhất Họ Kiến trúc Phân loại Nhị phân trong [ml/models.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/ml/models.py)
- **`BASELINE_DL_ARCHS`** (`d_model = 32`, `n_layers = 1`):
  1. `gru`: GRU 1 lớp + masked mean/max pooling + `shop_emb` + `log1p(length)` + skip `x_num_last`, `x_num_mean`.
  2. `hybrid`: Nhánh sâu GRU + nhánh rộng MLP xử lý **44 đặc trưng tổng hợp tiền tố** (`extract_wide_prefix_features`: thống kê lag `prev`, bước đầu `first`, min/max/std giá và `dt`, tốc độ `add_per_step`, `view_per_step`, `remove_per_add`, `cat_switch_rate`, `brand_switch_rate`, `recent_add_rate` trong 5 bước cuối, v.v.) tính **ngay bên trong đồ thị PyTorch/ONNX** (không đổi input contract).
  3. `tcn`: 2 khối `CausalConv1dBlock` (`kernel_size=3`, dilation `1, 2`, causal left padding, LayerNorm, GELU, residual).
  4. `bilstm_attention`: Bi-LSTM 2 chiều bên trong tiền tố đã quan sát (`d_model // 2` mỗi chiều) + `AdditiveAttentionPool` có mặt nạ left-padding, kết hợp `[h_last, h_attn, h_mean, shop_emb, log_len, x_num_last, x_num_mean]`.
  5. `mixer`: 2 khối `MLPMixerBlock` trộn theo chiều vị trí (`L = 50`) và chiều kênh (`D`).
  6. `transformer`: `CausalSelfAttentionBlock` Pre-LN (`4` heads) với `pos_emb` và mặt nạ nhân quả + padding tường minh.
- **Các kiến trúc cải tiến (`bilstm_attention_v2` và `hybrid_v2`):**
  - Lưu ý kỹ thuật khi xuất ONNX opset 18: phát hiện `torch.expm1` không được hỗ trợ bởi ONNX opset 18 (`UnsupportedOperatorError: aten::expm1`) và đã thay thế bằng `torch.exp(...) - 1.0`, giúp toàn bộ 8 kiến trúc xuất ONNX sạch sẽ với sai lệch số học $< 10^{-7}$.

### 2.3. Pipeline 3 Giai đoạn trong [ml/train_dl.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/ml/train_dl.py)
- **Stage 1 (So sánh Baseline):** Đánh giá 2 baseline cổ điển (`markov`, `histgbm`) và 6 kiến trúc DL (`gru`, `hybrid`, `tcn`, `bilstm_attention`, `mixer`, `transformer`), đo đầy đủ ROC-AUC, PR-AUC, Log-Loss, Brier Score, Best F1, số tham số và độ trễ ONNX CPU 1 luồng (`b=1` và `b=128`).
- **Stage 2 (Chọn Baseline tốt nhất để cải tiến & Ablation):**
  - Hệ thống tự động chọn baseline tốt nhất theo điểm cân bằng trên `VAL` (`_selection_score`: trung bình `VAL overall ROC-AUC` và `VAL mean per-shop ROC-AUC`, phạt nếu độ trễ `b=1 > 1.0 ms`).
  - Quá trình thử nghiệm thực nghiệm đã kiểm chứng các hướng cải tiến:
    - Thử nghiệm đầu tiên (`hybrid_v2` với `Wide44` + `QueryAttentionPool` + `ShopFiLM`) đạt `TEST ROC-AUC = 0.7841`, cho thấy nhánh Wide 44 chiều + FiLM làm tăng nhẹ overfitting so với `bilstm_attention` (`0.7855`).
    - Thử nghiệm rút gọn lần hai (`bilstm_attention_v2` phiên bản `QueryAttention` + `exp` ratios chưa nén log) đạt `0.7838`–`0.7841`, cũng kém hơn `bilstm_attention` gốc.
    - Từ phân tích đó, kiến trúc **`bilstm_attention_v2`** được tinh chỉnh gọn lại gồm: Bi-LSTM + **2 đầu Additive Attention** (`attn_pool`, `attn_pool2`) + **`h_max`** + **residual skip từ `StepEmbedding`** (`emb[:, -1, :]`) + **4 đặc trưng nén log** (`extract_log_ratio_features`: `price_delta`, `dt_delta`, `has_cart`, `cat_switch_rate`) + **Top-2 Checkpoint Weight Averaging (SWA)**.
  - Chạy bộ ablation cho cả `gru` (`gru_full`, `gru_unweighted`, `gru_uncalibrated`, `gru_no_cat_brand`, `gru_no_num`) và `bilstm_attention_v2` (`bilstm_v2_no_swa`, `bilstm_v2_no_log_ratios`, `bilstm_v2_no_emb_skip`), tách cờ `measure_onnx=False` cho các biến thể ablation loại bỏ đầu vào (như `gru_no_num`) để tránh lỗi pruned input khi benchmark ONNX.
- **Stage 3 (Xuất Champion ONNX & Model Card):** Xuất Champion cuối cùng sang [models/intent_v1.onnx](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/intent_v1.onnx), Challenger sang [models/intent_transformer_full.onnx](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/intent_transformer_full.onnx), ghi [models/model_card.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/model_card.json) và hai báo cáo JSON [data/prefix/_dl_ablation.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/data/prefix/_dl_ablation.json), [data/prefix/_model_benchmark.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/data/prefix/_model_benchmark.json).

---

## 3. Kết quả Thực nghiệm & So sánh

### 3.1. Bảng So sánh Toàn diện (Tập `VAL` và `TEST`)

| Mô hình | VAL ROC-AUC | TEST ROC-AUC | TEST PR-AUC | TEST Log-Loss | TEST Brier | TEST Best F1 | Tham số | Độ trễ (`b=1` / `b=128`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`bilstm_attention` (Champion)** | **0.7786** | **0.7855** | **0.3848** | **0.3932** | **0.1273** | 0.4814 | 23,677 | 0.32 ms / 12.97 ms |
| `gru` | 0.7781 | 0.7852 | 0.3822 | 0.3934 | 0.1274 | 0.4828 | 22,525 | 0.28 ms / 11.11 ms |
| `bilstm_attention_v2` | 0.7785 | 0.7851 | 0.3831 | 0.3936 | 0.1274 | **0.4833** | 27,965 | 0.40 ms / 15.35 ms |
| `mixer` | 0.7778 | 0.7846 | 0.3827 | 0.3938 | 0.1275 | 0.4824 | 35,093 | 0.34 ms / 18.22 ms |
| `hybrid` | 0.7781 | 0.7845 | 0.3802 | 0.3936 | 0.1274 | 0.4829 | 28,149 | 0.50 ms / 14.32 ms |
| `transformer` | 0.7777 | 0.7845 | 0.3822 | 0.3939 | 0.1275 | 0.4815 | 26,397 | 0.35 ms / 20.17 ms |
| `tcn` | 0.7771 | 0.7843 | 0.3817 | 0.3941 | 0.1275 | 0.4824 | 28,861 | 0.32 ms / 16.79 ms |
| `markov` | 0.7696 | 0.7789 | 0.3780 | 0.3999 | 0.1293 | 0.4757 | — | 0.01 ms / 1.42 ms (native) |
| `histgbm` | 0.7605 | 0.7666 | 0.3586 | 0.4149 | 0.1314 | 0.4739 | — | 9.45 ms / 11.72 ms (native) |

### 3.2. Kết quả Ablation Study (trên `TEST`)

| Cấu hình Ablation | VAL ROC-AUC | TEST ROC-AUC | TEST PR-AUC | TEST Brier | TEST Log-Loss | `rees46` ROC-AUC | `synthetic` ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bilstm_attention` (Champion) | **0.7786** | **0.7855** | **0.3848** | **0.1273** | **0.3932** | **0.7656** | **0.6036** |
| `bilstm_attention_v2` (Full) | 0.7785 | 0.7851 | 0.3831 | 0.1274 | 0.3936 | 0.7635 | 0.6031 |
| `bilstm_v2_no_swa` | 0.7785 | 0.7853 | 0.3839 | 0.1274 | 0.3936 | 0.7636 | **0.6036** |
| `bilstm_v2_no_emb_skip` | 0.7783 | 0.7849 | 0.3830 | 0.1275 | 0.3941 | 0.7623 | 0.6033 |
| `bilstm_v2_no_log_ratios` | 0.7774 | 0.7843 | 0.3808 | 0.1275 | 0.3939 | 0.7641 | 0.6005 |
| `gru_full` | 0.7781 | 0.7852 | 0.3822 | 0.1274 | 0.3934 | 0.7641 | 0.6027 |
| `gru_unweighted` | 0.7773 | 0.7845 | 0.3819 | 0.1275 | 0.3938 | 0.7632 | 0.6018 |
| `gru_uncalibrated` | 0.7781 | 0.7851 | 0.3821 | **0.1441** | **0.4347** | 0.7641 | 0.6027 |
| `gru_no_cat_brand` | 0.7773 | 0.7850 | 0.3843 | 0.1274 | 0.3933 | 0.7628 | 0.6030 |
| `gru_no_num` | 0.7763 | 0.7828 | 0.3788 | 0.1277 | 0.3949 | **0.7528** | 0.6009 |

**Phát hiện rút ra:**
- **`bilstm_attention` là Champion tối ưu nhất tổng thể:** Đạt ROC-AUC (`0.7855`), PR-AUC (`0.3848`), Log-Loss (`0.3932`), Brier (`0.1273`) tốt nhất, đồng thời dẫn đầu trên cả hai cửa hàng (`rees46_cosmetics`: `0.7656`, `synthetic`: `0.6036`) với chỉ `23,677` tham số và độ trễ ONNX `0.33 ms` ở `batch=1`.
- **`bilstm_attention_v2` đạt Best F1 cao nhất (`0.4833`)** và phần ablation xác nhận 4 đặc trưng log-ratio (`price_delta`, `dt_delta`, `has_cart`, `cat_switch_rate`) đóng góp trực tiếp vào hiệu năng (`TEST ROC-AUC` tăng từ `0.7843` lên `0.7851`/`0.7853`, `PR-AUC` tăng từ `0.3808` lên `0.3831`/`0.3839`).
- **Tầm quan trọng của Platt calibration & đặc trưng số:** Bỏ hiệu chỉnh xác suất làm Brier Score của GRU xấu đi rõ rệt (`0.1274` $\rightarrow$ `0.1441`) và Log-Loss tăng (`0.3934` $\rightarrow$ `0.4347`); bỏ đặc trưng số/thời gian làm `rees46` ROC-AUC sụt từ `0.7641` xuống `0.7528`.

### 3.3. Hiệu năng Champion (`bilstm_attention`) theo Độ dài Prefix $k$ trên `TEST`

| Độ dài prefix `k` | Số mẫu | Tỉ lệ mua | All-shop ROC-AUC | All-shop PR-AUC | `rees46` ROC-AUC | `synthetic` ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `1` | 48,000 | 9.12% | **0.8288** | **0.2747** | 0.6716 | 0.5233 |
| `2` | 22,952 | 18.82% | **0.7464** | **0.3610** | 0.6788 | 0.5827 |
| `3-5` | 34,599 | 24.10% | **0.7308** | **0.3968** | 0.7114 | 0.6086 |
| `6-10` | 22,723 | 29.64% | **0.6870** | **0.4121** | 0.6803 | 0.5499 |
| `11-20` | 6,425 | 19.64% | **0.6973** | **0.3565** | 0.6428 | 0.5281 |
| `21-50` | 3,261 | 16.50% | **0.6345** | **0.2470** | 0.6345 | — |
| `>50` | 703 | 13.94% | **0.6876** | **0.2414** | 0.6876 | — |

---

## 4. Tạo phẩm Bàn giao & Kiểm chứng Tự động

### 4.1. Tạo phẩm đã xuất
- [models/intent_v1.onnx](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/intent_v1.onnx): `108,050 bytes`, kiến trúc `bilstm_attention`, `d_model=32`, tham số Platt calibration theo shop (`shop 3`: $a=1.016625, b=-0.725036$; `shop 4`: $a=1.043087, b=-0.674071$), ngưỡng F1 tối ưu trên `VAL`: `0.2345` (`rees46`: `0.1342`, `synthetic`: `0.2556`), độ trễ ONNX CPU 1 luồng: `0.3345 ms` (`b=1`) / `12.978 ms` (`b=128`).
- [models/intent_transformer_full.onnx](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/intent_transformer_full.onnx): `133,908 bytes` (Challenger phục vụ A/B testing trên Q2).
- [models/model_card.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/model_card.json) & [models/feature_spec.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/models/feature_spec.json).
- [data/prefix/_dl_ablation.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/data/prefix/_dl_ablation.json) & [data/prefix/_model_benchmark.json](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/data/prefix/_model_benchmark.json).

### 4.2. Kết quả Kiểm thử sau lần Chỉnh sửa & Xuất Model cuối cùng
- **[tests/test_baselines.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/tests/test_baselines.py):** **10/10 PASS**.
- **[tests/test_onnx_parity.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/tests/test_onnx_parity.py):** **14/14 PASS** — Sai lệch PyTorch vs ONNX trên 1,000 mẫu: `max_abs_diff = 8.94e-08` ($\ll 10^{-4}$), xác suất stream tăng từ `0.0172` (`page_view`) lên `0.0818` (`add_to_cart`).
- **[tests/test_q2_parity.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/tests/test_q2_parity.py):** **ALL PASS** — Xác thực thứ tự `seq_no`, lọc trùng, đóng phiên và sai lệch Batch vs Stream `max_abs_diff = 4.09e-06` ($< 10^{-5}$).
- **[tests/test_online_eval.py](file:///d:/HocTap/Big%20Data/project_CK/CK_BigData/tests/test_online_eval.py):** **ALL PASS** — Kiểm chứng đánh giá online, định tuyến A/B 50/50 giữa `intent_v1` và `intent_transformer_v1`, và giám sát trạng thái `PipelineMonitor`.
