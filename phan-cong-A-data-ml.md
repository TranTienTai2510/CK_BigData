# Phân công của A (Data & ML)

Đồ án Real-time E-commerce Clickstream Analytics

## Phạm vi sở hữu

- `batch/` + `data/unified` (tiền xử lý, adapter 3 shop; đã chạy trên dữ liệu thật, `check_unified` PASS)
- `features.py`, train và đánh giá model, export ONNX
- Spark Q2 (state session + gọi model)
- View đánh giá online (`rt.v_online_eval_1m`)
- `analysis/live.py` (đã xong, đã test)
- Chỉ review, không sửa trực tiếp: `docker-compose.yml`, simulator, Q0/Q1, dashboard

## Việc theo tuần

| Tuần | Việc của A |
|---|---|
| 1 | Chạy lại test trên repo Git chung (cùng B). Viết `features.py` + tập prefix (cắt tại `cut_seq_no`, tối đa 50 event, chia train/val/test theo thời gian). Làm baseline Markov + GBM, lập bảng AUC và PR-AUC theo độ dài prefix |
| 2 | Train GRU + Transformer (loss có trọng số, hiệu chỉnh xác suất trên val), làm bảng ablation. Export ONNX + `model_card.json`, test lệch so với PyTorch < 1e-4 → **G4** |
| 3 | Viết Q2: state session (cờ đã đóng và cờ đã checkout nằm trong state, sắp theo `seq_no`) + `mapInPandas` ONNX → **G5**. Test parity giữa batch và stream. Bắt đầu viết chương của mình |
| 4 | Đo AUC online so với offline (lệch ≤ 0,02), tách theo độ dài prefix. Tuỳ chọn: thí nghiệm train lại champion/challenger. Cùng B quay video demo dự phòng |
| 5 | Phân tích lỗi model, drift conversion 3,40% → 3,02%. Hỗ trợ đo nghẽn Q2 khi B stress test. Xong bản nháp báo cáo, review chéo |
| 6 | Chốt báo cáo và slide, tập demo 2 lần, tập hỏi đáp chéo |

## Điểm bàn giao liên quan đến A

| # | Bàn giao | Từ → cho | Hạn | Kiểm chứng |
|---|---|---|---|---|
| G2 | `features.py` + tập prefix train/val/test | A → A (Q2) | Giữa tuần 2 | Test parity: cùng một session qua batch và qua hàm stream ra cùng tensor |
| G3 | Image Spark 3.5.5, script submit, volume checkpoint | B → A | Cuối tuần 2 | A submit được một job mẫu đọc `events.raw` |
| G4 | `intent_v1.onnx` + `model_card.json` | A → B | Cuối tuần 2 | Output ONNX lệch PyTorch < 1e-4 trên 1.000 mẫu |
| G5 | Q2 ghi `pred.intent` + `session.closed` | A → B | Cuối tuần 3 | `rt.v_online_eval_1m` có số; AUC online lệch offline ≤ 0,02 |
| G6 | Toàn bộ pipeline ổn định 60 phút | A + B | Cuối tuần 4 | `analysis/live.py monitor` không báo STALL/BACKPRESSURE suốt 1 lần replay |

## Báo cáo và bảo vệ

- **Viết chính:** chương 1 (Giới thiệu, bài toán), 2 (Dữ liệu và tiền xử lý), 4 (Module Deep Learning), 5b (Thực nghiệm model)
- **Review:** các chương của B
- **Slide:** làm slide cho các chương mình viết
- **Demo:** trình bày panel model và các session đang có xác suất mua cao
- **Hỏi đáp:** tập trả lời 5 câu về phần của B

## Rủi ro A phải xử lý

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Q2 (Python) chậm hơn tốc độ dữ liệu vào | Cảnh báo BACKPRESSURE | Tăng `maxRecordsPerBatch`; giảm tải replay; tách inference thành service riêng |
| Model không vượt baseline | AUC xấp xỉ GBM | Trình bày bảng ablation trung thực; phân tích theo độ dài prefix |

## Quy tắc chung cần nhớ

- Một việc chỉ tính là xong khi có code, test pass, lệnh chạy trong README và một số liệu hoặc ảnh đưa được vào báo cáo.
- Sửa contract (`common/messages.py`, `common/canonical.py`, DDL ClickHouse) cần cả hai đồng ý, chạy lại `tests/test_clickhouse_schema.py` và tăng `schema_version`.
- Môi trường: Python 3.10, PySpark 3.5.5, JDK 17; chạy `python -m common.runtime_env` trước khi báo lỗi cho người kia.
- Git: nhánh `feat/<việc>`, PR nhỏ, người kia review trong 24 giờ.
