# Phân công đồ án Real-time E-commerce Clickstream Analytics

Nov 4, 2026 · @Someone

## Nguyên tắc chia việc

Chia theo ranh giới hệ thống, không theo giai đoạn: **A (Data & ML)** sở hữu mọi thứ về dữ liệu và model, **B (Platform & Streaming)** sở hữu hạ tầng và luồng dữ liệu. Ranh giới giữa hai người là các contract đã viết sẵn (`common/messages.py`, schema canonical, DDL ClickHouse), nên cả hai làm song song ngay từ tuần 1 mà không chờ nhau.

### Vai trò

| Vai trò | Sở hữu (viết code, chịu trách nhiệm chạy đúng) | Chỉ review, không sửa trực tiếp |
|---|---|---|
| **A — Data & ML** | `batch/` + `data/unified`, `features.py`, train và đánh giá model, export ONNX, Spark Q2 (state session + gọi model), view đánh giá online, `analysis/live.py` | `docker-compose.yml`, simulator, Q0/Q1, dashboard |
| **B — Platform & Streaming** | `docker-compose.yml`, Kafka, simulator, image Spark + Q0/Q1 + listener metrics, vận hành ClickHouse, dashboard Grafana, stress test và test phục hồi | `batch/`, model, Q2 |

Người quen PyTorch nhận A, người quen Docker/Linux nhận B. Q2 giao cho A vì nó dùng chung `features.py` với lúc train: một người giữ cả hai đầu thì không lệch đặc trưng giữa train và serve. Cả hai vẫn phải giải thích được toàn hệ thống khi bảo vệ, nên mọi PR đều do người kia review.

## Những phần đã xong

Lớp batch và thiết kế lưu trữ đã xong và có test; phần còn lại là hạ tầng chạy thật, model và Spark.

| Phần | Trạng thái | Tiếp quản |
|---|---|---|
| Tiền xử lý + adapter 3 shop (`batch/`, `data/unified`) | Chạy trên dữ liệu thật, `check_unified` PASS | A |
| Contract message Kafka (`common/messages.py`) + script tạo topic | Xong, có test khớp với ClickHouse | B (A review) |
| DDL ClickHouse: bảng, ingest, 15 view, cấu hình RAM | Xong, 44/44 test trên ClickHouse nhúng; chưa chạy với Kafka thật | B |
| `analysis/live.py` (tail, monitor, kéo cửa sổ có nhãn) | Xong, đã test | A |
| Sơ đồ kiến trúc (`docs/`) | Xong | B |

**Việc đầu tiên của cả hai:** đưa toàn bộ code hiện có lên một repo Git chung và chạy lại các test trên máy mỗi người.

## Kế hoạch 6 tuần

A và B chạy song song suốt 5 tuần, tuần 6 cả hai dồn vào báo cáo và bảo vệ; mốc quan trọng nhất là cuối tuần 2, khi B giao image Spark (G3) và A giao model ONNX (G4).

> Bản gốc có một sơ đồ tiến độ: ô trái là việc của A, ô phải là việc của B, mũi tên ở cột giữa chỉ bên nhận của mỗi bàn giao (chi tiết ở phần Điểm tích hợp). Sơ đồ này là hình nên không có trong bản văn bản.

### Việc từng tuần

#### Tuần 1

- **Cả hai:** đưa code lên repo Git chung, chạy lại test trên máy mỗi người
- **B:** `docker-compose.yml` (Kafka KRaft, tạo topic, ClickHouse, Grafana) trong 10 GB; `.wslconfig`
- **B:** simulator replay 3 shop (time-warp, dời cả `estimated_delivery_ts`), đạt ~1.100 event/s → G1
- **A:** `features.py` + tập prefix (cắt tại `cut_seq_no`, tối đa 50 event, chia train/val/test theo thời gian)
- **A:** baseline Markov + GBM, bảng AUC và PR-AUC theo độ dài prefix

#### Tuần 2

- **A:** GRU + Transformer (loss có trọng số, hiệu chỉnh xác suất trên val), bảng ablation
- **A:** export ONNX + `model_card.json`, test lệch so với PyTorch < 1e-4 → G4
- **B:** image Spark 3.5.5 + script submit + volume checkpoint → G3
- **B:** Q0 kiểm tra contract → DLQ; Q1 KPI cửa sổ 10 s → `agg.kpi`; listener → `ops.metrics`
- **B:** dashboard KPI và vận hành (funnel, độ trễ, đối chiếu Spark với ClickHouse)

#### Tuần 3

- **A:** Q2 state session (cờ đã đóng và cờ đã checkout nằm trong state, sắp theo `seq_no`) + `mapInPandas` ONNX → G5
- **A:** test parity giữa batch và stream
- **B:** bật chèn trễ/trùng; thử watermark 5/10/30 s, lập bảng đánh đổi độ trễ và tỉ lệ mất
- **B:** dashboard SLA Olist; dashboard model bằng dữ liệu giả
- **Cả hai:** bắt đầu viết chương của mình

#### Tuần 4

- **B:** đưa Q2 vào compose, chạy 60 phút ổn định → G6; chốt máy chạy demo
- **A:** AUC online so với offline (lệch ≤ 0,02), tách theo độ dài prefix
- **A (tuỳ chọn):** thí nghiệm train lại champion/challenger
- **Cả hai:** quay video demo dự phòng

#### Tuần 5

- **B:** stress test `--fold 1/5/10/20` (throughput, p95, RAM); kill worker rồi phục hồi từ checkpoint, không mất và không trùng dữ liệu
- **A:** phân tích lỗi model, drift conversion 3,40% → 3,02%; hỗ trợ đo nghẽn Q2 khi stress test
- **Cả hai:** xong bản nháp báo cáo, review chéo

#### Tuần 6

- **Cả hai:** chốt báo cáo và slide, tập demo 2 lần, tập hỏi đáp chéo

## Điểm tích hợp

Hai người chỉ chờ nhau ở 6 điểm bàn giao; mỗi điểm có một cách kiểm chứng cụ thể để biết là đã giao xong.

| # | Bàn giao | Từ → cho | Hạn | Kiểm chứng |
|---|---|---|---|---|
| G1 | Stack chạy được: Kafka + ClickHouse + 7 topic | B → A | Cuối tuần 1 | `docker compose up`, liệt kê đủ 7 topic, `rt.v_funnel_1m` có số khi simulator chạy |
| G2 | `features.py` + tập prefix train/val/test | A → A (Q2) | Giữa tuần 2 | Test parity: cùng một session qua batch và qua hàm stream ra cùng tensor |
| G3 | Image Spark 3.5.5 (pandas, pyarrow, onnxruntime), script submit, volume checkpoint | B → A | Cuối tuần 2 | A submit được một job mẫu đọc `events.raw` |
| G4 | `intent_v1.onnx` + `model_card.json` (model_ver, input, AUC offline) | A → B | Cuối tuần 2 | Output ONNX lệch PyTorch < 1e-4 trên 1.000 mẫu |
| G5 | Q2 ghi `pred.intent` + `session.closed` | A → B | Cuối tuần 3 | `rt.v_online_eval_1m` có số; AUC online lệch offline ≤ 0,02 |
| G6 | Toàn bộ pipeline ổn định 60 phút | A + B | Cuối tuần 4 | `analysis/live.py monitor` không báo STALL/BACKPRESSURE suốt 1 lần replay |

Trước khi G5 xong, B làm dashboard model bằng dữ liệu dự đoán giả (cách test ClickHouse đã làm), nên không phải chờ A.

## Quy tắc làm việc chung

- **Định nghĩa "xong":** một việc chỉ tính là xong khi có code, test pass, lệnh chạy trong README và một số liệu hoặc ảnh đưa được vào báo cáo.
- **Git:** một repo chung, nhánh `main` luôn chạy được; mỗi việc một nhánh `feat/<việc>`; PR nhỏ, người kia review trong 24 giờ.
- **Contract:** sửa `common/messages.py`, `common/canonical.py` hoặc DDL ClickHouse cần cả hai đồng ý, chạy lại `tests/test_clickhouse_schema.py` và tăng `schema_version`.
- **Môi trường giống nhau:** Python 3.10, PySpark 3.5.5, JDK 17; chạy `python -m common.runtime_env` trước khi báo lỗi cho người kia.
- **Dữ liệu không commit:** `data/`, `models/*.onnx` để trong `.gitignore`, chia sẻ qua Drive; chỉ commit code, cấu hình và báo cáo DQ.
- **Họp:** 15 phút vào thứ Hai và thứ Năm (đã làm gì, đang kẹt gì); thứ Sáu là ngày tích hợp, cả hai chạy toàn bộ stack cùng nhau.
- **Số liệu thực nghiệm:** ghi ngay vào một file kết quả chung khi có (ngày, lệnh chạy, số đo), không dồn đến tuần cuối.

## Báo cáo, slide và bảo vệ

Mỗi người viết chương về phần mình sở hữu và review chương của người kia; viết dần từ tuần 3, không để hết sang tuần 6.

| Chương | Viết chính | Nội dung cốt lõi |
|---|---|---|
| 1. Giới thiệu, bài toán | A | Mục tiêu, phạm vi, đóng góp |
| 2. Dữ liệu và tiền xử lý | A | 3 bộ dữ liệu, lỗi đã phát hiện (thứ tự `event_id`, sessionization, qty), báo cáo DQ |
| 3. Kiến trúc và luồng streaming | B | Sơ đồ, Kafka, Spark Q0/Q1, ClickHouse, lý do chọn từng công nghệ |
| 4. Module Deep Learning | A | Bài toán, chống rò rỉ, model, Q2 nhúng vào Spark |
| 5a. Thực nghiệm hệ thống | B | Throughput, độ trễ p95, watermark, phục hồi sau sự cố |
| 5b. Thực nghiệm model | A | Baseline và ablation, AUC theo độ dài prefix, online so với offline |
| 6. Kết luận, hướng phát triển | B | Giới hạn, việc chưa làm |

- **Slide:** mỗi người làm slide cho các chương mình viết.
- **Demo:** B chạy stack và Grafana, A trình bày panel model và các session đang có xác suất mua cao.
- **Hỏi đáp:** mỗi người tập trả lời 5 câu về phần của người kia, ví dụ: vì sao dùng ClickHouse thay Cassandra, watermark loại bao nhiêu event, vì sao không train trên dữ liệu replay.

## Rủi ro và phương án dự phòng

Rủi ro lớn nhất là RAM laptop và tốc độ của Q2; cả hai đều có dấu hiệu đo được từ sớm.

| Rủi ro | Dấu hiệu | Phương án | Ai xử lý |
|---|---|---|---|
| Không đủ RAM (WSL2 mặc định chỉ cấp 8 GB) | Container bị kill, restart lặp lại | `.wslconfig` nâng lên 10 GB; tắt Kafka UI; giảm worker xuống 2,5 GB | B |
| Q2 (Python) chậm hơn tốc độ dữ liệu vào | Cảnh báo BACKPRESSURE | Tăng `maxRecordsPerBatch`; giảm tải replay; tách inference thành service riêng | A |
| Model không vượt baseline | AUC xấp xỉ GBM | Trình bày bảng ablation trung thực; phân tích theo độ dài prefix | A |
| Một người trễ mốc tích hợp | G3 hoặc G4 trễ quá thứ Sáu tuần 2 | Người kia dùng dữ liệu giả theo contract để làm tiếp; cả hai bàn lại phạm vi | Cả hai |
| Hai máy khác cấu hình | Chạy được trên máy này, lỗi trên máy kia | `runtime_env` chuẩn hoá phiên bản; chốt một máy chạy demo từ tuần 4 | B |
| Demo lỗi trước hội đồng | — | Quay video demo dự phòng ngay khi G6 đạt | Cả hai |

## Thông tin cần nhóm chốt

- [ ] Hạn nộp và ngày bảo vệ, để đổi "Tuần 1–6" thành ngày thật
- [ ] Ai nhận vai A, ai nhận vai B
- [ ] Cấu hình máy mỗi người (số nhân CPU, RAM, kết quả `docker info | findstr /i "Total Memory"`), để chọn máy chạy demo
- [ ] Có làm thí nghiệm train lại theo thời gian (champion/challenger) không: thêm khoảng 1 tuần cho A và 60 phút demo
- [ ] Barem yêu cầu gì về báo cáo (số trang, mẫu, có cần video không)
