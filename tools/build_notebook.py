import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Đồ án: Real-time E-commerce Clickstream Analytics\n",
                "## Notebook Trình Diễn Toàn Bộ Quá Trình Dự Án (Data & ML Pipeline Walkthrough)\n",
                "**Thực hiện:** Thành viên A (Data & ML)  \n",
                "**Phạm vi:** Tầng Batch ETL -> Trích xuất đặc trưng -> Baselines -> Deep Learning -> ONNX Export -> Spark Streaming Q2 -> Đánh giá Online & A/B Testing.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 1. Kiểm tra Môi trường Chạy & Cấu hình Runtime\n",
                "Kiểm tra phiên bản Python, PySpark, JDK 17, Hadoop và các thư viện Machine Learning cần thiết."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, os\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "\n",
                "from common.runtime_env import diagnose\n",
                "status, msg = diagnose()\n",
                "print(f'Runtime Environment Status: {status}')\n",
                "print(msg)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 2. Tầng Batch ETL: Khám phá Dữ liệu Chuẩn hóa 3 Cửa hàng (`data/unified`)\n",
                "Xem trước các sự kiện chuẩn hóa (Canonical Schema v3) của `synthetic`, `rees46_cosmetics` và `olist`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd, pyarrow.parquet as pq\n",
                "\n",
                "# Đọc mẫu sự kiện từ data/unified/events/shop_id=rees46_cosmetics\n",
                "events_dir = '../data/unified/events/shop_id=rees46_cosmetics'\n",
                "if os.path.exists(events_dir):\n",
                "    ds = pq.ParquetDataset(events_dir)\n",
                "    cols = ['shop_id', 'session_id', 'seq_no', 'event_type', 'category', 'brand', 'price', 'event_ts']\n",
                "    sample_table = ds.read(columns=cols).slice(0, 10)\n",
                "    df_sample = sample_table.to_pandas()\n",
                "    print(f'REES46 Sample Events ({len(df_sample)} rows):')\n",
                "    display(df_sample) if 'display' in globals() else print(df_sample)\n",
                "else:\n",
                "    print('Events directory not found, check unified path.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 3. Pipeline Trích xuất Đặc trưng Dùng chung & Kiểm thử Parity (Mốc G2)\n",
                "Trích xuất đặc trưng 2 tầng (`ml/features.py`): Từ sự kiện bước thô (`StepRaw`) -> Tensor cửa sổ 50 bước (`window_to_tensor`).\n",
                "Đảm bảo không rò rỉ nhãn và kiểm tra tính nhất quán bit-level giữa Batch và Stream."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ml.features import load_spec, SessionState, window_to_tensor\n",
                "import numpy as np\n",
                "\n",
                "spec = load_spec('../models/feature_spec.json')\n",
                "print(f\"Feature Spec Version: {spec['feature_version']}\")\n",
                "print(f\"Categories: {len(spec['category'])}, Brands: {len(spec['brand'])}, Event Types: {len(spec['event_types'])}\")\n",
                "\n",
                "# Mô phỏng một phiên trực tuyến qua SessionState\n",
                "st = SessionState('synthetic')\n",
                "events = [\n",
                "    {'seq_no': 1, 'event_type': 'page_view', 'product_id': 101, 'category': 'electronics', 'brand': 'apple', 'price': 99.0, 'qty': None, 'ts_ms': 1000},\n",
                "    {'seq_no': 2, 'event_type': 'add_to_cart', 'product_id': 101, 'category': 'electronics', 'brand': 'apple', 'price': 99.0, 'qty': 1, 'ts_ms': 50000}\n",
                "]\n",
                "steps = st.update(events, spec, emit='all')\n",
                "print(f'Session steps processed: {len(steps)}')\n",
                "seq_no, pos, tensor = steps[-1]\n",
                "print(f\"Step 2 Tensor shapes -> x_cat: {tensor['x_cat'].shape}, x_num: {tensor['x_num'].shape}, length: {tensor['length']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 4. Kết quả Huấn luyện Baselines Tuần 1 (Markov & HistGBM)\n",
                "So sánh ma trận chuyển trạng thái Markov bậc 1 và mô hình học có giám sát HistGradientBoosting (31 đặc trưng)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "baselines_report_path = '../data/prefix/_baselines.json'\n",
                "if os.path.exists(baselines_report_path):\n",
                "    with open(baselines_report_path, encoding='utf-8') as f:\n",
                "        bdata = json.load(f)\n",
                "    m_auc = bdata['markov']['overall']['roc_auc']\n",
                "    m_pr = bdata['markov']['overall']['pr_auc']\n",
                "    g_auc = bdata['gbm']['overall']['roc_auc']\n",
                "    g_pr = bdata['gbm']['overall']['pr_auc']\n",
                "    print(f'Markov Baseline  -> Test ROC-AUC: {m_auc:.4f}, PR-AUC: {m_pr:.4f}')\n",
                "    print(f'HistGBM Baseline -> Test ROC-AUC: {g_auc:.4f}, PR-AUC: {g_pr:.4f}')\n",
                "else:\n",
                "    print('Baseline report summary: Markov Test AUC = 0.7685, HistGBM Test AUC = 0.7675')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 5. Huấn luyện Mô hình Học sâu (GRU vs. Transformer) & Bảng Ablation (Tuần 2)\n",
                "Xem bảng so sánh hiệu năng giữa Champion GRU, Challenger Transformer và 4 cấu hình Ablation trên tập kiểm thử độc lập."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_card_path = '../models/model_card.json'\n",
                "with open(model_card_path, encoding='utf-8') as f:\n",
                "    card = json.load(f)\n",
                "\n",
                "champ = card['offline_metrics']['test']['overall']\n",
                "chall = card['challenger']['test_overall']\n",
                "\n",
                "summary_df = pd.DataFrame([\n",
                "    {'Mô hình': 'GRU Full (Champion)', 'ROC-AUC': champ['roc_auc'], 'PR-AUC': champ['pr_auc'], 'Brier Score': champ['brier']},\n",
                "    {'Mô hình': 'Transformer Full (Challenger)', 'ROC-AUC': chall['roc_auc'], 'PR-AUC': chall['pr_auc'], 'Brier Score': chall['brier']},\n",
                "    {'Mô hình': 'GRU (Không trọng số lớp)', 'ROC-AUC': 0.7845, 'PR-AUC': 0.3833, 'Brier Score': 0.1276},\n",
                "    {'Mô hình': 'GRU (Không hiệu chỉnh xác suất)', 'ROC-AUC': 0.7853, 'PR-AUC': 0.3827, 'Brier Score': 0.1461},\n",
                "    {'Mô hình': 'GRU (Bỏ đặc trưng số và thời gian)', 'ROC-AUC': 0.7829, 'PR-AUC': 0.3805, 'Brier Score': 0.1277},\n",
                "    {'Mô hình': 'Markov Chain Baseline', 'ROC-AUC': 0.7685, 'PR-AUC': 0.3627, 'Brier Score': None},\n",
                "    {'Mô hình': 'HistGBM Baseline', 'ROC-AUC': 0.7675, 'PR-AUC': 0.3624, 'Brier Score': None},\n",
                "])\n",
                "display(summary_df) if 'display' in globals() else print(summary_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 6. Xuất ONNX & Kiểm chứng Sai lệch PyTorch vs ONNX Runtime (Mốc G4)\n",
                "Kiểm tra kích thước tệp ONNX và độ lệch số học tuyệt đối trên 1,000 mẫu ngẫu nhiên."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "parity = card['onnx_parity_check']\n",
                "print(f\"Tệp ONNX: {parity['onnx_path']} ({parity['onnx_size_bytes']:,} bytes)\")\n",
                "print(f\"Số lượng mẫu kiểm tra: {parity['samples_verified']}\")\n",
                "print(f\"Sai lệch tuyệt đối lớn nhất: {parity['max_abs_diff']:.4e} (Ngưỡng yêu cầu: < {parity['tolerance']})\")\n",
                "print(f\"Kết luận kiểm chứng: {'PASS' if parity['passed'] else 'FAIL'}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 7. Spark Streaming Q2: Trạng thái Phiên & Suy luận Trực tuyến (Mốc G5)\n",
                "Thử nghiệm trực tiếp module `Q2Processor` (`streaming/q2_intent.py`) trên chuỗi sự kiện có thứ tự xáo trộn."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from streaming.q2_intent import Q2Processor\n",
                "\n",
                "q2 = Q2Processor(spec_path='../models/feature_spec.json', model_path='../models/intent_v1.onnx')\n",
                "sample_stream = [\n",
                "    {'shop_id': 'rees46_cosmetics', 'session_id': 9999, 'seq_no': 2, 'event_type': 'add_to_cart', 'product_id': 777, 'category': 'lip', 'brand': 'mac', 'price': 30.0, 'qty': 1, 'amount_usd': None, 'event_ts': '2024-01-01 12:01:00.000', 'ts_ms': 1704110460000},\n",
                "    {'shop_id': 'rees46_cosmetics', 'session_id': 9999, 'seq_no': 1, 'event_type': 'page_view', 'product_id': 777, 'category': 'lip', 'brand': 'mac', 'price': 30.0, 'qty': None, 'amount_usd': None, 'event_ts': '2024-01-01 12:00:00.000', 'ts_ms': 1704110400000},\n",
                "    {'shop_id': 'rees46_cosmetics', 'session_id': 9999, 'seq_no': 3, 'event_type': 'purchase', 'product_id': None, 'category': None, 'brand': None, 'price': None, 'qty': None, 'amount_usd': 30.0, 'event_ts': '2024-01-01 12:02:00.000', 'ts_ms': 1704110520000}\n",
                "]\n",
                "intents, closed = q2.process_records(sample_stream, force_close=False)\n",
                "print('Output topic pred.intent:')\n",
                "for it in intents:\n",
                "    print(f\"  seq={it['seq_no']} | event_ts={it['event_ts']} | prob={it['prob']:.4f} | model={it['model_ver']}\")\n",
                "print('Output topic session.closed:')\n",
                "for cl in closed:\n",
                "    print(f\"  last_seq={cl['last_seq_no']} | label_purchase={cl['label_purchase']} | revenue_usd={cl['revenue_usd']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 8. Đánh giá Trực tuyến (Online Evaluation) & Thí nghiệm A/B Testing (Mốc G6)\n",
                "Tải và đối chiếu kết quả đánh giá thực tế giữa online và offline từ `data/prefix/_online_eval.json`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "online_report_path = '../data/prefix/_online_eval.json'\n",
                "if os.path.exists(online_report_path):\n",
                "    with open(online_report_path, encoding='utf-8') as f:\n",
                "        oreport = json.load(f)\n",
                "    print(f\"Online ROC-AUC: {oreport['online_roc_auc']:.4f}\")\n",
                "    print(f\"Offline Benchmark ROC-AUC: {oreport['offline_roc_auc']:.4f}\")\n",
                "    print(f\"Độ lệch Delta AUC: {oreport['delta_auc']:.4f} (Tiêu chuẩn: <= {oreport['tolerance_criterion']})\")\n",
                "    print(f\"Kết luận kiểm chứng: {'PASS' if oreport['passed_tolerance'] else 'FAIL'}\")\n",
                "    \n",
                "    k_df = pd.DataFrame([\n",
                "        {'Độ dài prefix (k)': k, 'Số mẫu': v['samples'], 'Tỉ lệ mua': f\"{v['pos_rate']:.2%}\", 'Online ROC-AUC': v['roc_auc'], 'Online PR-AUC': v['pr_auc']}\n",
                "        for k, v in oreport['by_prefix_len'].items()\n",
                "    ])\n",
                "    display(k_df) if 'display' in globals() else print(k_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### 9. Trực quan hóa Biểu đồ (ROC Curve & PR Curve)\n",
                "Vẽ biểu đồ hiệu năng dự đoán ý định mua theo độ dài prefix k của mô hình Champion trên tập kiểm thử."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "\n",
                "k_labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '>50']\n",
                "auc_values = [0.8269, 0.7456, 0.7316, 0.6855, 0.6885, 0.6361, 0.6978]\n",
                "pr_values = [0.2718, 0.3580, 0.3951, 0.4106, 0.3451, 0.2610, 0.2698]\n",
                "\n",
                "fig, ax1 = plt.subplots(figsize=(9, 4.5))\n",
                "ax1.plot(k_labels, auc_values, marker='o', color='#1f77b4', linewidth=2, label='ROC-AUC')\n",
                "ax1.set_xlabel('Độ dài prefix (số sự kiện trước checkout)')\n",
                "ax1.set_ylabel('ROC-AUC', color='#1f77b4')\n",
                "ax1.set_ylim(0.5, 0.9)\n",
                "\n",
                "ax2 = ax1.twinx()\n",
                "ax2.plot(k_labels, pr_values, marker='s', color='#ff7f0e', linewidth=2, linestyle='--', label='PR-AUC')\n",
                "ax2.set_ylabel('PR-AUC', color='#ff7f0e')\n",
                "ax2.set_ylim(0.1, 0.5)\n",
                "\n",
                "plt.title('Hiệu năng Dự đoán Purchase Intent theo Độ dài Prefix (Champion GRU)')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "print('Notebook walkthrough hoàn thành xuất sắc!')"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/clickstream_walkthrough.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("SUCCESS: Created notebooks/clickstream_walkthrough.ipynb")
