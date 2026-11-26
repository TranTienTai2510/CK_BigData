"""Unified Deep Learning models for Session-prefix Purchase Intent Prediction (Binary Classification).

All architectures solve ONE unified task:
  Given an observed session prefix S = (e_1, ..., e_k) truncated to the last L = 50 steps,
  predict calibrated purchase probability P(session ends with purchase | S) in (0, 1).

Shared Input/Output Contract (matches ml.features and Spark Streaming Q2 ONNX inference):
  - x_cat  : int64   [B, L, 3]   (type_idx, cat_idx, brand_idx), LEFT-padded with PAD=0
  - x_num  : float32 [B, L, 10]  (NUM_FEATURES of ml/features.py), LEFT-padded with 0.0
  - length : int64   [B]         number of valid steps in window (1 .. L)
  - shop   : int64   [B]         shop vocabulary index (FIRST_IDX ..)
  -> prob  : float32 [B]         calibrated probability in (0, 1)

Supported Architectures (`MODEL_ARCHS`):
  Baseline Sequence / Tabular-Hybrid Candidates:
    1. 'gru'              : Gated Recurrent Unit sequence classifier
    2. 'bilstm_attention' : Bidirectional LSTM + Additive (Bahdanau) Attention over prefix
    3. 'tcn'              : Causal Dilated Temporal Convolutional Network
    4. 'transformer'      : Pre-LN Causal Self-Attention Transformer
    5. 'mixer'            : Fixed-window MLP-Mixer (token-mixing + channel-mixing)
    6. 'hybrid'           : Hybrid GRU + Wide & Deep (GRU + ONNX-native 44-dim Wide prefix MLP)
  Improved Champion Architecture (developed after baseline comparison):
    7. 'bilstm_attention_v2' : Enhanced Bi-LSTM with Dual Additive Attention Pooling, max-pooling,
                               StepEmbedding residual skip, and 4 log-scale behavioral ratio features
                               computed natively in ONNX + Per-Shop Platt Calibration.
    8. 'hybrid_v2'           : Enhanced Hybrid GRU with Shop-Conditioned Attention Pooling,
                               ONNX-native 44-dim Wide Prefix Features, and Shop-Conditioned FiLM.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml import features as FT

BASELINE_DL_ARCHS = ("gru", "hybrid", "tcn", "bilstm_attention", "mixer", "transformer")
ALL_DL_ARCHS = BASELINE_DL_ARCHS + ("bilstm_attention_v2", "hybrid_v2")
WIDE_FEATURE_DIM = 44


class StepEmbedding(nn.Module):
    """Embeds categorical indices (type_idx, cat_idx, brand_idx) + projects numerical features x_num."""

    def __init__(self, card, d_model=64, dropout=0.1, use_cat_brand=True, use_num=True):
        super().__init__()
        self.use_cat_brand = bool(use_cat_brand)
        self.use_num = bool(use_num)
        d_type = 8
        d_cat = 16 if use_cat_brand else 0
        d_brand = 8 if use_cat_brand else 0
        d_num = 16 if use_num else 0

        self.emb_type = nn.Embedding(int(card["type_idx"]), d_type, padding_idx=FT.PAD)
        if self.use_cat_brand:
            self.emb_cat = nn.Embedding(int(card["cat_idx"]), d_cat, padding_idx=FT.PAD)
            self.emb_brand = nn.Embedding(int(card["brand_idx"]), d_brand, padding_idx=FT.PAD)
        if self.use_num:
            self.num_norm = nn.LayerNorm(FT.N_NUM)
            self.num_proj = nn.Linear(FT.N_NUM, d_num)

        in_dim = d_type + d_cat + d_brand + d_num
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_cat, x_num, valid_mask):
        parts = [self.emb_type(x_cat[:, :, 0])]
        if self.use_cat_brand:
            parts.append(self.emb_cat(x_cat[:, :, 1]))
            parts.append(self.emb_brand(x_cat[:, :, 2]))
        if self.use_num:
            parts.append(F.gelu(self.num_proj(self.num_norm(x_num))))
        h = self.proj(torch.cat(parts, dim=-1))
        return h * valid_mask


class CausalSelfAttentionBlock(nn.Module):
    """ONNX-friendly Pre-LN Transformer block with explicit causal + padding bias."""

    def __init__(self, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias, valid_mask):
        bsz, seq_len, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(bsz, seq_len, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias
        weights = self.drop(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(weights, v).transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        x = (x + self.drop(self.out_proj(ctx))) * valid_mask
        x = (x + self.ff(self.ln2(x))) * valid_mask
        return x


class CausalConv1dBlock(nn.Module):
    """Dilated Causal Conv1d residual block (left-only padding, ONNX opset 18 safe)."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.ln2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid_mask):
        # x: [B, L, D], valid_mask: [B, L, 1]
        h = x.transpose(1, 2)  # [B, D, L]
        h = F.pad(h, (self.pad, 0))
        h = self.conv1(h).transpose(1, 2)
        h = self.drop(F.gelu(self.ln1(h))) * valid_mask

        h = h.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv2(h).transpose(1, 2)
        h = self.drop(F.gelu(self.ln2(h))) * valid_mask
        return (x + h) * valid_mask


class MLPMixerBlock(nn.Module):
    """Token-mixing + Channel-mixing MLP block for fixed window length L."""

    def __init__(self, seq_len, d_model, token_dim=None, channel_dim=None, dropout=0.1):
        super().__init__()
        token_dim = token_dim or max(seq_len, 32)
        channel_dim = channel_dim or d_model * 2
        self.ln_tok = nn.LayerNorm(d_model)
        self.tok_mlp = nn.Sequential(
            nn.Linear(seq_len, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.ln_ch = nn.LayerNorm(d_model)
        self.ch_mlp = nn.Sequential(
            nn.Linear(d_model, channel_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid_mask):
        h = self.ln_tok(x).transpose(1, 2)
        x = (x + self.tok_mlp(h).transpose(1, 2)) * valid_mask
        x = (x + self.ch_mlp(self.ln_ch(x))) * valid_mask
        return x


class AdditiveAttentionPool(nn.Module):
    """Masked Bahdanau additive attention pooling over valid prefix steps."""

    def __init__(self, d_model, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or d_model
        self.proj = nn.Linear(d_model, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h_seq, valid_2d):
        # h_seq: [B, L, D], valid_2d: [B, L]
        e = self.score(torch.tanh(self.proj(h_seq))).squeeze(-1)  # [B, L]
        e = e + (1.0 - valid_2d) * (-1e4)
        alpha = torch.softmax(e, dim=-1).unsqueeze(1)             # [B, 1, L]
        return torch.bmm(alpha, h_seq).squeeze(1)                 # [B, D]


class QueryAttentionPool(nn.Module):
    """Conditioned attention pooling: query (last step + shop context) attends over valid steps."""

    def __init__(self, d_model, query_dim):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, h_seq, query, valid_2d):
        q = self.q_proj(query).unsqueeze(1)                       # [B, 1, D]
        k = self.k_proj(h_seq)                                    # [B, L, D]
        v = self.v_proj(h_seq)                                    # [B, L, D]
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale  # [B, L]
        scores = scores + (1.0 - valid_2d) * (-1e4)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(1)        # [B, 1, L]
        return torch.bmm(alpha, v).squeeze(1)                     # [B, D]


def extract_wide_prefix_features(x_cat, x_num, valid, length, pos_idx):
    """Compute WIDE_FEATURE_DIM (44) domain-grounded tabular prefix summary features purely inside PyTorch/ONNX.

    Guarantees 0 training-serving skew with Q2 ONNX inference without adding extra ONNX input tensors.
    """
    dtype = x_num.dtype
    max_len = x_num.shape[1]
    valid_2d = valid.squeeze(-1)                                  # [B, L]
    denom = valid_2d.sum(dim=1, keepdim=True).clamp(min=1.0)      # [B, 1]
    x_num_valid = x_num * valid                                   # [B, L, 10]

    # Current step (L - 1), previous step (L - 2 if length > 1), and first step in window
    x_num_last = x_num[:, -1, :]                                  # [B, 10]
    has_prev = (length > 1).to(dtype).unsqueeze(-1)               # [B, 1]
    x_num_prev = x_num[:, -2, :] * has_prev                       # [B, 10]
    prev_key = torch.cat([x_num_prev[:, 0:1], x_num_prev[:, 2:3],
                          x_num_prev[:, 4:5], x_num_prev[:, 8:9]], dim=-1)  # [B, 4]

    start = (max_len - length.clamp(min=1, max=max_len)).unsqueeze(1)       # [B, 1]
    first_mask = (pos_idx == start).to(dtype).unsqueeze(-1)                 # [B, L, 1]
    x_num_first = (x_num_valid * first_mask).sum(dim=1)                     # [B, 10]
    first_log_price = x_num_first[:, 0:1]                                   # [B, 1]

    # Summary statistics over valid window steps
    x_num_mean = x_num_valid.sum(dim=1) / denom                             # [B, 10]
    x_num_max = (x_num + (1.0 - valid) * (-1e4)).max(dim=1)[0]              # [B, 10]
    min_log_price = (x_num[:, :, 0:1] + (1.0 - valid) * 1e4).min(dim=1)[0]  # [B, 1]
    max_log_price = x_num_max[:, 0:1]                                       # [B, 1]
    max_log_dt = x_num_max[:, 2:3]                                          # [B, 1]

    price_var = (((x_num[:, :, 0:1] - x_num_mean[:, 0:1].unsqueeze(1)) ** 2) * valid).sum(dim=1) / denom
    dt_var = (((x_num[:, :, 2:3] - x_num_mean[:, 2:3].unsqueeze(1)) ** 2) * valid).sum(dim=1) / denom
    price_std = torch.sqrt(price_var.clamp(min=1e-6))                       # [B, 1]
    dt_std = torch.sqrt(dt_var.clamp(min=1e-6))                             # [B, 1]

    # Behavioral rates & session momentum (indices from ml.features.NUM_FEATURES)
    n_view = torch.exp(x_num_last[:, 5:6].clamp(max=15.0)) - 1.0
    n_add = torch.exp(x_num_last[:, 6:7].clamp(max=15.0)) - 1.0
    n_rem = torch.exp(x_num_last[:, 7:8].clamp(max=15.0)) - 1.0
    pos_k = (torch.exp(x_num_last[:, 9:10].clamp(max=15.0)) - 1.0).clamp(min=1.0)
    t_first_s = torch.exp(x_num_last[:, 3:4].clamp(max=18.0)) - 1.0

    add_per_step = n_add / pos_k
    view_per_step = n_view / pos_k
    remove_per_add = n_rem / n_add.clamp(min=1.0)
    has_cart = (x_num_last[:, 8:9] > 0.0).to(dtype)
    sec_per_step = torch.log1p((t_first_s / pos_k).clamp(min=0.0))
    price_delta = x_num_last[:, 0:1] - x_num_mean[:, 0:1]
    price_vs_max = x_num_last[:, 0:1] - max_log_price
    dt_accel = x_num_last[:, 2:3] - x_num_mean[:, 2:3]

    # Category & brand switching rates across adjacent valid steps
    pair_valid = valid_2d[:, :-1] * valid_2d[:, 1:]
    pair_denom = (denom - 1.0).clamp(min=1.0)
    cat_switch_rate = ((x_cat[:, 1:, 1] != x_cat[:, :-1, 1]).to(dtype) * pair_valid).sum(dim=1, keepdim=True) / pair_denom
    brand_switch_rate = ((x_cat[:, 1:, 2] != x_cat[:, :-1, 2]).to(dtype) * pair_valid).sum(dim=1, keepdim=True) / pair_denom

    # Recent-5-step intent density (focuses attention on recent add_to_cart vs passive browsing)
    r5_valid = valid_2d[:, -5:]
    r5_denom = r5_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    recent_add_rate = ((x_cat[:, -5:, 0] == FT.TYPE_IDX["add_to_cart"]).to(dtype) * r5_valid).sum(dim=1, keepdim=True) / r5_denom
    recent_view_rate = ((x_cat[:, -5:, 0] == FT.TYPE_IDX["page_view"]).to(dtype) * r5_valid).sum(dim=1, keepdim=True) / r5_denom

    log_len = torch.log1p(length.to(dtype)).unsqueeze(-1)
    is_k1 = (length == 1).to(dtype).unsqueeze(-1)

    return torch.cat([
        x_num_last,                                               # 10
        x_num_mean,                                               # 10
        prev_key,                                                 # 4
        first_log_price,                                          # 1
        max_log_price, min_log_price, max_log_dt, price_std, dt_std,  # 5
        add_per_step, view_per_step, remove_per_add, has_cart,    # 4
        sec_per_step, price_delta, price_vs_max, dt_accel,        # 4
        cat_switch_rate, brand_switch_rate,                       # 2
        recent_add_rate, recent_view_rate,                        # 2
        log_len, is_k1,                                           # 2
    ], dim=-1)                                                    # Total = 44


def extract_log_ratio_features(x_cat, x_num, valid):
    """Extract 4 log-scale behavioral ratio features directly inside PyTorch/ONNX."""
    dtype = x_num.dtype
    valid_2d = valid.squeeze(-1)
    denom = valid_2d.sum(dim=1, keepdim=True).clamp(min=1.0)
    x_num_last = x_num[:, -1, :]
    x_num_mean = (x_num * valid).sum(dim=1) / denom

    price_delta = x_num_last[:, 0:1] - x_num_mean[:, 0:1]
    dt_delta = x_num_last[:, 2:3] - x_num_mean[:, 2:3]
    has_cart = (x_num_last[:, 8:9] > 0.0).to(dtype)
    pair_valid = valid_2d[:, :-1] * valid_2d[:, 1:]
    cat_sw = ((x_cat[:, 1:, 1] != x_cat[:, :-1, 1]).to(dtype) * pair_valid).sum(dim=1, keepdim=True) / (denom - 1.0).clamp(min=1.0)
    return torch.cat([price_delta, dt_delta, has_cart, cat_sw], dim=-1)


class IntentModel(nn.Module):
    """Unified binary classification model for session-prefix purchase intent prediction."""

    def __init__(self, spec, arch="gru", d_model=32, n_layers=1, n_heads=4, dropout=0.1,
                 use_cat_brand=True, use_num=True, use_wide=True, use_attn=True, use_film=True):
        super().__init__()
        if arch not in ALL_DL_ARCHS:
            raise ValueError(f"unknown architecture '{arch}'; expected one of {ALL_DL_ARCHS}")
        self.arch = arch
        self.max_len = int(spec["max_len"])
        self.d_model = int(d_model)
        self.use_num = bool(use_num)
        self.use_cat_brand = bool(use_cat_brand)
        self.use_wide = bool(use_wide) if arch in ("hybrid", "hybrid_v2") else False
        self.use_log_ratios = bool(use_wide) if arch == "bilstm_attention_v2" else False
        self.use_emb_skip = bool(use_film) if arch == "bilstm_attention_v2" else False
        self.use_attn = bool(use_attn) if arch in ("bilstm_attention", "bilstm_attention_v2", "hybrid_v2") else False
        self.use_film = bool(use_film) if arch == "hybrid_v2" else False
        card = spec["cardinality"]

        self.step_emb = StepEmbedding(card, d_model=d_model, dropout=dropout,
                                      use_cat_brand=use_cat_brand, use_num=use_num)
        self.shop_emb = nn.Embedding(int(card["shop"]), 8, padding_idx=FT.PAD)

        self.register_buffer("pos_idx", torch.arange(self.max_len, dtype=torch.int64).unsqueeze(0))
        causal = torch.triu(torch.full((self.max_len, self.max_len), -1e4, dtype=torch.float32), diagonal=1)
        self.register_buffer("causal_bias", causal.unsqueeze(0).unsqueeze(0))

        # Sequence backbone
        if arch in ("gru", "hybrid", "hybrid_v2"):
            self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        elif arch in ("bilstm_attention", "bilstm_attention_v2"):
            assert d_model % 2 == 0, f"d_model must be even for {arch}"
            self.rnn = nn.LSTM(d_model, d_model // 2, num_layers=n_layers, batch_first=True,
                               bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
            self.attn_pool = AdditiveAttentionPool(d_model, attn_dim=d_model)
            if arch == "bilstm_attention_v2" and self.use_attn:
                self.attn_pool2 = AdditiveAttentionPool(d_model, attn_dim=d_model)
        elif arch == "tcn":
            n_tcn_blocks = max(n_layers + 1, 2)
            self.tcn_blocks = nn.ModuleList([
                CausalConv1dBlock(d_model, kernel_size=3, dilation=2 ** i, dropout=dropout)
                for i in range(n_tcn_blocks)
            ])
        elif arch == "transformer":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, d_model))
            nn.init.normal_(self.pos_emb, std=0.02)
            self.blocks = nn.ModuleList([
                CausalSelfAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_model * 2, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.norm_out = nn.LayerNorm(d_model)
        elif arch == "mixer":
            n_mix_blocks = max(n_layers + 1, 2)
            self.mixer_blocks = nn.ModuleList([
                MLPMixerBlock(self.max_len, d_model, token_dim=max(self.max_len, 32),
                              channel_dim=d_model * 2, dropout=dropout)
                for _ in range(n_mix_blocks)
            ])
            self.norm_out = nn.LayerNorm(d_model)

        # Wide & Attention & Shop-Gating enhancements
        wide_out_dim = 0
        if self.use_wide:
            wide_in = WIDE_FEATURE_DIM if self.use_num else 4
            wide_out_dim = d_model
            self.wide_norm = nn.LayerNorm(wide_in)
            self.wide_mlp = nn.Sequential(
                nn.Linear(wide_in, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, wide_out_dim),
                nn.LayerNorm(wide_out_dim),
                nn.GELU(),
            )
        elif self.use_log_ratios and self.use_num:
            wide_out_dim = 4

        attn_dim = 0
        if arch == "hybrid_v2" and self.use_attn:
            self.query_attn = QueryAttentionPool(d_model, query_dim=d_model + 8)
            attn_dim = d_model

        if arch == "bilstm_attention":
            seq_repr_dim = d_model * 3  # h_last, h_attn, h_mean
        elif arch == "bilstm_attention_v2":
            seq_repr_dim = d_model * (4 + (1 if self.use_attn else 0) + (1 if self.use_emb_skip else 0))
        else:
            seq_repr_dim = d_model * 3 + attn_dim  # h_last, h_mean, h_max (+ h_attn for hybrid_v2)

        num_skip_dim = (FT.N_NUM * 2) if (self.use_num and not self.use_wide) else 0
        fused_dim = seq_repr_dim + 8 + 1 + num_skip_dim + wide_out_dim

        if self.use_film and arch == "hybrid_v2":
            self.film_gamma = nn.Linear(8, fused_dim)
            self.film_beta = nn.Linear(8, fused_dim)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Per-shop Platt / temperature calibration parameters baked into ONNX graph
        n_shop = int(card["shop"])
        self.register_buffer("calib_a", torch.ones(n_shop, dtype=torch.float32))
        self.register_buffer("calib_b", torch.zeros(n_shop, dtype=torch.float32))

    def _encode_sequence(self, h, valid):
        bsz = h.shape[0]
        if self.arch in ("gru", "hybrid", "hybrid_v2"):
            h0 = torch.zeros(self.rnn.num_layers, bsz, self.d_model, dtype=h.dtype, device=h.device)
            h, _ = self.rnn(h, h0)
            return h * valid
        if self.arch in ("bilstm_attention", "bilstm_attention_v2"):
            d_half = self.d_model // 2
            h0 = torch.zeros(self.rnn.num_layers * 2, bsz, d_half, dtype=h.dtype, device=h.device)
            c0 = torch.zeros(self.rnn.num_layers * 2, bsz, d_half, dtype=h.dtype, device=h.device)
            h, _ = self.rnn(h, (h0, c0))
            return h * valid
        if self.arch == "tcn":
            for blk in self.tcn_blocks:
                h = blk(h, valid)
            return h
        if self.arch == "transformer":
            h = (h + self.pos_emb) * valid
            key_pad_bias = (1.0 - valid.squeeze(-1)).unsqueeze(1).unsqueeze(2) * (-1e4)
            attn_bias = self.causal_bias + key_pad_bias
            for blk in self.blocks:
                h = blk(h, attn_bias, valid)
            return self.norm_out(h) * valid
        if self.arch == "mixer":
            for blk in self.mixer_blocks:
                h = blk(h, valid)
            return self.norm_out(h) * valid
        raise RuntimeError(f"Unhandled arch: {self.arch}")

    def forward_logits(self, x_cat, x_num, length, shop):
        start = (self.max_len - length.clamp(min=1, max=self.max_len)).unsqueeze(1)  # [B, 1]
        valid = (self.pos_idx >= start).to(x_num.dtype).unsqueeze(-1)                # [B, L, 1]
        valid_2d = valid.squeeze(-1)

        emb = self.step_emb(x_cat, x_num, valid)
        h = self._encode_sequence(emb, valid)

        h_last = h[:, -1, :]
        denom = valid.sum(dim=1).clamp(min=1.0)
        h_mean = h.sum(dim=1) / denom
        s_emb = self.shop_emb(shop)
        l_feat = torch.log1p(length.to(x_num.dtype)).unsqueeze(-1)

        if self.arch == "bilstm_attention":
            h_attn = self.attn_pool(h, valid_2d)
            parts = [h_last, h_attn, h_mean, s_emb, l_feat]
        elif self.arch == "bilstm_attention_v2":
            h_attn1 = self.attn_pool(h, valid_2d)
            h_max = (h + (1.0 - valid) * (-1e4)).max(dim=1)[0]
            parts = [h_last, h_attn1, h_mean, h_max]
            if self.use_attn:
                parts.append(self.attn_pool2(h, valid_2d))
            if self.use_emb_skip:
                parts.append(emb[:, -1, :])
            parts.extend([s_emb, l_feat])
        else:
            h_max = (h + (1.0 - valid) * (-1e4)).max(dim=1)[0]
            parts = [h_last, h_mean, h_max]
            if self.arch == "hybrid_v2" and self.use_attn:
                h_attn = self.query_attn(h, torch.cat([h_last, s_emb], dim=-1), valid_2d)
                parts.append(h_attn)
            parts.extend([s_emb, l_feat])

        if self.use_wide:
            if self.use_num:
                wide_raw = extract_wide_prefix_features(x_cat, x_num, valid, length, self.pos_idx)
            else:
                pair_valid = valid_2d[:, :-1] * valid_2d[:, 1:]
                pair_denom = (denom - 1.0).clamp(min=1.0)
                cat_sw = ((x_cat[:, 1:, 1] != x_cat[:, :-1, 1]).to(x_num.dtype) * pair_valid).sum(dim=1, keepdim=True) / pair_denom
                r5_v = valid_2d[:, -5:]
                r5_add = ((x_cat[:, -5:, 0] == FT.TYPE_IDX["add_to_cart"]).to(x_num.dtype) * r5_v).sum(dim=1, keepdim=True) / r5_v.sum(dim=1, keepdim=True).clamp(min=1.0)
                wide_raw = torch.cat([cat_sw, r5_add, l_feat, (length == 1).to(x_num.dtype).unsqueeze(-1)], dim=-1)
            parts.append(self.wide_mlp(self.wide_norm(wide_raw)))
        else:
            if self.use_num:
                x_num_valid = x_num * valid
                parts.extend([x_num[:, -1, :], x_num_valid.sum(dim=1) / denom])
                if self.use_log_ratios:
                    parts.append(extract_log_ratio_features(x_cat, x_num, valid))

        z = torch.cat(parts, dim=-1)
        if self.use_film:
            gamma = torch.tanh(self.film_gamma(s_emb))
            beta = self.film_beta(s_emb)
            z = z * (1.0 + gamma) + beta

        raw_logit = self.head(z).squeeze(-1)
        return self.calib_a[shop] * raw_logit + self.calib_b[shop]

    def forward(self, x_cat, x_num, length, shop):
        return torch.sigmoid(self.forward_logits(x_cat, x_num, length, shop))

    @torch.no_grad()
    def set_calibration(self, shop_params):
        """shop_params: dict shop_idx (int) -> (a, b)."""
        for sidx, (a, b) in shop_params.items():
            self.calib_a[int(sidx)] = float(a)
            self.calib_b[int(sidx)] = float(b)
