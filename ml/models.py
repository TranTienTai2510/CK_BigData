"""Deep Learning models (GRU + Transformer) for real-time purchase-intent prediction on clickstream prefixes.

Input contract (matches ml.features.window_to_tensor / session_tensors and Q2 mapInPandas ONNX inference):
  - x_cat  : int64   [B, L, 3]   (type_idx, cat_idx, brand_idx), LEFT-padded with PAD=0 at 0 .. L - length - 1
  - x_num  : float32 [B, L, 10]  (NUM_FEATURES of ml/features.py), LEFT-padded with 0.0
  - length : int64   [B]         number of valid steps in the window (1 .. L)
  - shop   : int64   [B]         shop vocabulary index (FIRST_IDX ..)

Output contract:
  - prob   : float32 [B]         calibrated purchase probability in (0, 1)

Because prefixes are LEFT-padded (the most recent event is always at index L - 1, and padding is at indices
< L - length):
  - Mask of valid positions: pos_idx >= (L - length), shape [B, L].
  - Both GRU and Transformer zero out embeddings/projections on padded positions, pool the last step h[:, -1, :]
    concatenated with masked mean/max pooling over valid positions and the shop embedding, and pass through an MLP
    classification head.
  - Temperature / Platt calibration parameters (scale `a`, bias `b` per shop or global) are stored as buffers inside
    the nn.Module (`calib_a`, `calib_b`), so `torch.onnx.export` bakes probability calibration directly into the ONNX
    computation graph (`intent_v1.onnx`), guaranteeing < 1e-4 parity between PyTorch and ONNX Runtime.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml import features as FT


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
        # x_cat: [B, L, 3], x_num: [B, L, 10], valid_mask: [B, L, 1] float32 (1.0 on real steps, 0.0 on PAD)
        parts = [self.emb_type(x_cat[:, :, 0])]
        if self.use_cat_brand:
            parts.append(self.emb_cat(x_cat[:, :, 1]))
            parts.append(self.emb_brand(x_cat[:, :, 2]))
        if self.use_num:
            parts.append(F.gelu(self.num_proj(self.num_norm(x_num))))
        h = self.proj(torch.cat(parts, dim=-1))
        return h * valid_mask


class CausalSelfAttentionBlock(nn.Module):
    """ONNX-friendly pre-norm Transformer block with explicit mask (no nested tensor / FlashAttention ops that
    could diverge across ONNX opsets)."""
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
        # x: [B, L, D], attn_bias: [B, 1, L, L] (0 for allowed, -1e4 for masked), valid_mask: [B, L, 1]
        bsz, seq_len, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(bsz, seq_len, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                # [B, H, L, d_head]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias
        weights = torch.softmax(scores, dim=-1)
        weights = self.drop(weights)
        ctx = torch.matmul(weights, v).transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        x = (x + self.drop(self.out_proj(ctx))) * valid_mask
        x = (x + self.ff(self.ln2(x))) * valid_mask
        return x


class IntentModel(nn.Module):
    """Unified purchase-intent sequence model supporting arch='gru' and arch='transformer', plus ablation flags."""
    def __init__(self, spec, arch="gru", d_model=64, n_layers=2, n_heads=4, dropout=0.1,
                 use_cat_brand=True, use_num=True):
        super().__init__()
        self.arch = arch
        self.max_len = int(spec["max_len"])
        self.d_model = d_model
        card = spec["cardinality"]

        self.step_emb = StepEmbedding(card, d_model=d_model, dropout=dropout,
                                      use_cat_brand=use_cat_brand, use_num=use_num)
        self.shop_emb = nn.Embedding(int(card["shop"]), 8, padding_idx=FT.PAD)

        # Fixed position index buffer [1, L] for left-padding mask computation
        self.register_buffer("pos_idx", torch.arange(self.max_len, dtype=torch.int64).unsqueeze(0))
        # Causal mask buffer [1, 1, L, L]: 0.0 on j <= i, -1e4 on j > i
        causal = torch.triu(torch.full((self.max_len, self.max_len), -1e4, dtype=torch.float32), diagonal=1)
        self.register_buffer("causal_bias", causal.unsqueeze(0).unsqueeze(0))

        if arch == "gru":
            self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        elif arch == "transformer":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, d_model))
            nn.init.normal_(self.pos_emb, std=0.02)
            self.blocks = nn.ModuleList([
                CausalSelfAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_model * 2, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.norm_out = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"unknown architecture: {arch}")

        self.use_num = bool(use_num)
        self.use_cat_brand = bool(use_cat_brand)
        # Direct residual projection of last-step numerical features + summary stats (mean/max over valid steps)
        num_skip_dim = (FT.N_NUM * 2) if self.use_num else 0
        head_in = d_model * 3 + 8 + 1 + num_skip_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Per-shop Platt / temperature calibration parameters: logit_cal = calib_a[shop] * logit + calib_b[shop]
        n_shop = int(card["shop"])
        self.register_buffer("calib_a", torch.ones(n_shop, dtype=torch.float32))
        self.register_buffer("calib_b", torch.zeros(n_shop, dtype=torch.float32))

    def forward_logits(self, x_cat, x_num, length, shop):
        # valid_mask: left-padded -> valid indices are >= (max_len - length)
        start = (self.max_len - length.clamp(min=1, max=self.max_len)).unsqueeze(1)  # [B, 1]
        valid = (self.pos_idx >= start).to(x_num.dtype).unsqueeze(-1)                # [B, L, 1]

        h = self.step_emb(x_cat, x_num, valid)
        if self.arch == "gru":
            h0 = torch.zeros(self.rnn.num_layers, x_num.shape[0], self.d_model,
                             dtype=x_num.dtype, device=x_num.device)
            h, _ = self.rnn(h, h0)
            h = h * valid
        else:
            h = (h + self.pos_emb) * valid
            # Mask out padded keys j < start: [B, 1, 1, L]
            key_pad_bias = (1.0 - valid.squeeze(-1)).unsqueeze(1).unsqueeze(2) * (-1e4)
            attn_bias = self.causal_bias + key_pad_bias
            for blk in self.blocks:
                h = blk(h, attn_bias, valid)
            h = self.norm_out(h) * valid

        h_last = h[:, -1, :]
        denom = valid.sum(dim=1).clamp(min=1.0)
        h_mean = h.sum(dim=1) / denom
        # Masked max pooling over valid sequence positions
        h_masked_for_max = h + (1.0 - valid) * (-1e4)
        h_max = h_masked_for_max.max(dim=1)[0]
        s_emb = self.shop_emb(shop)
        l_feat = torch.log1p(length.to(x_num.dtype)).unsqueeze(-1)
        parts = [h_last, h_mean, h_max, s_emb, l_feat]
        if self.use_num:
            x_num_valid = x_num * valid
            x_num_last = x_num[:, -1, :]
            x_num_mean = x_num_valid.sum(dim=1) / denom
            parts.extend([x_num_last, x_num_mean])
        z = torch.cat(parts, dim=-1)
        raw_logit = self.head(z).squeeze(-1)

        a = self.calib_a[shop]
        b = self.calib_b[shop]
        return a * raw_logit + b

    def forward(self, x_cat, x_num, length, shop):
        return torch.sigmoid(self.forward_logits(x_cat, x_num, length, shop))

    @torch.no_grad()
    def set_calibration(self, shop_params):
        """shop_params: dict shop_idx (int) -> (a, b)."""
        for sidx, (a, b) in shop_params.items():
            self.calib_a[int(sidx)] = float(a)
            self.calib_b[int(sidx)] = float(b)
