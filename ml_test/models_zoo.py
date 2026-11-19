"""Neural architectures sharing the purchase-intent feature and ONNX contracts."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml import features as FT
from ml.models import IntentModel, StepEmbedding


FLAT_FEATURES = 31


def window_to_flat_features(x_cat, x_num, length):
    """Build 31 wide features from left-padded NumPy tensors [B,L,3] and [B,L,10]."""
    import numpy as np

    cats = np.asarray(x_cat)
    nums = np.asarray(x_num, dtype=np.float32)
    lengths = np.asarray(length, dtype=np.int64).reshape(-1)
    if cats.ndim != 3 or cats.shape[2] != FT.N_CAT:
        raise ValueError(f"x_cat must have shape [B,L,{FT.N_CAT}]")
    if nums.ndim != 3 or nums.shape[:2] != cats.shape[:2] or nums.shape[2] != FT.N_NUM:
        raise ValueError(f"x_num must have shape [B,L,{FT.N_NUM}] matching x_cat")
    if len(lengths) != len(cats) or np.any(lengths < 1) or np.any(lengths > cats.shape[1]):
        raise ValueError("length must contain one value in 1..L per sample")

    batch, width, _ = cats.shape
    valid = np.arange(width)[None, :] >= (width - lengths)[:, None]
    last = nums[:, -1]
    last_cat = cats[:, -1, :].astype(np.float32)
    prev_idx = np.maximum(width - 1 - 1, 0)
    prev_num = np.zeros_like(last)
    prev_cat = np.zeros((batch, 2), dtype=np.float32)
    has_prev = lengths > 1
    if width > 1:
        prev_num[has_prev] = nums[has_prev, prev_idx]
        prev_cat[has_prev] = cats[has_prev, prev_idx, :2]
    first_idx = width - lengths
    rows = np.arange(batch)
    first_cat = cats[rows, first_idx, :2].astype(np.float32)
    price = nums[:, :, 0]
    delta = nums[:, :, 2]
    masked_price = np.where(valid, price, np.nan)
    masked_delta = np.where(valid, delta, np.nan)
    mean_price = np.nanmean(masked_price, axis=1)
    max_price = np.nanmax(np.where(valid, price, -np.inf), axis=1)
    min_price = np.nanmin(np.where(valid, price, np.inf), axis=1)
    mean_delta = np.nanmean(masked_delta, axis=1)
    max_delta = np.nanmax(np.where(valid, delta, -np.inf), axis=1)
    count_add = np.expm1(np.maximum(nums[:, -1, 6], 0.0))
    count_view = np.expm1(np.maximum(nums[:, -1, 5], 0.0))
    count_remove = np.expm1(np.maximum(nums[:, -1, 7], 0.0))
    signed_log_cart = nums[:, -1, 8]
    cart_units = np.sign(signed_log_cart) * np.expm1(np.abs(signed_log_cart))
    switches = np.zeros(batch, dtype=np.float32)
    for j in range(1, width):
        pair_valid = valid[:, j] & valid[:, j - 1]
        switches += (pair_valid & (cats[:, j, 1] != cats[:, j - 1, 1])).astype(np.float32)
    flat = np.column_stack((
        last_cat, prev_cat, first_cat, last,
        prev_num[:, 0], prev_num[:, 2], nums[rows, first_idx, 0],
        mean_price, max_price, min_price, mean_delta, max_delta,
        count_add / lengths, count_view / lengths, count_remove / np.maximum(count_add, 1.0),
        (cart_units > 0).astype(np.float32),
        np.log1p(np.expm1(np.maximum(nums[:, -1, 3], 0.0)) / lengths),
        switches / np.maximum(lengths - 1, 1),
    )).astype(np.float32)
    if flat.shape[1] != FLAT_FEATURES:
        raise AssertionError(f"expected {FLAT_FEATURES} flat features, got {flat.shape[1]}")
    return flat


def _torch_flat_features(x_cat, x_num, valid, length):
    dtype = x_num.dtype
    width = x_cat.size(1)
    valid2 = valid.squeeze(-1).to(dtype)
    last = x_num[:, -1]
    last_cat = x_cat[:, -1, :].to(dtype)
    has_prev = (length > 1).to(dtype).unsqueeze(-1)
    prev_num = x_num[:, -2] * has_prev
    prev_cat = x_cat[:, -2, :2].to(dtype) * has_prev
    first_pos = (width - length.clamp(min=1, max=width)).unsqueeze(1)
    first_cat = x_cat.gather(1, first_pos.unsqueeze(-1).expand(-1, -1, FT.N_CAT)).squeeze(1)[:, :2].to(dtype)
    first_price = x_num.gather(
        1, first_pos.unsqueeze(-1).expand(-1, -1, FT.N_NUM)
    ).squeeze(1)[:, 0:1]
    denom = valid2.sum(dim=1, keepdim=True).clamp(min=1.0)
    price = x_num[:, :, 0]
    delta = x_num[:, :, 2]
    masked_price = torch.where(valid2 > 0, price, torch.full_like(price, -1e4))
    masked_delta = torch.where(valid2 > 0, delta, torch.full_like(delta, -1e4))
    mean_price = (price * valid2).sum(dim=1, keepdim=True) / denom
    max_price = masked_price.max(dim=1, keepdim=True)[0]
    min_price = torch.where(valid2 > 0, price, torch.full_like(price, 1e4)).min(dim=1, keepdim=True)[0]
    mean_delta = (delta * valid2).sum(dim=1, keepdim=True) / denom
    max_delta = masked_delta.max(dim=1, keepdim=True)[0]
    count_add = torch.exp(torch.clamp(last[:, 6:7], min=0.0)) - 1.0
    count_view = torch.exp(torch.clamp(last[:, 5:6], min=0.0)) - 1.0
    count_remove = torch.exp(torch.clamp(last[:, 7:8], min=0.0)) - 1.0
    signed_log_cart = last[:, 8:9]
    cart_units = torch.sign(signed_log_cart) * (torch.exp(torch.abs(signed_log_cart)) - 1.0)
    cat_seq = x_cat[:, :, 1].to(dtype)
    pair_valid = valid2[:, 1:] * valid2[:, :-1]
    switches = ((cat_seq[:, 1:] != cat_seq[:, :-1]).to(dtype) * pair_valid).sum(dim=1, keepdim=True)
    k = length.to(dtype).unsqueeze(-1).clamp(min=1.0)
    return torch.cat((
        last_cat, prev_cat, first_cat, last,
        prev_num[:, 0:1], prev_num[:, 2:3], first_price,
        mean_price, max_price, min_price, mean_delta, max_delta,
        count_add / k, count_view / k, count_remove / count_add.clamp(min=1.0),
        (cart_units > 0).to(dtype),
        torch.log1p((torch.exp(torch.clamp(last[:, 3:4], min=0.0)) - 1.0) / k),
        switches / (k - 1.0).clamp(min=1.0),
    ), dim=1)


class SequenceModelBase(nn.Module):
    """Common embedding, masked pooling, calibration buffers and output contract."""
    def __init__(self, spec, d_model=32, dropout=0.1, head_extra=0):
        super().__init__()
        self.max_len = int(spec["max_len"])
        self.d_model = int(d_model)
        card = spec["cardinality"]
        self.step_emb = StepEmbedding(card, d_model=d_model, dropout=dropout)
        self.shop_emb = nn.Embedding(int(card["shop"]), 8, padding_idx=FT.PAD)
        self.register_buffer("pos_idx", torch.arange(self.max_len, dtype=torch.int64).unsqueeze(0))
        self.head = nn.Sequential(
            nn.Linear(d_model * 3 + 8 + 1 + FT.N_NUM * 2 + head_extra, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.register_buffer("calib_a", torch.ones(int(card["shop"]), dtype=torch.float32))
        self.register_buffer("calib_b", torch.zeros(int(card["shop"]), dtype=torch.float32))

    def _mask(self, x_num, length):
        start = (self.max_len - length.clamp(min=1, max=self.max_len)).unsqueeze(1)
        return (self.pos_idx >= start).to(x_num.dtype).unsqueeze(-1)

    def _score(self, h, x_num, length, shop, valid, extra=None):
        denom = valid.sum(dim=1).clamp(min=1.0)
        h_last = h[:, -1, :]
        h_mean = h.sum(dim=1) / denom
        h_max = (h + (1.0 - valid) * -1e4).max(dim=1)[0]
        x_num_valid = x_num * valid
        num_mean = x_num_valid.sum(dim=1) / denom
        parts = [h_last, h_mean, h_max, self.shop_emb(shop),
                 torch.log1p(length.to(x_num.dtype)).unsqueeze(-1),
                 x_num[:, -1, :], num_mean]
        if extra is not None:
            parts.append(extra)
        raw_logit = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return self.calib_a[shop] * raw_logit + self.calib_b[shop]

    def forward(self, x_cat, x_num, length, shop):
        return torch.sigmoid(self.forward_logits(x_cat, x_num, length, shop))

    @torch.no_grad()
    def set_calibration(self, shop_params):
        for shop_idx, (a, b) in shop_params.items():
            self.calib_a[int(shop_idx)] = float(a)
            self.calib_b[int(shop_idx)] = float(b)


class BiLSTMAttention(SequenceModelBase):
    def __init__(self, spec, d_model=32, dropout=0.1):
        if d_model % 2:
            raise ValueError("BiLSTM d_model must be even")
        super().__init__(spec, d_model, dropout, head_extra=d_model)
        half = d_model // 2
        self.forward_cell = nn.LSTMCell(d_model, half)
        self.backward_cell = nn.LSTMCell(d_model, half)
        self.merge = nn.Linear(d_model, d_model)
        self.attn = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward_logits(self, x_cat, x_num, length, shop):
        valid = self._mask(x_num, length)
        emb = self.step_emb(x_cat, x_num, valid)
        batch, width, _ = emb.shape
        h_f = emb.new_zeros((batch, self.d_model // 2))
        c_f = torch.zeros_like(h_f)
        out_f = []
        for t in range(width):
            h_new, c_new = self.forward_cell(emb[:, t], (h_f, c_f))
            active = valid[:, t]
            h_f = active * h_new + (1.0 - active) * h_f
            c_f = active * c_new + (1.0 - active) * c_f
            out_f.append(h_f)
        h_b = emb.new_zeros((batch, self.d_model // 2))
        c_b = torch.zeros_like(h_b)
        out_b = [None] * width
        for t in range(width - 1, -1, -1):
            h_new, c_new = self.backward_cell(emb[:, t], (h_b, c_b))
            active = valid[:, t]
            h_b = active * h_new + (1.0 - active) * h_b
            c_b = active * c_new + (1.0 - active) * c_b
            out_b[t] = h_b
        h = self.merge(torch.cat((torch.stack(out_f, dim=1), torch.stack(out_b, dim=1)), dim=-1))
        h = h * valid
        attn_score = self.attn(h).squeeze(-1) + (1.0 - valid.squeeze(-1)) * -1e4
        context = torch.sum(torch.softmax(attn_score, dim=1).unsqueeze(-1) * h, dim=1)
        return self._score(h, x_num, length, shop, valid, extra=context)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def _causal(self, conv, x):
        return conv(F.pad(x, (2 * self.dilation, 0)))

    def forward(self, x, mask):
        h = self._causal(self.conv1, x).transpose(1, 2)
        h = self.dropout(F.gelu(self.norm1(h))).transpose(1, 2)
        h = self._causal(self.conv2, h).transpose(1, 2)
        h = self.dropout(F.gelu(self.norm2(h))).transpose(1, 2)
        return (x + h) * mask


class TemporalConvNet(SequenceModelBase):
    def __init__(self, spec, d_model=32, dropout=0.1, n_layers=3):
        super().__init__(spec, d_model, dropout)
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(d_model, 2 ** i, dropout) for i in range(n_layers)
        ])

    def forward_logits(self, x_cat, x_num, length, shop):
        valid = self._mask(x_num, length)
        h = self.step_emb(x_cat, x_num, valid).transpose(1, 2)
        mask = valid.transpose(1, 2)
        for block in self.blocks:
            h = block(h, mask)
        return self._score(h.transpose(1, 2), x_num, length, shop, valid)


class MixerBlock(nn.Module):
    def __init__(self, max_len, d_model, dropout):
        super().__init__()
        self.norm_tokens = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(max_len, max_len * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max_len * 2, max_len), nn.Dropout(dropout),
        )
        self.norm_channels = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, valid):
        token = self.token_mlp(self.norm_tokens(x).transpose(1, 2)).transpose(1, 2)
        x = (x + token) * valid
        return (x + self.channel_mlp(self.norm_channels(x))) * valid


class MLPMixer(SequenceModelBase):
    def __init__(self, spec, d_model=32, dropout=0.1, n_layers=2):
        super().__init__(spec, d_model, dropout)
        self.blocks = nn.ModuleList([
            MixerBlock(self.max_len, d_model, dropout) for _ in range(n_layers)
        ])

    def forward_logits(self, x_cat, x_num, length, shop):
        valid = self._mask(x_num, length)
        h = self.step_emb(x_cat, x_num, valid)
        for block in self.blocks:
            h = block(h, valid)
        return self._score(h, x_num, length, shop, valid)


class HybridGRUWideDeep(SequenceModelBase):
    def __init__(self, spec, d_model=32, dropout=0.1, n_layers=1):
        super().__init__(spec, d_model, dropout, head_extra=32)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.wide = nn.Sequential(nn.Linear(FLAT_FEATURES, 32), nn.LayerNorm(32), nn.GELU())

    def forward_logits(self, x_cat, x_num, length, shop):
        valid = self._mask(x_num, length)
        h = self.step_emb(x_cat, x_num, valid)
        h, _ = self.gru(h)
        h = h * valid
        flat = _torch_flat_features(x_cat, x_num, valid, length)
        return self._score(h, x_num, length, shop, valid, extra=self.wide(flat))


class TCN(TemporalConvNet):
    """Short alias retained for command-line model naming."""


MODEL_SPECS = {
    "gru": "GRU",
    "bilstm_attention": "Bi-LSTM + Additive Attention",
    "tcn": "Causal TCN",
    "transformer": "Causal Transformer",
    "hybrid": "Hybrid GRU + Wide & Deep",
    "mixer": "MLP-Mixer",
}


def build_model(name, spec, d_model=32, dropout=0.1):
    if name == "gru":
        return IntentModel(spec, arch="gru", d_model=d_model, n_layers=1, dropout=dropout)
    if name == "transformer":
        return IntentModel(spec, arch="transformer", d_model=d_model, n_layers=1, n_heads=4,
                           dropout=dropout)
    constructors = {
        "bilstm_attention": BiLSTMAttention,
        "tcn": TemporalConvNet,
        "hybrid": HybridGRUWideDeep,
        "mixer": MLPMixer,
    }
    if name not in constructors:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_SPECS)}")
    return constructors[name](spec, d_model=d_model, dropout=dropout)
