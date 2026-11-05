"""Shared feature encoding of the purchase-intent model: ONE definition for training (batch) and serving (Q2).

Problem: after each event of a clickstream session, predict P(the session ends with a purchase). A prefix may
only use events with seq_no < cut_seq_no (first checkout/purchase, see common.canonical.build_sessions), and the
model sees at most the last MAX_LEN = 50 events of the prefix.

Two layers, so batch and stream give bit-identical tensors:
1. RAW STEP (exact values: ints + the source price double). One per event, computed in seq_no order with O(1)
   running state, so a step does not depend on which prefix it is looked at from:
       seq_no, type_idx, cat_idx, brand_idx,                      <- categorical (vocab indices)
       price, qty, dt_prev_ms, t_first_ms, same_prod,             <- event + previous event
       n_view, n_add, n_remove, cart_units, pos                   <- running counters
   - batch : Spark SQL window functions (ml/prefixes.py, scales to the 18M REES46 events without Python workers)
   - stream: SessionState.update() below, plain Python, used by Q2 inside applyInPandasWithState
   tests/test_features_parity.py checks both produce the same steps and the same tensors.
2. TENSOR: window_to_tensor() / session_tensors() turn the last <= MAX_LEN raw steps into model input. Both
   paths call this same Python code (math.log1p per scalar, no SIMD differences), so the result is bit-exact.

Time features use only differences between events (dt, time since the first event), never wall-clock time:
the simulator replays with a time-warp, so absolute time (hour, date) differs online; gaps do not as long as
the replay speed is 1x.

Categorical indices (every vocabulary): 0 = padding, 1 = NULL value, 2 = out-of-vocabulary, 3.. = vocab.
Category / brand vocabularies are keyed "<shop_id>|<value>" (ids are only meaningful inside a shop) and are
built from the TRAIN split only (feature_spec.json), so val/test values unseen in train map to OOV.
"""
import json
import math

from common.canonical import EVENT_TYPES

FEATURE_VERSION = 1
MAX_LEN = 50
PAD, NULL_IDX, OOV_IDX, FIRST_IDX = 0, 1, 2, 3
CUT_TYPES = ("checkout", "purchase")
TYPE_IDX = {t: FIRST_IDX + i for i, t in enumerate(EVENT_TYPES)}

CAT_FIELDS = ("type_idx", "cat_idx", "brand_idx")
RAW_NUM_FIELDS = ("price", "qty", "dt_prev_ms", "t_first_ms", "same_prod",
                  "n_view", "n_add", "n_remove", "cart_units", "pos")
STEP_FIELDS = ("seq_no",) + CAT_FIELDS + RAW_NUM_FIELDS
NUM_FEATURES = ("log_price", "log_qty", "log_dt_prev_s", "log_t_first_s", "same_prod",
                "log_n_view", "log_n_add", "log_n_remove", "slog_cart_units", "log_pos")
N_CAT, N_NUM = len(CAT_FIELDS), len(NUM_FEATURES)

# Columns the stream (Q2) and the batch job both feed in. ts_ms = unix_millis(event_ts) is computed in Spark,
# because Python collect()/pandas would convert timestamps to the machine timezone.
EVENT_INPUT_COLS = ("shop_id", "session_id", "seq_no", "event_type", "product_id", "category", "brand",
                    "price", "qty", "ts_ms")


# ---------------------------------------------------------------- feature spec (vocabularies)
def vocab_key(shop_id, value):
    return f"{shop_id}|{value}"


def make_spec(shops, categories, brands, max_len=MAX_LEN, source=None):
    """categories / brands: vocab keys ordered by frequency (most frequent first)."""
    return {
        "feature_version": FEATURE_VERSION,
        "max_len": max_len,
        "reserved": {"pad": PAD, "null": NULL_IDX, "oov": OOV_IDX},
        "event_types": {t: i for t, i in TYPE_IDX.items()},
        "cut_types": list(CUT_TYPES),
        "step_fields": list(STEP_FIELDS),
        "cat_fields": list(CAT_FIELDS),
        "num_features": list(NUM_FEATURES),
        "shops": {s: FIRST_IDX + i for i, s in enumerate(shops)},
        "category": {k: FIRST_IDX + i for i, k in enumerate(categories)},
        "brand": {k: FIRST_IDX + i for i, k in enumerate(brands)},
        "cardinality": {"type_idx": FIRST_IDX + len(TYPE_IDX), "cat_idx": FIRST_IDX + len(categories),
                        "brand_idx": FIRST_IDX + len(brands), "shop": FIRST_IDX + len(shops)},
        "source": source or {},
    }


def save_spec(spec, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)


def load_spec(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"feature_spec version {spec.get('feature_version')} != code version {FEATURE_VERSION}")
    return spec


def lookup(vocab, shop_id, value):
    if value is None:
        return NULL_IDX
    return vocab.get(vocab_key(shop_id, value), OOV_IDX)


def shop_index(spec, shop_id):
    return spec["shops"].get(shop_id, OOV_IDX)


# ---------------------------------------------------------------- layer 2: raw steps -> tensor (shared)
def _slog1p(x):
    return math.copysign(math.log1p(abs(x)), x) if x else 0.0


def step_num_vector(step):
    """Transform one raw step (tuple in STEP_FIELDS order) to the N_NUM model features (python floats)."""
    (_seq, _t, _c, _b, price, qty, dt_ms, tf_ms, same, n_view, n_add, n_remove, cart_units, pos) = step
    return (math.log1p(max(price, 0.0)), math.log1p(max(qty, 0)), math.log1p(dt_ms / 1000.0),
            math.log1p(tf_ms / 1000.0), float(same), math.log1p(n_view), math.log1p(n_add),
            math.log1p(n_remove), _slog1p(cart_units), math.log1p(pos))


def window_to_tensor(steps, shop_idx, max_len=MAX_LEN):
    """Model input for the prefix that ends with steps[-1]. Uses the last <= max_len steps, LEFT-padded
    (the most recent event is always at position max_len - 1; padding rows are 0 and length says how many
    rows are real). Returns dict of numpy arrays: x_cat int64 [L, N_CAT], x_num float32 [L, N_NUM],
    length int64 [], shop int64 []."""
    import numpy as np
    w = steps[-max_len:] if max_len else steps
    x_cat = np.zeros((max_len, N_CAT), dtype=np.int64)
    x_num = np.zeros((max_len, N_NUM), dtype=np.float32)
    n = len(w)
    if n:
        x_cat[max_len - n:] = [s[1:1 + N_CAT] for s in w]
        x_num[max_len - n:] = np.asarray([step_num_vector(s) for s in w], dtype=np.float64).astype(np.float32)
    return {"x_cat": x_cat, "x_num": x_num, "length": np.int64(n), "shop": np.int64(shop_idx)}


def session_tensors(steps, shop_idx, ks=None, max_len=MAX_LEN):
    """Training-side batch version: tensors for several prefix lengths k (k = number of steps used,
    1..len(steps)) of ONE session, stacked on axis 0. The transform is computed once per step and sliced,
    which gives exactly window_to_tensor(steps[:k]) for every k."""
    import numpy as np
    n = len(steps)
    ks = list(range(1, n + 1)) if ks is None else list(ks)
    cat_all = np.asarray([s[1:1 + N_CAT] for s in steps], dtype=np.int64).reshape(n, N_CAT)
    num_all = np.asarray([step_num_vector(s) for s in steps], dtype=np.float64).reshape(n, N_NUM).astype(np.float32)
    x_cat = np.zeros((len(ks), max_len, N_CAT), dtype=np.int64)
    x_num = np.zeros((len(ks), max_len, N_NUM), dtype=np.float32)
    length = np.zeros(len(ks), dtype=np.int64)
    for i, k in enumerate(ks):
        if not 1 <= k <= n:
            raise ValueError(f"prefix length {k} outside 1..{n}")
        lo = max(0, k - max_len)
        m = k - lo
        x_cat[i, max_len - m:] = cat_all[lo:k]
        x_num[i, max_len - m:] = num_all[lo:k]
        length[i] = m
    return {"x_cat": x_cat, "x_num": x_num, "length": length,
            "shop": np.full(len(ks), shop_idx, dtype=np.int64), "k": np.asarray(ks, dtype=np.int64)}


# ---------------------------------------------------------------- layer 1 (stream): raw steps, incremental
def _val(v):
    """None for None / NaN (pandas turns NULL numerics into NaN), else v."""
    if v is None:
        return None
    try:
        if v != v:   # NaN
            return None
    except (TypeError, ValueError):
        pass
    return v


def _get(ev, name):
    try:
        return _val(ev[name])
    except (KeyError, IndexError, TypeError):
        return _val(getattr(ev, name, None))


def _norm_event(ev):
    """Event (dict, Row, pandas row/namedtuple) -> plain-python tuple
    (event_type, product_id, category, brand, price, qty, ts_ms); numpy scalars become int/float/str."""
    def opt(name, typ):
        v = _get(ev, name)
        return None if v is None else typ(v)
    return (opt("event_type", str), opt("product_id", int), opt("category", str), opt("brand", str),
            opt("price", float), opt("qty", int), int(_get(ev, "ts_ms")))


class SessionState:
    """Per-session state of Q2 (one instance per (shop_id, session_id)), JSON-serialisable for
    applyInPandasWithState.

    Events are processed strictly in seq_no order: a micro-batch may hold late (out-of-order) and duplicated
    events (the simulator injects 3% late, 2% duplicates), so events with seq_no > next_seq wait in `pending`.
    Duplicates (seq_no already processed or already pending) are dropped. If more than max_pending events wait
    for a missing one, the gap is skipped (n_skipped) -- only then can the stream differ from batch.

    Flags: `cut` = a checkout/purchase was processed (no prediction from that event on, as in training);
    `closed` = set by Q2 when the session times out (later events are ignored).
    """
    MAX_PENDING = 64
    _FIELDS = ("shop_id", "next_seq", "pending", "cut", "cut_seq_no", "closed", "window", "prev_ts", "first_ts",
               "prev_product", "n_view", "n_add", "n_remove", "cart_units", "pos", "n_dup", "n_skipped",
               "n_after_close")

    def __init__(self, shop_id):
        self.shop_id = shop_id
        self.next_seq = 1
        self.pending = {}          # seq_no -> event tuple (EVENT_INPUT_COLS order, minus the keys)
        self.cut = False
        self.cut_seq_no = None
        self.closed = False
        self.window = []           # last MAX_LEN raw steps
        self.prev_ts = None
        self.first_ts = None
        self.prev_product = None
        self.n_view = self.n_add = self.n_remove = self.cart_units = self.pos = 0
        self.n_dup = self.n_skipped = self.n_after_close = 0

    # ---- serialisation (state store)
    def to_json(self):
        d = {k: getattr(self, k) for k in self._FIELDS}
        d["pending"] = [[s, list(e)] for s, e in sorted(self.pending.items())]
        d["window"] = [list(s) for s in self.window]
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        st = cls(d["shop_id"])
        for k in cls._FIELDS:
            setattr(st, k, d[k])
        st.pending = {int(seq): tuple(e) for seq, e in d["pending"]}
        st.window = [tuple(x) for x in d["window"]]
        return st

    # ---- processing
    def update(self, events, spec, emit="all", max_pending=None):
        """Feed the new events of this session (any order, may contain duplicates). Returns a list of
        (seq_no, k, tensor) for every newly processed pre-cut event (emit="all"), only the latest one
        (emit="last") or none (emit="none"); k = prefix length (number of events used so far)."""
        max_pending = self.MAX_PENDING if max_pending is None else max_pending
        for ev in events:
            seq = int(_get(ev, "seq_no"))
            if self.closed:
                self.n_after_close += 1
                continue
            if seq < self.next_seq or seq in self.pending:
                self.n_dup += 1
                continue
            self.pending[seq] = _norm_event(ev)
        sidx, max_len = shop_index(spec, self.shop_id), spec["max_len"]
        out, last = [], None
        while self.pending:
            if self.next_seq not in self.pending:
                if len(self.pending) <= max_pending:
                    break
                nxt = min(self.pending)
                self.n_skipped += nxt - self.next_seq
                self.next_seq = nxt
            seq = self.next_seq
            ev = self.pending.pop(seq)
            self.next_seq += 1
            if self.cut:
                continue
            if ev[0] in CUT_TYPES:
                self.cut, self.cut_seq_no = True, seq
                continue
            self.window.append(self._step(seq, ev, spec))
            del self.window[:-max_len]
            last = (seq, self.pos)
            if emit == "all":   # tensor now: the window moves on with the next event
                out.append((seq, self.pos, window_to_tensor(self.window, sidx, max_len)))
        if emit == "last" and last is not None:   # cut / skipped events after it do not change the window
            out.append((last[0], last[1], window_to_tensor(self.window, sidx, max_len)))
        return out

    def _step(self, seq, ev, spec):
        etype, product, category, brand, price, qty, ts = ev
        ts = int(ts)
        self.pos += 1
        dt = max(ts - self.prev_ts, 0) if self.prev_ts is not None else 0
        if self.first_ts is None:
            self.first_ts = ts
        tf = max(ts - self.first_ts, 0)
        product = int(product) if product is not None else None
        same = 1 if (product is not None and self.prev_product is not None and product == self.prev_product) else 0
        q = int(qty) if qty is not None else 0
        if etype == "page_view":
            self.n_view += 1
        elif etype == "add_to_cart":
            self.n_add += 1
            self.cart_units += q
        elif etype == "remove_from_cart":
            self.n_remove += 1
            self.cart_units -= q
        self.prev_ts, self.prev_product = ts, product
        return (seq, TYPE_IDX.get(etype, OOV_IDX), lookup(spec["category"], self.shop_id, category),
                lookup(spec["brand"], self.shop_id, brand), float(price) if price is not None else 0.0, q,
                dt, tf, same, self.n_view, self.n_add, self.n_remove, self.cart_units, self.pos)


# ---------------------------------------------------------------- Spark helpers (batch job and Q2)
def spark_event_input(df):
    """Select EVENT_INPUT_COLS from a canonical event DataFrame (batch Parquet or Q2 from_json)."""
    from pyspark.sql import functions as F
    return df.select("shop_id", "session_id", F.col("seq_no").cast("int").alias("seq_no"), "event_type",
                     "product_id", "category", "brand", "price", "qty", F.unix_millis("event_ts").alias("ts_ms"))
