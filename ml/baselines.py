"""Week 1 baselines for the purchase-intent prediction task: Markov chain + Gradient Boosted Trees (GBM),
with AUC and PR-AUC tables broken down by prefix length k.

Run on the prefix dataset produced by ml/prefixes.py:
  python ml/baselines.py --prefix data/prefix --out data/prefix/_baselines.json

Models (trained on the TRAIN split only; evaluated on VAL and TEST across all prefixes k = 1..n_steps):
1. Markov (1st & 2nd order + backoff):
   - Sequence of discrete event tokens s_t = (type_idx, cat_idx, has_cart) for t = 1..k, plus length bucket.
   - Smoothed empirical conversion rate P(purchase | suffix of length m, has_cart, len_bucket) with hierarchical
     Dirichlet/Laplace backoff from 3-gram -> 2-gram -> 1-gram -> shop prior:
         p_m = (pos_m + alpha * p_{m-1}) / (n_m + alpha)
   - Also reports pure 1st-order event-type Markov chain (transition log-odds + terminal state) as a reference,
     and the hierarchical Markov chain as the main Markov baseline.
2. GBM (sklearn HistGradientBoostingClassifier, LightGBM-style histogram GBDT):
   - Uses a compact tabular summary of the prefix at step k built directly from the shared STEP_FIELDS /
     NUM_FEATURES of ml/features.py:
       current step (type_idx, cat_idx, brand_idx, 10 transformed numerical features),
       previous step (prev_type_idx, prev_cat_idx, prev_log_price, prev_log_dt_s),
       first step (first_type_idx, first_cat_idx, first_log_price),
       prefix aggregates over the window (mean/max/min log_price, mean/max log_dt_prev_s, view/add/remove ratios,
       mean_s_per_event, n_unique_cats_w, n_switches_w), plus shop_idx.
   - Early stopping on VAL (or internal validation split), isotonic/sigmoid probability calibration option on VAL.

Outputs:
  - JSON report (default: data/prefix/_baselines.json) with overall and per-prefix-length (1, 2, 3-5, 6-10, 11-20,
    21-50, >50) ROC-AUC, PR-AUC (Average Precision), Brier score and LogLoss per shop and combined.
  - Markdown table printed to stdout ready to paste into Chapter 5b of the report.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from ml import features as FT  # noqa: E402
from ml.prefixes import K_BUCKETS  # noqa: E402

BUCKET_NAMES = [f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f">{lo - 1}") for lo, hi in K_BUCKETS]


def k_to_bucket(k):
    for (lo, hi), name in zip(K_BUCKETS, BUCKET_NAMES):
        if k >= lo and (hi is None or k <= hi):
            return name
    return BUCKET_NAMES[-1]


def safe_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    n = int(len(y_true))
    pos = int(y_true.sum())
    if n == 0 or pos == 0 or pos == n:
        return {"samples": n, "positives": pos, "pos_rate": round(pos / max(n, 1), 4),
                "roc_auc": None, "pr_auc": None, "brier": None, "logloss": None}
    return {
        "samples": n,
        "positives": pos,
        "pos_rate": round(pos / n, 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "logloss": round(float(log_loss(y_true, y_prob)), 4),
    }


# ---------------------------------------------------------------- dataset expansion (sessions -> prefix samples)
TABULAR_FEATURE_NAMES = (
    # categorical (declare as categorical in HistGBR)
    "shop_idx", "type_idx", "cat_idx", "brand_idx", "prev_type_idx", "prev_cat_idx", "first_type_idx", "first_cat_idx",
    # 10 shared numerical features at step k (FT.NUM_FEATURES)
    "log_price", "log_qty", "log_dt_prev_s", "log_t_first_s", "same_prod",
    "log_n_view", "log_n_add", "log_n_remove", "slog_cart_units", "log_pos",
    # lag & summary features over prefix 1..k
    "prev_log_price", "prev_log_dt_s", "first_log_price",
    "cum_log_price_mean", "cum_log_price_max", "cum_log_price_min",
    "cum_log_dt_mean", "cum_log_dt_max",
    "add_per_step", "view_per_step", "remove_per_add", "has_cart",
    "sec_per_step", "cat_switch_rate",
)
CATEGORICAL_COLS = list(range(8))


def load_prefix_samples(prefix_dir, shop_filter=None, max_sessions_per_split=200000, max_k_per_session=15, seed=42):
    """Read data/prefix/sessions per (shop_id, split) partition and expand sessions into prefix samples (session, k).
    Caps each (shop, split) at max_sessions_per_split (all positives kept + random negatives, or uniform subsample)
    so 4.8M REES46 sessions fit comfortably in laptop RAM without starving smaller shops.
    """
    base = os.path.join(prefix_dir, "sessions")
    shop_dirs = sorted(d for d in os.listdir(base) if d.startswith("shop_id="))
    rng = np.random.default_rng(seed)

    splits_parts = defaultdict(list)
    for sd in shop_dirs:
        shop_name = sd.split("=", 1)[1]
        if shop_filter and shop_name not in shop_filter:
            continue
        s_path = os.path.join(base, sd)
        for spd in sorted(d for d in os.listdir(s_path) if d.startswith("split=")):
            split_name = spd.split("=", 1)[1]
            df = pq.read_table(os.path.join(s_path, spd)).to_pandas()
            df["shop_id"] = shop_name
            df["split"] = split_name
            if max_sessions_per_split and len(df) > max_sessions_per_split:
                idx = rng.choice(len(df), size=max_sessions_per_split, replace=False)
                df = df.iloc[np.sort(idx)].reset_index(drop=True)
            splits_parts[split_name].append(df)

    import pandas as pd
    splits = {}
    for split, parts in splits_parts.items():
        sdf = pd.concat(parts, ignore_index=True)

        # Count total prefix samples to pre-allocate numpy arrays
        steps_col = sdf["steps"].tolist()
        labels_col = sdf["label"].to_numpy(dtype=np.int32)
        shop_idx_col = sdf["shop_idx"].to_numpy(dtype=np.int32)
        shop_id_col = sdf["shop_id"].astype(str).tolist()

        ks_per_sess = []
        total = 0
        for st in steps_col:
            n = len(st)
            if max_k_per_session and n > max_k_per_session:
                # always include k=1,2,last and a deterministic subsample of interior steps
                cand = sorted(set([1, min(2, n), n] + rng.choice(range(1, n + 1), size=max_k_per_session,
                                                                 replace=False).tolist()))
                ks_per_sess.append(cand)
                total += len(cand)
            else:
                ks_per_sess.append(None)
                total += n

        X = np.zeros((total, len(TABULAR_FEATURE_NAMES)), dtype=np.float32)
        y = np.zeros(total, dtype=np.int32)
        k_arr = np.zeros(total, dtype=np.int32)
        shops = np.empty(total, dtype=object)
        # Markov discrete keys per sample: (shop_idx, t_last, t_prev, t_prev2, c_last, has_cart, k_bucket_idx, trans_ll_key)
        m_shop = np.zeros(total, dtype=np.int32)
        m_t1 = np.zeros(total, dtype=np.int32)
        m_t2 = np.zeros(total, dtype=np.int32)
        m_t3 = np.zeros(total, dtype=np.int32)
        m_cat = np.zeros(total, dtype=np.int32)
        m_cart = np.zeros(total, dtype=np.int32)
        m_kb = np.zeros(total, dtype=np.int32)
        # Transition counts inside prefix (from type a -> type b, types 3..6 mapped to 0..3)
        m_trans = np.zeros((total, 16), dtype=np.int16)

        row = 0
        for s_i, st in enumerate(steps_col):
            n = len(st)
            if n == 0:
                continue
            lbl = labels_col[s_i]
            sidx = shop_idx_col[s_i]
            sname = shop_id_col[s_i]
            ks_sel = ks_per_sess[s_i]
            ks_set = set(ks_sel) if ks_sel is not None else None

            # Running accumulators along the session prefix k = 1..n
            first = st[0]
            first_type = int(first["type_idx"])
            first_cat = int(first["cat_idx"])
            first_lp = math.log1p(max(float(first["price"]), 0.0))

            prev_type, prev_cat, prev_lp, prev_ldt = 0, 0, 0.0, 0.0
            prev2_type = 0
            lp_sum, lp_max, lp_min = 0.0, -1e9, 1e9
            ldt_sum, ldt_max = 0.0, 0.0
            cat_switches = 0
            trans_counts = np.zeros(16, dtype=np.int16)

            for idx_zero, step in enumerate(st):
                k = idx_zero + 1
                t_idx = int(step["type_idx"])
                c_idx = int(step["cat_idx"])
                b_idx = int(step["brand_idx"])
                price = float(step["price"])
                qty = int(step["qty"])
                dt_ms = int(step["dt_prev_ms"])
                tf_ms = int(step["t_first_ms"])
                same = int(step["same_prod"])
                n_view = int(step["n_view"])
                n_add = int(step["n_add"])
                n_rem = int(step["n_remove"])
                cart_u = int(step["cart_units"])
                pos = int(step["pos"])

                lp = math.log1p(max(price, 0.0))
                lq = math.log1p(max(qty, 0))
                ldt = math.log1p(dt_ms / 1000.0)
                ltf = math.log1p(tf_ms / 1000.0)
                lnv = math.log1p(n_view)
                lna = math.log1p(n_add)
                lnr = math.log1p(n_rem)
                scu = math.copysign(math.log1p(abs(cart_u)), cart_u) if cart_u else 0.0
                lpos = math.log1p(pos)

                lp_sum += lp
                if lp > lp_max:
                    lp_max = lp
                if lp < lp_min:
                    lp_min = lp
                ldt_sum += ldt
                if ldt > ldt_max:
                    ldt_max = ldt
                if k > 1 and c_idx != prev_cat:
                    cat_switches += 1
                if k > 1:
                    a_t = min(max(prev_type - FT.FIRST_IDX, 0), 3)
                    b_t = min(max(t_idx - FT.FIRST_IDX, 0), 3)
                    trans_counts[a_t * 4 + b_t] += 1

                if ks_set is None or k in ks_set:
                    has_cart = 1 if cart_u > 0 else 0
                    X[row] = (
                        sidx, t_idx, c_idx, b_idx, prev_type, prev_cat, first_type, first_cat,
                        lp, lq, ldt, ltf, float(same), lnv, lna, lnr, scu, lpos,
                        prev_lp, prev_ldt, first_lp,
                        lp_sum / k, lp_max, lp_min,
                        ldt_sum / k, ldt_max,
                        n_add / k, n_view / k, n_rem / max(n_add, 1), float(has_cart),
                        math.log1p((tf_ms / 1000.0) / k), cat_switches / max(k - 1, 1),
                    )
                    y[row] = lbl
                    k_arr[row] = k
                    shops[row] = sname
                    m_shop[row] = sidx
                    m_t1[row] = t_idx
                    m_t2[row] = prev_type
                    m_t3[row] = prev2_type
                    m_cat[row] = c_idx
                    m_cart[row] = has_cart
                    m_kb[row] = 0 if k == 1 else (1 if k == 2 else (2 if k <= 5 else (3 if k <= 10 else 4)))
                    m_trans[row] = trans_counts
                    row += 1

                prev2_type = prev_type
                prev_type, prev_cat, prev_lp, prev_ldt = t_idx, c_idx, lp, ldt

        splits[split] = {
            "X": X[:row], "y": y[:row], "k": k_arr[:row], "shop_id": shops[:row],
            "markov": (m_shop[:row], m_t1[:row], m_t2[:row], m_t3[:row], m_cat[:row], m_cart[:row], m_kb[:row],
                       m_trans[:row]),
            "n_sessions": int(len(sdf)),
        }
    return splits


# ---------------------------------------------------------------- Baseline 1: Markov Chain
class MarkovPurchaseBaseline:
    """1st-order Markov chain baseline over event_type transitions (4 states: page_view, add_to_cart,
    remove_from_cart, checkout):
      - Class-conditional transition matrices P(e_t | e_{t-1}, Y=c) and initial/terminal state P(e_k | Y=c)
        with Laplace smoothing, combined with smoothed empirical P(Y=1 | shop, e_k, e_{k-1}).
    """
    def __init__(self, alpha=50.0, mix_generative=0.50):
        self.alpha = float(alpha)
        self.mix = float(mix_generative)
        self.global_prior = 0.1
        self.tables = {}
        self.trans_llr = {}
        self.init_llr = {}
        self.shop_logit = {}

    @staticmethod
    def _fit_table(keys, y):
        n = defaultdict(int)
        pos = defaultdict(int)
        for k, yi in zip(keys, y):
            n[k] += 1
            pos[k] += int(yi)
        return dict(n), dict(pos)

    def fit(self, markov_tuple, y):
        s, t1, t2, _t3, _cat, _cart, _kb, trans = markov_tuple
        y = np.asarray(y, dtype=np.int32)
        self.global_prior = float(y.mean()) if len(y) else 0.1

        # 1st-order Markov transition keys: shop prior -> P(Y | shop, e_k) -> P(Y | shop, e_k, e_{k-1})
        k0 = list(zip(s.tolist()))
        k1 = list(zip(s.tolist(), t1.tolist()))
        k2 = list(zip(s.tolist(), t1.tolist(), t2.tolist()))
        for lvl, keys in enumerate([k0, k1, k2]):
            self.tables[lvl] = self._fit_table(keys, y)

        # Generative class-conditional 1st-order Markov transition matrices P(b | a, Y=c) per shop
        for sidx in np.unique(s):
            mask = (s == sidx)
            y_s = y[mask]
            tr_s = trans[mask]
            t1_s = t1[mask]
            pos_m = (y_s == 1)
            neg_m = ~pos_m
            p_prior = (pos_m.sum() + 1.0) / (len(y_s) + 2.0)
            self.shop_logit[int(sidx)] = math.log(p_prior / (1.0 - p_prior))

            tr_pos = tr_s[pos_m].astype(np.float64).sum(axis=0).reshape(4, 4) + 1.0
            tr_neg = tr_s[neg_m].astype(np.float64).sum(axis=0).reshape(4, 4) + 1.0
            tr_pos /= tr_pos.sum(axis=1, keepdims=True)
            tr_neg /= tr_neg.sum(axis=1, keepdims=True)
            self.trans_llr[int(sidx)] = np.log(tr_pos / tr_neg).reshape(16)

            init_llr = {}
            for ti in range(FT.FIRST_IDX, FT.FIRST_IDX + 4):
                m_t = (t1_s == ti)
                p1 = (np.sum(m_t & pos_m) + 1.0) / (pos_m.sum() + 4.0)
                p0 = (np.sum(m_t & neg_m) + 1.0) / (neg_m.sum() + 4.0)
                init_llr[ti] = math.log(p1 / p0)
            self.init_llr[int(sidx)] = init_llr
        return self

    def predict_proba(self, markov_tuple):
        s, t1, t2, _t3, _cat, _cart, _kb, trans = markov_tuple
        n_samples = len(s)
        out = np.empty(n_samples, dtype=np.float64)
        a = self.alpha
        t0_n, t0_p = self.tables[0]
        t1_n, t1_p = self.tables[1]
        t2_n, t2_p = self.tables[2]

        s_l, t1_l, t2_l = s.tolist(), t1.tolist(), t2.tolist()
        for i in range(n_samples):
            si, e1, e2 = s_l[i], t1_l[i], t2_l[i]
            p = self.global_prior
            for (tn, tp), key in (
                ((t0_n, t0_p), (si,)),
                ((t1_n, t1_p), (si, e1)),
                ((t2_n, t2_p), (si, e1, e2)),
            ):
                cnt = tn.get(key, 0)
                if cnt:
                    p = (tp[key] + a * p) / (cnt + a)

            if self.mix > 0 and si in self.trans_llr:
                llr = self.shop_logit[si] + self.init_llr[si].get(e1, 0.0)
                step_cnt = float(trans[i].sum())
                if step_cnt > 0:
                    llr += float(np.dot(trans[i], self.trans_llr[si])) / math.sqrt(step_cnt)
                p_gen = 1.0 / (1.0 + math.exp(-max(min(llr, 20.0), -20.0)))
                p = (1.0 - self.mix) * p + self.mix * p_gen
            out[i] = p
        return out


# ---------------------------------------------------------------- Baseline 2: GBM (HistGradientBoosting)
class GBMPurchaseBaseline:
    """Histogram GBDT (LightGBM-style) with 5-fold out-of-fold smoothed target encoding for high-cardinality
    category/brand + Markov transition probability feature + all numerical prefix features.
    """
    MAX_CAT_BINS = 250

    def __init__(self, seed=42, alpha=30.0):
        self.seed = seed
        self.alpha = float(alpha)
        self.prior = 0.1
        self.cat_te = {}
        self.brand_te = {}
        self.seq_te = {}
        self.model = None
        self.n_iter_ = 0

    @staticmethod
    def _fit_te(vals, y, prior, alpha):
        vals = vals.astype(np.int32)
        max_v = int(vals.max()) + 1 if len(vals) else 1
        cnt = np.bincount(vals, minlength=max_v).astype(np.float64)
        pos = np.bincount(vals, weights=y.astype(np.float64), minlength=max_v)
        te = (pos + alpha * prior) / np.maximum(cnt + alpha, 1e-9)
        return {i: float(te[i]) for i in np.nonzero(cnt)[0]}

    def _encode_col(self, vals, table):
        p = self.prior
        return np.fromiter((table.get(int(v), p) for v in vals), dtype=np.float32, count=len(vals))

    def _oof_encode_col(self, vals, y, rng, n_folds=5):
        vals = vals.astype(np.int32)
        out = np.empty(len(vals), dtype=np.float32)
        folds = rng.integers(0, n_folds, size=len(vals))
        for f in range(n_folds):
            tr_m = (folds != f)
            val_m = ~tr_m
            tbl = self._fit_te(vals[tr_m], y[tr_m], float(y[tr_m].mean()) if tr_m.any() else self.prior, self.alpha)
            out[val_m] = self._encode_col(vals[val_m], tbl)
        return out

    @staticmethod
    def _seq_keys(X):
        return (X[:, 0].astype(np.int32) * 10000 + X[:, 1].astype(np.int32) * 1000
                + X[:, 4].astype(np.int32) * 100 + X[:, 29].astype(np.int32) * 10)

    def _transform(self, X, y_train=None, rng=None):
        X_out = X.copy()
        c_cur = X[:, 2].astype(np.int32)
        b_cur = X[:, 3].astype(np.int32)
        c_prv = X[:, 5].astype(np.int32)
        c_fst = X[:, 7].astype(np.int32)
        s_key = self._seq_keys(X)
        if y_train is not None and rng is not None:
            te_cols = np.column_stack([
                self._oof_encode_col(c_cur, y_train, rng),
                self._oof_encode_col(b_cur, y_train, rng),
                self._oof_encode_col(c_prv, y_train, rng),
                self._oof_encode_col(c_fst, y_train, rng),
                self._oof_encode_col(s_key, y_train, rng),
            ])
        else:
            te_cols = np.column_stack([
                self._encode_col(c_cur, self.cat_te),
                self._encode_col(b_cur, self.brand_te),
                self._encode_col(c_prv, self.cat_te),
                self._encode_col(c_fst, self.cat_te),
                self._encode_col(s_key, self.seq_te),
            ])
        for col_i in CATEGORICAL_COLS:
            col = X_out[:, col_i]
            X_out[:, col_i] = np.where(col > self.MAX_CAT_BINS, float(FT.OOV_IDX), col)
        return np.hstack([X_out, te_cols])

    def fit(self, X_train, y_train):
        y_train = np.asarray(y_train, dtype=np.int32)
        self.prior = float(y_train.mean()) if len(y_train) else 0.1
        self.cat_te = self._fit_te(X_train[:, 2], y_train, self.prior, self.alpha)
        self.brand_te = self._fit_te(X_train[:, 3], y_train, self.prior, self.alpha)
        self.seq_te = self._fit_te(self._seq_keys(X_train), y_train, self.prior, self.alpha)

        rng = np.random.default_rng(self.seed)
        X_tr = self._transform(X_train, y_train=y_train, rng=rng)
        # Shuffle rows before HistGradientBoostingClassifier so its internal 10% validation split has both shops
        perm = rng.permutation(len(y_train))
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=63,
            min_samples_leaf=100,
            l2_regularization=2.0,
            categorical_features=CATEGORICAL_COLS,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            scoring="roc_auc",
            random_state=self.seed,
        )
        self.model.fit(X_tr[perm], y_train[perm])
        self.n_iter_ = int(self.model.n_iter_)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._transform(X))


def fit_gbm(X_train, y_train, seed=42):
    return GBMPurchaseBaseline(seed=seed).fit(X_train, y_train)


# ---------------------------------------------------------------- Evaluation & tables
def evaluate_predictions(y_true, y_prob, k_arr, shop_arr):
    buckets = np.array([k_to_bucket(int(k)) for k in k_arr], dtype=object)
    res = {"overall": safe_metrics(y_true, y_prob), "by_prefix_len": {}, "by_shop": {}}
    for b in BUCKET_NAMES:
        m = (buckets == b)
        if m.any():
            res["by_prefix_len"][b] = safe_metrics(y_true[m], y_prob[m])
    for shop in sorted(set(shop_arr.tolist())):
        ms = (shop_arr == shop)
        res["by_shop"][shop] = {"overall": safe_metrics(y_true[ms], y_prob[ms]), "by_prefix_len": {}}
        for b in BUCKET_NAMES:
            msb = ms & (buckets == b)
            if msb.any():
                res["by_shop"][shop]["by_prefix_len"][b] = safe_metrics(y_true[msb], y_prob[msb])
    return res


def format_markdown_tables(report):
    lines = ["## Bảng kết quả Baseline (Markov vs GBM) theo độ dài prefix `k`", ""]
    for split in ("val", "test"):
        if split not in report["results"]:
            continue
        r_m = report["results"][split]["markov"]
        r_g = report["results"][split]["gbm"]
        lines.append(f"### Tập `{split.upper()}` — Gộp cả 2 shop (Tổng: {r_g['overall']['samples']:,} mẫu prefix, "
                     f"tỉ lệ mua = {r_g['overall']['pos_rate'] * 100:.2f}%)")
        lines.append("")
        lines.append("| Độ dài prefix `k` | Số mẫu | Tỉ lệ mua | Markov ROC-AUC | GBM ROC-AUC | Δ AUC (GBM - Markov) | Markov PR-AUC | GBM PR-AUC | Δ PR-AUC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for b in BUCKET_NAMES:
            if b not in r_g["by_prefix_len"]:
                continue
            mm = r_m["by_prefix_len"][b]
            mg = r_g["by_prefix_len"][b]
            da = (mg["roc_auc"] - mm["roc_auc"]) if (mg["roc_auc"] is not None and mm["roc_auc"] is not None) else 0.0
            dp = (mg["pr_auc"] - mm["pr_auc"]) if (mg["pr_auc"] is not None and mm["pr_auc"] is not None) else 0.0
            lines.append(f"| `{b}` | {mg['samples']:,} | {mg['pos_rate'] * 100:.2f}% | "
                         f"{mm['roc_auc']:.4f} | **{mg['roc_auc']:.4f}** | {da:+.4f} | "
                         f"{mm['pr_auc']:.4f} | **{mg['pr_auc']:.4f}** | {dp:+.4f} |")
        om, og = r_m["overall"], r_g["overall"]
        da = og["roc_auc"] - om["roc_auc"]
        dp = og["pr_auc"] - om["pr_auc"]
        lines.append(f"| **Toàn bộ (`All k`)** | **{og['samples']:,}** | **{og['pos_rate'] * 100:.2f}%** | "
                     f"**{om['roc_auc']:.4f}** | **{og['roc_auc']:.4f}** | **{da:+.4f}** | "
                     f"**{om['pr_auc']:.4f}** | **{og['pr_auc']:.4f}** | **{dp:+.4f}** |")
        lines.append("")
        for shop in sorted(r_g["by_shop"]):
            sm = r_m["by_shop"][shop]
            sg = r_g["by_shop"][shop]
            lines.append(f"#### Shop `{shop}` ({split.upper()} — {sg['overall']['samples']:,} mẫu, "
                         f"tỉ lệ mua = {sg['overall']['pos_rate'] * 100:.2f}%)")
            lines.append("")
            lines.append("| Độ dài prefix `k` | Số mẫu | Tỉ lệ mua | Markov ROC-AUC | GBM ROC-AUC | Δ AUC | Markov PR-AUC | GBM PR-AUC | Δ PR-AUC |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for b in BUCKET_NAMES:
                if b not in sg["by_prefix_len"] or sg["by_prefix_len"][b]["roc_auc"] is None:
                    continue
                mm = sm["by_prefix_len"][b]
                mg = sg["by_prefix_len"][b]
                da = mg["roc_auc"] - mm["roc_auc"]
                dp = mg["pr_auc"] - mm["pr_auc"]
                lines.append(f"| `{b}` | {mg['samples']:,} | {mg['pos_rate'] * 100:.2f}% | "
                             f"{mm['roc_auc']:.4f} | **{mg['roc_auc']:.4f}** | {da:+.4f} | "
                             f"{mm['pr_auc']:.4f} | **{mg['pr_auc']:.4f}** | {dp:+.4f} |")
            som, sog = sm["overall"], sg["overall"]
            lines.append(f"| **Toàn bộ (`All k`)** | **{sog['samples']:,}** | **{sog['pos_rate'] * 100:.2f}%** | "
                         f"**{som['roc_auc']:.4f}** | **{sog['roc_auc']:.4f}** | **{sog['roc_auc'] - som['roc_auc']:+.4f}** | "
                         f"**{som['pr_auc']:.4f}** | **{sog['pr_auc']:.4f}** | **{sog['pr_auc'] - som['pr_auc']:+.4f}** |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data/prefix")
    ap.add_argument("--out", default="data/prefix/_baselines.json")
    ap.add_argument("--shops", default="", help="comma-separated shop filter (default: all in data/prefix)")
    ap.add_argument("--max-sessions-per-split", type=int, default=150000,
                    help="max sessions per (shop, split) to expand in RAM (0 = all)")
    ap.add_argument("--max-k-per-session", type=int, default=15,
                    help="max prefix steps sampled per long session (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    shop_filter = [s for s in a.shops.split(",") if s] or None

    t0 = time.time()
    splits = load_prefix_samples(a.prefix, shop_filter=shop_filter,
                                 max_sessions_per_split=a.max_sessions_per_split or None,
                                 max_k_per_session=a.max_k_per_session or None, seed=a.seed)
    t_load = round(time.time() - t0, 2)

    train = splits["train"]
    t1 = time.time()
    markov = MarkovPurchaseBaseline(alpha=25.0, mix_generative=0.20).fit(train["markov"], train["y"])
    t_markov = round(time.time() - t1, 2)

    t2 = time.time()
    gbm = fit_gbm(train["X"], train["y"], seed=a.seed)
    t_gbm = round(time.time() - t2, 2)

    results = {}
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            continue
        sp = splits[split_name]
        p_markov = markov.predict_proba(sp["markov"])
        p_gbm = gbm.predict_proba(sp["X"])[:, 1]
        results[split_name] = {
            "n_sessions": sp["n_sessions"],
            "markov": evaluate_predictions(sp["y"], p_markov, sp["k"], sp["shop_id"]),
            "gbm": evaluate_predictions(sp["y"], p_gbm, sp["k"], sp["shop_id"]),
        }

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "prefix_dir": os.path.abspath(a.prefix),
        "timing_s": {"load_expand": t_load, "fit_markov": t_markov, "fit_gbm": t_gbm,
                     "gbm_n_iter": int(gbm.n_iter_)},
        "features": {"n_tabular_features": len(TABULAR_FEATURE_NAMES), "names": list(TABULAR_FEATURE_NAMES)},
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(format_markdown_tables(report))
    print(f"\nSaved JSON report to {a.out} (load={t_load}s, markov={t_markov}s, gbm={t_gbm}s, "
          f"gbm_trees={gbm.n_iter_})")


if __name__ == "__main__":
    main()
