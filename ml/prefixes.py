"""Build the prefix dataset (train/val/test) of the purchase-intent model from data/unified.

Run (stop the Docker stack first; ~18.5M events for the 2 clickstream shops):
  python ml/prefixes.py --unified data/unified --out data/prefix
Reuse an existing vocabulary (e.g. the one shipped with a model) instead of rebuilding it:
  python ml/prefixes.py --unified data/unified --out data/prefix --spec models/feature_spec.json

Output
  data/prefix/feature_spec.json                        vocabularies + feature layout (ml/features.py), from TRAIN only
  data/prefix/sessions/shop_id=<shop>/split=<split>/   one row per session that has a non-empty prefix:
      session_id, label, n_steps, cut_seq_no, session_start_ts, shop_idx,
      steps: array<struct<STEP_FIELDS>> sorted by seq_no (raw steps, see ml/features.py)
  data/prefix/_report.json                             counts, positive rates, prefix length table

Prefix = events with seq_no < cut_seq_no (first checkout/purchase) or the whole session if there is none.
One training/evaluation SAMPLE = (session, k) for k = 1..n_steps: the model input is
ml.features.session_tensors(steps, shop_idx, ks) (last <= 50 steps, left-padded), the label is the session label.
This mirrors Q2, which scores the session after every event until the cut.
Split = time split by session start (common.canonical.SPLITS); Olist has no browsing data and is not used.

Raw steps are computed with Spark SQL window functions (no Python workers: PySpark 3.5 cannot pickle
functions on Python 3.14, and Spark SQL is faster on 18M rows). ml.features.SessionState computes the same
steps incrementally for the stream; tests/test_features_parity.py checks they are identical.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspark.sql import functions as F, Window as W  # noqa: E402
from common.canonical import SPLITS, get_spark, read_table  # noqa: E402
from ml import features as FT  # noqa: E402

CLICK_SHOPS = tuple(SPLITS)          # shops with browsing data + a session split
INT_MAX = 2 ** 31 - 1
K_BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, None)]


def bucket_col(k):
    expr = None
    for lo, hi in K_BUCKETS:
        name = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f">{lo - 1}")
        cond = (k >= lo) & (k <= hi) if hi else (k >= lo)
        expr = F.when(cond, name) if expr is None else expr.when(cond, name)
    return expr


def prefix_events(events, sessions):
    """Canonical events -> EVENT_INPUT_COLS of the pre-cut events, joined with the session labels/split."""
    s = sessions.select("shop_id", "session_id", "split", F.col("label_purchase").cast("int").alias("label"),
                        "cut_seq_no", "session_start_ts")
    return (FT.spark_event_input(events).join(s, ["shop_id", "session_id"])
            .filter(F.col("seq_no") < F.coalesce(F.col("cut_seq_no"), F.lit(INT_MAX))))


def build_spec(pre, shops, min_count, max_categories, max_brands, source):
    """Vocabularies from the TRAIN split prefix events only (no look-ahead into val/test)."""
    train = pre.filter(F.col("split") == "train")

    def vocab(col, max_size):
        rows = (train.filter(F.col(col).isNotNull())
                .groupBy(F.concat_ws("|", "shop_id", col).alias("key")).count()
                .filter(F.col("count") >= min_count)
                .orderBy(F.desc("count"), "key").limit(max_size).collect())
        return [r.key for r in rows]

    return FT.make_spec(list(shops), vocab("category", max_categories), vocab("brand", max_brands),
                        source=source)


def build_steps(spark, pre, spec):
    """Pre-cut events -> one row per session with the raw steps (same values as SessionState._step)."""
    def vocab_df(name):
        # built from literals (spark.createDataFrame would need Python workers: not available on Python 3.14)
        items = list(spec[name].items()) or [("", 0)]
        m = F.map_from_arrays(F.array(*[F.lit(k) for k, _ in items]), F.array(*[F.lit(int(i)) for _, i in items]))
        return F.broadcast(spark.range(1).select(F.explode(m).alias(f"{name}_key", f"{name}_vidx")))

    type_map = F.create_map(*[x for t, i in spec["event_types"].items() for x in (F.lit(t), F.lit(i))])
    shop_map = F.create_map(*[x for s, i in spec["shops"].items() for x in (F.lit(s), F.lit(i))])
    w = W.partitionBy("shop_id", "session_id").orderBy("seq_no")
    wc = w.rowsBetween(W.unboundedPreceding, W.currentRow)
    is_t = lambda t: F.when(F.col("event_type") == t, 1).otherwise(0)  # noqa: E731
    qty = F.coalesce(F.col("qty"), F.lit(0))
    prev_prod = F.lag("product_id").over(w)

    ev = (pre.withColumn("category_key", F.concat_ws("|", "shop_id", "category"))
          .withColumn("brand_key", F.concat_ws("|", "shop_id", "brand"))
          .join(vocab_df("category"), "category_key", "left")
          .join(vocab_df("brand"), "brand_key", "left"))
    idx = lambda c: (F.when(F.col(c).isNull(), F.lit(FT.NULL_IDX))  # noqa: E731
                     .otherwise(F.coalesce(F.col(f"{c}_vidx"), F.lit(FT.OOV_IDX))))
    steps = ev.select(
        "shop_id", "session_id", "split", "label", "cut_seq_no", "session_start_ts",
        F.col("seq_no"),
        F.coalesce(type_map[F.col("event_type")], F.lit(FT.OOV_IDX)).alias("type_idx"),
        idx("category").alias("cat_idx"),
        idx("brand").alias("brand_idx"),
        F.coalesce(F.col("price").cast("double"), F.lit(0.0)).alias("price"),
        qty.cast("int").alias("qty"),
        F.coalesce(F.greatest(F.col("ts_ms") - F.lag("ts_ms").over(w), F.lit(0)), F.lit(0)).cast("long")
         .alias("dt_prev_ms"),
        F.greatest(F.col("ts_ms") - F.first("ts_ms").over(wc), F.lit(0)).cast("long").alias("t_first_ms"),
        F.when(F.col("product_id").isNotNull() & prev_prod.isNotNull() & (F.col("product_id") == prev_prod), 1)
         .otherwise(0).alias("same_prod"),
        F.sum(is_t("page_view")).over(wc).cast("int").alias("n_view"),
        F.sum(is_t("add_to_cart")).over(wc).cast("int").alias("n_add"),
        F.sum(is_t("remove_from_cart")).over(wc).cast("int").alias("n_remove"),
        F.sum(F.when(F.col("event_type") == "add_to_cart", qty)
              .when(F.col("event_type") == "remove_from_cart", -qty).otherwise(0)).over(wc).cast("long")
         .alias("cart_units"),
        F.row_number().over(w).alias("pos"))
    return (steps.groupBy("shop_id", "split", "session_id")
            .agg(F.first("label").alias("label"), F.first("cut_seq_no").alias("cut_seq_no"),
                 F.first("session_start_ts").alias("session_start_ts"),
                 F.array_sort(F.collect_list(F.struct(*FT.STEP_FIELDS))).alias("steps"))
            .withColumn("n_steps", F.size("steps"))
            .withColumn("shop_idx", F.coalesce(shop_map[F.col("shop_id")], F.lit(FT.OOV_IDX)))
            .select("shop_id", "split", "session_id", "label", "n_steps", "cut_seq_no", "session_start_ts",
                    "shop_idx", "steps"))


def report(spark, out, sessions, spec):
    ps = spark.read.parquet(os.path.join(out, "sessions"))
    q = [0.5, 0.9, 0.99, 1.0]
    rep = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "feature_version": spec["feature_version"], "max_len": spec["max_len"],
           "vocab_sizes": {k: len(spec[k]) for k in ("shops", "category", "brand")}, "shops": {}}
    excluded = {(r.shop_id, r.split): (r.n, r.pos) for r in
                sessions.filter(F.col("cut_seq_no") == 1).groupBy("shop_id", "split")
                .agg(F.count("*").alias("n"), F.sum("label_purchase").alias("pos")).collect()}
    for r in (ps.groupBy("shop_id", "split")
              .agg(F.count("*").alias("sessions"), F.sum("label").alias("pos"),
                   F.sum("n_steps").alias("samples"),
                   F.sum(F.col("label") * F.col("n_steps")).alias("pos_samples"),
                   F.sum(F.least(F.col("n_steps"), F.lit(spec["max_len"]))).alias("samples_k_le_max"),
                   F.percentile_approx("n_steps", q, 10000).alias("nq"))
              .orderBy("shop_id", "split").collect()):
        ex = excluded.get((r.shop_id, r.split), (0, 0))
        rep["shops"].setdefault(r.shop_id, {})[r.split] = {
            "sessions": r.sessions, "positives": r.pos, "pos_rate": round(r.pos / r.sessions, 4),
            "samples_all_k": r.samples, "pos_rate_samples": round(r.pos_samples / r.samples, 4),
            "samples_k_le_max_len": r.samples_k_le_max,
            "n_steps": dict(zip(["p50", "p90", "p99", "max"], r.nq)),
            "excluded_no_prefix": {"sessions": ex[0], "positives": ex[1] or 0}}
    by_k = (ps.select("shop_id", "split", "label", F.explode(F.sequence(F.lit(1), "n_steps")).alias("k"))
            .withColumn("bucket", bucket_col(F.col("k")))
            .groupBy("shop_id", "split", "bucket")
            .agg(F.count("*").alias("n"), F.avg("label").alias("pos_rate")).collect())
    for r in by_k:
        rep["shops"][r.shop_id][r.split].setdefault("by_prefix_len", {})[r.bucket] = {
            "samples": r.n, "pos_rate": round(r.pos_rate, 4)}
    order = [f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f">{lo - 1}") for lo, hi in K_BUCKETS]
    for shop in rep["shops"].values():
        for sp in shop.values():
            sp["by_prefix_len"] = {b: sp["by_prefix_len"][b] for b in order if b in sp.get("by_prefix_len", {})}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default="data/unified")
    ap.add_argument("--out", default="data/prefix")
    ap.add_argument("--shops", default=",".join(CLICK_SHOPS))
    ap.add_argument("--spec", help="reuse this feature_spec.json instead of building the vocab from train")
    ap.add_argument("--min-count", type=int, default=5, help="min train occurrences to enter a vocabulary")
    ap.add_argument("--max-categories", type=int, default=5000)
    ap.add_argument("--max-brands", type=int, default=5000)
    ap.add_argument("--files", type=int, default=8, help="parquet files per (shop, split)")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--shuffle-partitions", type=int, default=64)
    a = ap.parse_args()
    shops = [s for s in a.shops.split(",") if s]
    unknown = set(shops) - set(CLICK_SHOPS)
    if unknown:
        sys.exit(f"no session split for {sorted(unknown)} (clickstream shops: {CLICK_SHOPS})")

    spark = get_spark("prefixes", a.driver_memory, a.cores, a.shuffle_partitions)
    events = read_table(spark, a.unified, "events").filter(F.col("shop_id").isin(shops))
    sessions = read_table(spark, a.unified, "sessions").filter(F.col("shop_id").isin(shops))
    pre = prefix_events(events, sessions)
    if a.spec:
        spec = FT.load_spec(a.spec)
    else:
        spec = build_spec(pre, shops, a.min_count, a.max_categories, a.max_brands,
                          {"unified": os.path.abspath(a.unified), "split": "train", "min_count": a.min_count,
                           "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})
    os.makedirs(a.out, exist_ok=True)
    FT.save_spec(spec, os.path.join(a.out, "feature_spec.json"))

    (build_steps(spark, pre, spec)
     .repartition(a.files, "session_id")      # every task holds all (shop, split) -> a.files files per folder
     .sortWithinPartitions("shop_id", "split", "session_start_ts", "session_id")
     .write.mode("overwrite").partitionBy("shop_id", "split").parquet(os.path.join(a.out, "sessions")))

    rep = report(spark, a.out, sessions, spec)
    with open(os.path.join(a.out, "_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
