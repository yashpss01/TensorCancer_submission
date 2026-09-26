"""Shared machinery for training and inference.

* load candidates produced by blocking.py and map ids to integer indices
* chunked pairwise feature computation + stage-1 scoring
* context (competition) features + stage-2 scoring
* one-to-one assignment and expected-F0.5 decision per Source-1 entity
* macro F0.5 evaluation
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import polars as pl

from features import add_features, FEATURE_COLS

NORM_COLS = ["entity_id", "name_full", "name_core", "name_sig", "name_legal", "name_nospace", "name_sorted",
             "name_nonlatin", "addr_full", "addr_alpha", "addr_nums", "addr_first_num", "addr_state", "addr_empty"]

# raw columns kept after stage 1 (small) for the context model
KEEP_AFTER_S1 = ["cos_joint", "cos_jskel", "cos_fullskel", "cos_name", "h", "h_rank", "h_gap", "h_ntie", "src_extra",
                 "q_ncand_all", "s1_name_freq", "s1_sig_freq",
                 "r_core_tset", "r_addr_tset", "an_jac", "firstnum_agree", "state_agree", "nonlatin_b",
                 "addr_empty_b", "addr_empty_a", "is_s2", "sig_eq", "nc_jac", "aa_jac", "len_addr_b"]

CONTEXT_COLS = ["p1", "p1_logit", "q_p1_max", "q_p1_ratio", "q_p1_rank", "q_p1_second", "q_p1_gap", "q_ncand",
                "q_p1_ntie", "q_n_strong", "q_same_name_cnt",
                "s_top_cnt", "s_top_cnt_s2", "s_top_cnt_s3", "s_strong_cnt", "s_p1_mean_others", "s_ncand",
                "q_h_rank_of_top", "q_is_top"]
STAGE2_COLS = CONTEXT_COLS + KEEP_AFTER_S1


def load_norm(work_dir: str, split: str):
    nd = os.path.join(work_dir, "norm")
    s1 = pl.read_parquet(os.path.join(nd, f"{split}_s1.parquet"), columns=NORM_COLS + ["country"])
    qs = pl.concat([pl.read_parquet(os.path.join(nd, f"{split}_s{i}.parquet"), columns=NORM_COLS) for i in (2, 3)])
    return s1, qs


# pairs whose stage-1 probability is below this are dropped before the context
# model (they never become matches; dropping them keeps memory bounded).
P1_MIN = 0.02


def load_cands(work_dir: str, split: str, s1: pl.DataFrame, qs: pl.DataFrame, country: str | None = None) -> pl.DataFrame:
    pattern = "*.parquet" if country is None else f"{country}_*.parquet"
    files = sorted(glob.glob(os.path.join(work_dir, "cands", split, pattern)))
    s1_idx = s1.select("entity_id").with_row_index("a").rename({"entity_id": "s1_id"})
    q_idx = qs.select("entity_id").with_row_index("b").rename({"entity_id": "q_id"})
    parts = []
    for f in files:
        c = pl.read_parquet(f)
        c = c.join(s1_idx, on="s1_id", how="inner").join(q_idx, on="q_id", how="inner")
        c = c.with_columns(pl.col("a").cast(pl.Int32), pl.col("b").cast(pl.Int32),
                           pl.col("country").cast(pl.Categorical),
                           pl.lit(1 if "_extra" in os.path.basename(f) else 0, dtype=pl.Int8).alias("src_extra"),
                           ).drop("s1_id", "q_id", "h_rank", "h_gap")
        parts.append(c)
    c = pl.concat(parts)
    del parts
    # tie-aware heuristic rank (identical candidates share the rank) + tie count
    c = c.with_columns(
        pl.col("h").rank(method="min", descending=True).over("b").cast(pl.Int16).alias("h_rank"),
        (pl.col("h").max().over("b") - pl.col("h")).alias("h_gap"),
        pl.len().over("b").cast(pl.Int16).alias("q_ncand_all"),
    ).with_columns(
        (pl.col("h_gap") < 1e-4).cast(pl.Int16).sum().over("b").alias("h_ntie"),
    )
    # how many Source-1 entities of the same country share this candidate's name
    nf = s1.group_by("country", "name_core").len().rename({"len": "s1_name_freq"})
    sf = s1.group_by("country", "name_sig").len().rename({"len": "s1_sig_freq"})
    s1f = (s1.with_row_index("a").with_columns(pl.col("a").cast(pl.Int32))
           .join(nf, on=["country", "name_core"], how="left").join(sf, on=["country", "name_sig"], how="left")
           .select("a", pl.col("s1_name_freq").cast(pl.Int32), pl.col("s1_sig_freq").cast(pl.Int32)))
    c = c.join(s1f, on="a", how="left")
    return c


def add_labels(c: pl.DataFrame, work_dir: str, s1: pl.DataFrame, qs: pl.DataFrame) -> pl.DataFrame:
    pairs = pl.read_parquet(os.path.join(work_dir, "norm", "train_pairs.parquet"))
    s1_idx = s1.select("entity_id").with_row_index("a").rename({"entity_id": "s1"})
    q_idx = qs.select("entity_id").with_row_index("b").rename({"entity_id": "mid"})
    p = pairs.join(s1_idx, on="s1", how="inner").join(q_idx, on="mid", how="inner").select(
        pl.col("a").cast(pl.Int32), pl.col("b").cast(pl.Int32)).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    c = c.join(p, on=["a", "b"], how="left").with_columns(pl.col("label").fill_null(0))
    n_true = p.shape[0]
    n_found = int(c["label"].sum())
    print(f"blocking recall: {n_found}/{n_true} = {n_found/max(n_true,1):.4f}; pairs={c.shape[0]}, "
          f"per query={c.shape[0]/qs.shape[0]:.2f}, per S1={c.shape[0]/s1.shape[0]:.2f}", flush=True)
    return c


def truth_pairs(work_dir: str, s1: pl.DataFrame, qs: pl.DataFrame) -> pl.DataFrame:
    pairs = pl.read_parquet(os.path.join(work_dir, "norm", "train_pairs.parquet"))
    s1_idx = s1.select("entity_id").with_row_index("a").rename({"entity_id": "s1"})
    q_idx = qs.select("entity_id").with_row_index("b").rename({"entity_id": "mid"})
    return pairs.join(s1_idx, on="s1", how="inner").join(q_idx, on="mid", how="inner").select(
        pl.col("a").cast(pl.Int32), pl.col("b").cast(pl.Int32))


def attach_norm(chunk: pl.DataFrame, s1: pl.DataFrame, qs: pl.DataFrame) -> pl.DataFrame:
    a = s1.drop("country").with_row_index("a").with_columns(pl.col("a").cast(pl.Int32)).rename(
        {c: c + "_a" for c in NORM_COLS})
    b = qs.with_row_index("b").with_columns(pl.col("b").cast(pl.Int32)).rename({c: c + "_b" for c in NORM_COLS})
    return chunk.with_columns(pl.col("country").cast(pl.Utf8)).join(a, on="a", how="left").join(b, on="b", how="left")


def featurize(chunk: pl.DataFrame, s1: pl.DataFrame, qs: pl.DataFrame, workers: int = 8) -> pl.DataFrame:
    df = attach_norm(chunk, s1, qs)
    df = add_features(df, workers=workers)
    return df


def stage1_matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURE_COLS).to_numpy().astype(np.float32)


def score_stage1_chunked(c: pl.DataFrame, s1: pl.DataFrame, qs: pl.DataFrame, model, chunk: int = 1_000_000,
                         workers: int = 8, extra_cols=()) -> pl.DataFrame:
    """Compute features chunk by chunk, predict stage-1, keep a light frame."""
    outs = []
    t0 = time.time()
    n = c.shape[0]
    for i, start in enumerate(range(0, n, chunk)):
        part = c.slice(start, chunk)
        df = featurize(part, s1, qs, workers)
        X = stage1_matrix(df)
        p1 = model.predict(X, num_threads=workers).astype(np.float32)
        keep = ["a", "b", "country"] + [x for x in ("label",) if x in df.columns] + list(extra_cols)
        keep += [k for k in KEEP_AFTER_S1 if k in df.columns and k not in keep]
        light = df.select(keep).with_columns(pl.Series("p1", p1)).filter(pl.col("p1") >= P1_MIN)
        outs.append(light)
        print(f"  stage1 chunk {i}: {part.shape[0]} pairs ({time.time()-t0:.0f}s)", flush=True)
    return pl.concat(outs)


def add_context(df: pl.DataFrame) -> pl.DataFrame:
    """Competition features: how this pair compares to the other candidates of the
    same Source-2/3 record (b) and of the same Source-1 entity (a)."""
    eps = 1e-6
    df = df.with_columns((pl.col("p1").clip(eps, 1 - eps)).alias("_p"))
    df = df.with_columns((pl.col("_p") / (1 - pl.col("_p"))).log().alias("p1_logit")).drop("_p")
    df = df.with_columns(
        pl.col("p1").max().over("b").alias("q_p1_max"),
        pl.col("p1").rank(method="min", descending=True).over("b").alias("q_p1_rank"),
        pl.len().over("b").alias("q_ncand"),
        pl.col("p1").sort(descending=True).slice(1, 1).first().over("b").alias("q_p1_second"),
        (pl.col("p1") > 0.5).cast(pl.Int16).sum().over("b").alias("q_n_strong"),
        pl.col("sig_eq").cast(pl.Int16).sum().over("b").alias("q_same_name_cnt"),
    ).with_columns(
        ((pl.col("q_p1_max") - pl.col("p1")) < 1e-3).cast(pl.Int16).sum().over("b").alias("q_p1_ntie"),
    ).with_columns(
        (pl.col("p1") / (pl.col("q_p1_max") + eps)).alias("q_p1_ratio"),
        pl.col("q_p1_second").fill_null(0.0),
        (pl.col("q_p1_rank") == 1).cast(pl.Int8).alias("q_is_top"),
    ).with_columns(
        pl.when(pl.col("q_is_top") == 1).then(pl.col("p1") - pl.col("q_p1_second"))
          .otherwise(pl.col("p1") - pl.col("q_p1_max")).alias("q_p1_gap"),
        pl.when(pl.col("q_is_top") == 1).then(pl.col("h_rank")).otherwise(None).max().over("b").alias("q_h_rank_of_top"),
    )
    # Source-1 side: how many records already claim this entity as their best match
    top = (pl.col("q_is_top") == 1) & (pl.col("p1") > 0.5)
    df = df.with_columns(
        top.cast(pl.Int32).sum().over("a").alias("s_top_cnt"),
        (top & (pl.col("is_s2") == 1)).cast(pl.Int32).sum().over("a").alias("s_top_cnt_s2"),
        (top & (pl.col("is_s2") == 0)).cast(pl.Int32).sum().over("a").alias("s_top_cnt_s3"),
        (pl.col("p1") > 0.9).cast(pl.Int32).sum().over("a").alias("s_strong_cnt"),
        pl.col("p1").sum().over("a").alias("_s_sum"),
        pl.len().over("a").alias("s_ncand"),
    ).with_columns(
        ((pl.col("_s_sum") - pl.col("p1")) / pl.max_horizontal(pl.col("s_ncand") - 1, 1)).alias("s_p1_mean_others"),
    ).drop("_s_sum")
    # exclude self from top counts
    df = df.with_columns(
        (pl.col("s_top_cnt") - top.cast(pl.Int32)).alias("s_top_cnt"),
        (pl.col("s_top_cnt_s2") - (top & (pl.col("is_s2") == 1)).cast(pl.Int32)).alias("s_top_cnt_s2"),
        (pl.col("s_top_cnt_s3") - (top & (pl.col("is_s2") == 0)).cast(pl.Int32)).alias("s_top_cnt_s3"),
        (pl.col("s_strong_cnt") - (pl.col("p1") > 0.9).cast(pl.Int32)).alias("s_strong_cnt"),
    )
    return df


def stage2_matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(STAGE2_COLS).to_numpy().astype(np.float32)


# ----------------------------------------------------------------------------- #
# Decision
# ----------------------------------------------------------------------------- #
def assign_one_to_one(df: pl.DataFrame, prob_col: str = "p2") -> pl.DataFrame:
    """Each Source-2/3 record may belong to at most one Source-1 entity: keep the
    candidate with the highest probability for every record."""
    return df.filter(pl.col(prob_col) == pl.col(prob_col).max().over("b")).unique(subset=["b"], keep="first")


def expected_f05_decision(probs: np.ndarray) -> np.ndarray:
    """Given the (calibrated) match probabilities of the candidates of ONE entity,
    return a boolean mask of the subset that maximises the expected F0.5 under an
    independence assumption.  Candidates are considered in decreasing probability
    order; the optimal subset is always a prefix of that order.

    F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP); an empty prediction scores 1 when
    the entity truly has no matches and 0 otherwise.
    """
    order = np.argsort(-probs)
    p = probs[order]
    n = len(p)
    # distribution of the number of true matches among the excluded set / included set
    # dp over candidates: we need for each prefix k: P(TP=t) for included and P(FN=f) for excluded.
    # Use full enumeration via dynamic programming (n is small, <= ~20).
    best_val = np.prod(1 - p)  # expected score of predicting nothing = P(no true matches)
    best_k = 0
    # included distribution
    inc = np.zeros(n + 1)
    inc[0] = 1.0
    # excluded distributions for suffixes: exc[k] = distribution of true count among p[k:]
    exc = [None] * (n + 1)
    d = np.zeros(n + 1)
    d[0] = 1.0
    exc[n] = d.copy()
    for k in range(n - 1, -1, -1):
        nd = np.zeros(n + 1)
        nd[1:] += d[:-1] * p[k]
        nd += d * (1 - p[k])
        d = nd
        exc[k] = d.copy()
    for k in range(1, n + 1):
        ninc = np.zeros(n + 1)
        ninc[1:] += inc[:-1] * p[k - 1]
        ninc += inc * (1 - p[k - 1])
        inc = ninc
        # expected F over TP in 0..k and FN in 0..n-k
        tp = np.arange(0, k + 1)
        fn = np.arange(0, n - k + 1)
        TP, FN = np.meshgrid(tp, fn, indexing="ij")
        FP = k - TP
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(TP > 0, 1.25 * TP / (1.25 * TP + 0.25 * FN + FP), 0.0)
        val = float(np.sum(F * np.outer(inc[: k + 1], exc[k][: n - k + 1])))
        if val > best_val + 1e-12:
            best_val = val
            best_k = k
    mask = np.zeros(n, dtype=bool)
    mask[order[:best_k]] = True
    return mask


def decide(df: pl.DataFrame, prob_col: str = "p2", mode: str = "expected", threshold: float = 0.5) -> pl.DataFrame:
    """Return the accepted pairs (a, b)."""
    if mode == "threshold":
        return df.filter(pl.col(prob_col) >= threshold).select("a", "b")
    out_a, out_b = [], []
    for (a,), g in df.group_by(["a"]):
        probs = g[prob_col].to_numpy().astype(np.float64)
        bs = g["b"].to_numpy()
        m = expected_f05_decision(probs)
        if m.any():
            out_a.append(np.full(int(m.sum()), a, dtype=np.int32))
            out_b.append(bs[m])
    if not out_a:
        return pl.DataFrame({"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.Int32)})
    return pl.DataFrame({"a": np.concatenate(out_a), "b": np.concatenate(out_b)})


# ----------------------------------------------------------------------------- #
# Evaluation
# ----------------------------------------------------------------------------- #
def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, all_a: np.ndarray, by: pl.DataFrame | None = None) -> dict:
    """pred/truth: frames of (a, b).  all_a: every Source-1 index to be scored."""
    tp = pred.join(truth, on=["a", "b"], how="inner").group_by("a").len().rename({"len": "tp"})
    np_ = pred.group_by("a").len().rename({"len": "npred"})
    nt = truth.group_by("a").len().rename({"len": "ntrue"})
    base = pl.DataFrame({"a": all_a.astype(np.int32)})
    m = base.join(np_, on="a", how="left").join(nt, on="a", how="left").join(tp, on="a", how="left").fill_null(0)
    m = m.with_columns(
        pl.when((pl.col("ntrue") == 0) & (pl.col("npred") == 0)).then(1.0)
          .when(pl.col("tp") == 0).then(0.0)
          .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("ntrue") - pl.col("tp")) + (pl.col("npred") - pl.col("tp"))))
          .alias("f")
    )
    res = {"macro_f05": float(m["f"].mean()), "n_entities": m.shape[0],
           "pair_precision": float(m["tp"].sum() / max(m["npred"].sum(), 1)),
           "pair_recall": float(m["tp"].sum() / max(m["ntrue"].sum(), 1)),
           "singleton_acc": float(m.filter(pl.col("ntrue") == 0)["f"].mean()),
           "nonsingleton_f": float(m.filter(pl.col("ntrue") > 0)["f"].mean())}
    if by is not None:
        mm = m.join(by, on="a", how="left")
        res["by_country"] = {k[0]: round(v, 5) for k, v in
                             mm.group_by(["country"]).agg(pl.col("f").mean()).iter_rows()}
    return res, m
