"""Step 3: train the two-stage matcher on the training replica and validate it.

Folds are assigned per Source-1 entity:
  folds 0-3 -> stage-1 (pairwise) model
  folds 4-6 -> stage-2 (context) model, scored with out-of-sample stage-1 probabilities
  folds 7-9 -> validation (macro F0.5, decision-rule selection)

Usage: python train.py --work-dir <work> --split trainsub --models-dir ../models
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from features import FEATURE_COLS
from pipeline_core import (STAGE2_COLS, add_context, add_labels, assign_one_to_one, decide, featurize,
                           load_cands, load_norm, macro_f05, score_stage1_chunked, stage1_matrix, stage2_matrix,
                           truth_pairs)

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=8, verbose=-1, max_bin=255)


def fold_of(a: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int64) * 2654435761) % 1000003) % 10


def train_lgb(X, y, Xv, yv, rounds=2500, params=LGB_PARAMS):
    dtr = lgb.Dataset(X, y)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="trainsub")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--s1-max-pairs", type=int, default=4_000_000)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.models_dir, exist_ok=True)
    t0 = time.time()
    s1, qs = load_norm(a.work_dir, a.split)
    c = load_cands(a.work_dir, a.split, s1, qs)
    c = add_labels(c, a.work_dir, s1, qs)
    c = c.with_columns(pl.Series("fold", fold_of(c["a"].to_numpy()).astype(np.int8)))
    print("candidates loaded", c.shape, f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- stage 1 ----------------
    tr = c.filter(pl.col("fold") <= 3)
    pos = tr.filter(pl.col("label") == 1)
    neg = tr.filter(pl.col("label") == 0)
    n_pos = min(pos.shape[0], int(a.s1_max_pairs / (1 + a.neg_ratio)))
    pos = pos.sample(n=n_pos, seed=1)
    neg = neg.sample(n=min(neg.shape[0], int(n_pos * a.neg_ratio)), seed=2)
    samp = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=3)
    print("stage-1 sample", samp.shape, "pos", n_pos, flush=True)
    df = featurize(samp, s1, qs, a.workers)
    X = stage1_matrix(df)
    y = df["label"].to_numpy()
    f4 = c.filter(pl.col("fold") == 4)
    es = f4.sample(n=min(600_000, f4.shape[0]), seed=4)
    del f4
    dfe = featurize(es, s1, qs, a.workers)
    Xe, ye = stage1_matrix(dfe), dfe["label"].to_numpy()
    print("features ready", X.shape, f"{time.time()-t0:.0f}s", flush=True)
    m1 = train_lgb(X, y, Xe, ye)
    m1.save_model(os.path.join(a.models_dir, "stage1.txt"))
    imp = sorted(zip(FEATURE_COLS, m1.feature_importance("gain")), key=lambda x: -x[1])
    print("stage-1 best iter", m1.best_iteration, "top features:", [(k, int(v)) for k, v in imp[:20]], flush=True)
    del df, X, dfe, Xe

    # ---------------- stage-1 scoring of everything ----------------
    # only the folds used for stage 2 and validation need stage-1 scores
    light = score_stage1_chunked(c.filter(pl.col("fold") >= 4), s1, qs, m1, chunk=1_000_000,
                                 workers=a.workers, extra_cols=("fold",))
    del c
    light = add_context(light)
    print("context features ready", light.shape, f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- stage 2 ----------------
    tr2 = light.filter((pl.col("fold") >= 4) & (pl.col("fold") <= 5))
    es2 = light.filter(pl.col("fold") == 6)
    if tr2.shape[0] > 8_000_000:
        tr2 = tr2.sample(n=8_000_000, seed=5)
    if es2.shape[0] > 1_000_000:
        es2 = es2.sample(n=1_000_000, seed=6)
    m2 = train_lgb(stage2_matrix(tr2), tr2["label"].to_numpy(), stage2_matrix(es2), es2["label"].to_numpy())
    m2.save_model(os.path.join(a.models_dir, "stage2.txt"))
    imp2 = sorted(zip(STAGE2_COLS, m2.feature_importance("gain")), key=lambda x: -x[1])
    print("stage-2 best iter", m2.best_iteration, "top features:", [(k, int(v)) for k, v in imp2[:20]], flush=True)

    # ---------------- validation ----------------
    va = light.filter(pl.col("fold") >= 7)
    va = va.with_columns(pl.Series("p2", m2.predict(stage2_matrix(va), num_threads=a.workers).astype(np.float32)))
    truth = truth_pairs(a.work_dir, s1, qs)
    all_a = np.arange(s1.shape[0], dtype=np.int32)
    va_a = all_a[fold_of(all_a) >= 7]
    truth_va = truth.filter(pl.Series(fold_of(truth["a"].to_numpy()) >= 7))
    by = s1.select("country").with_row_index("a").with_columns(pl.col("a").cast(pl.Int32))
    # pair-level metrics of p2 on the valid folds
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("valid pairs", va.shape, "AUC p1", round(roc_auc_score(va["label"], va["p1"]), 5),
          "AUC p2", round(roc_auc_score(va["label"], va["p2"]), 5),
          "AP p2", round(average_precision_score(va["label"], va["p2"]), 5), flush=True)
    assigned = assign_one_to_one(va, "p2")
    results = {}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = decide(assigned, "p2", mode="threshold", threshold=thr)
        r, _ = macro_f05(pred, truth_va, va_a, by)
        results[f"thr_{thr}"] = r
        print(f"threshold {thr}: {r}", flush=True)
    pred = decide(assigned, "p2", mode="expected")
    r, m = macro_f05(pred, truth_va, va_a, by)
    results["expected"] = r
    print(f"expected-F decision: {r}", flush=True)
    # also: no one-to-one constraint, threshold 0.5
    pred = decide(va, "p2", mode="threshold", threshold=0.5)
    r, _ = macro_f05(pred, truth_va, va_a, by)
    results["no_1to1_thr_0.5"] = r
    print(f"no one-to-one, thr 0.5: {r}", flush=True)
    best = max(results, key=lambda k: results[k]["macro_f05"])
    cfg = {"decision": "expected" if best == "expected" else "threshold",
           "threshold": float(best.split("_")[1]) if best.startswith("thr_") else 0.5,
           "results": results, "stage1_iter": m1.best_iteration, "stage2_iter": m2.best_iteration}
    with open(os.path.join(a.models_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("best:", best, cfg["results"][best], f"total {time.time()-t0:.0f}s", flush=True)
    # save validation frame for error analysis
    va.write_parquet(os.path.join(a.work_dir, f"valid_scored_{a.split}.parquet"))
    m.write_parquet(os.path.join(a.work_dir, f"valid_entities_{a.split}.parquet"))


if __name__ == "__main__":
    main()
