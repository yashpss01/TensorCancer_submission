"""Blocking quality report: confusion matrix of the candidate-generation stage.

Usage: python blocking_report.py --work-dir <work> --split trainsub
(For a split without ground truth only the reduction ratio is reported.)
"""
from __future__ import annotations

import argparse
import glob
import os

import polars as pl


def fmt(n):
    return f"{int(n):,}"


def report(title: str, n_s1: int, n_t: int, n_cand: int, tp: int | None, n_true: int | None):
    all_pairs = n_s1 * n_t
    print("=" * 71)
    print(f"      CANDIDATE GENERATION (BLOCKING) CONFUSION MATRIX  [{title}]")
    print("=" * 71)
    print(f"Source-1 entities: {fmt(n_s1)}   Source-2/3 targets: {fmt(n_t)}")
    print(f"Total Comparison Space (S1 x Targets): {fmt(all_pairs):>22}")
    print(f"Candidate pairs generated (C):         {fmt(n_cand):>22}   "
          f"({n_cand/max(n_t,1):.2f} per target, {n_cand/max(n_s1,1):.2f} per S1)")
    print("-" * 71)
    if tp is not None:
        fn = n_true - tp
        fp = n_cand - tp
        tn = all_pairs - tp - fn - fp
        rec = tp / max(n_true, 1)
        prec = tp / max(n_cand, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        print(f"True Positives  (TP - Matches Retained): {fmt(tp):>18} ({rec:7.2%})")
        print(f"False Negatives (FN - Matches Missed):   {fmt(fn):>18} ({1-rec:7.2%})")
        print(f"False Positives (FP - Distractor Pairs): {fmt(fp):>18} ({fp/max(n_cand,1):7.2%})")
        print(f"True Negatives  (TN - Pairs Pruned):     {fmt(tn):>18} ({spec:9.4%})")
        print("-" * 71)
        print("KEY PERFORMANCE METRICS:")
        print(f"  • Recall / Pair Completeness (TP / [TP+FN]):   {rec:8.2%}")
        print(f"  • Precision / Pair Quality   (TP / [TP+FP]):   {prec:8.2%}")
        print(f"  • Reduction Ratio (RR)       (1 - C / All):    {1 - n_cand/all_pairs:10.4%}")
        print(f"  • Specificity                (TN / [TN+FP]):   {spec:10.4%}")
        print(f"  • Candidate F1-Score:                          {f1:8.4f}")
    else:
        print("KEY PERFORMANCE METRICS (no ground truth for this split):")
        print(f"  • Reduction Ratio (RR)       (1 - C / All):    {1 - n_cand/all_pairs:10.4%}")
    print("=" * 71)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="trainsub")
    a = ap.parse_args()
    nd = os.path.join(a.work_dir, "norm")
    s1 = pl.read_parquet(os.path.join(nd, f"{a.split}_s1.parquet"), columns=["entity_id", "country"])
    qs = pl.concat([pl.read_parquet(os.path.join(nd, f"{a.split}_s{i}.parquet"), columns=["entity_id", "country"])
                    for i in (2, 3)])
    files = sorted(glob.glob(os.path.join(a.work_dir, "cands", a.split, "*.parquet")))
    c = pl.concat([pl.read_parquet(f, columns=["q_id", "s1_id", "country", "h_rank"]) for f in files])
    has_gt = os.path.exists(os.path.join(nd, "train_pairs.parquet")) and a.split.startswith("train")
    truth = None
    if has_gt:
        pairs = pl.read_parquet(os.path.join(nd, "train_pairs.parquet"))
        truth = (pairs.join(s1.select("entity_id"), left_on="s1", right_on="entity_id", how="semi")
                 .join(qs.select("entity_id"), left_on="mid", right_on="entity_id", how="semi"))
        truth = truth.join(s1.rename({"entity_id": "s1"}), on="s1", how="left")
    countries = sorted(s1["country"].unique().to_list())
    tot = dict(s1=0, t=0, c=0, tp=0, true=0)
    for ctry in countries + ["ALL"]:
        if ctry == "ALL":
            n_s1, n_t, cc = tot["s1"], tot["t"], None
            n_cand, tp, n_true = tot["c"], (tot["tp"] if has_gt else None), (tot["true"] if has_gt else None)
        else:
            n_s1 = s1.filter(pl.col("country") == ctry).shape[0]
            n_t = qs.filter(pl.col("country") == ctry).shape[0]
            cc = c.filter(pl.col("country") == ctry)
            n_cand = cc.shape[0]
            tp = n_true = None
            if has_gt:
                tr = truth.filter(pl.col("country") == ctry)
                n_true = tr.shape[0]
                tp = tr.join(cc, left_on=["s1", "mid"], right_on=["s1_id", "q_id"], how="semi").shape[0]
                tot["tp"] += tp
                tot["true"] += n_true
            tot["s1"] += n_s1
            tot["t"] += n_t
            tot["c"] += n_cand
        report(ctry, n_s1, n_t, n_cand, tp, n_true)
    if has_gt:
        # recall by heuristic-rank bucket and candidate count distribution
        hit = truth.join(c, left_on=["s1", "mid"], right_on=["s1_id", "q_id"], how="inner")
        print("Retained true matches by heuristic rank:",
              hit.with_columns(pl.when(pl.col("h_rank") <= 1).then(pl.lit("1")).when(pl.col("h_rank") <= 3).then(pl.lit("2-3"))
                               .when(pl.col("h_rank") <= 10).then(pl.lit("4-10")).otherwise(pl.lit("11+")).alias("bucket"))
              .group_by("bucket").len().sort("bucket").to_dicts())
    per_s1 = c.group_by("s1_id").len()
    print("Candidates per S1 entity (entities with >=1 candidate): mean %.2f, median %d, p90 %d, p99 %d, max %d" % (
        per_s1["len"].mean(), per_s1["len"].median(), per_s1["len"].quantile(0.9), per_s1["len"].quantile(0.99), per_s1["len"].max()))
    print("S1 entities with zero candidates:", s1.shape[0] - per_s1.shape[0])


if __name__ == "__main__":
    main()
