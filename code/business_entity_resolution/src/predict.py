"""Step 4: score the test candidates and write the two submission files.

Countries are processed one at a time: candidate context features, the one-to-one
assignment and the per-entity decision never cross countries, so this is exact and
keeps memory bounded.

Usage: python predict.py --work-dir <work> --split test --models-dir ../models --out-dir <output>
Writes <out-dir>/matching_results.tsv and <out-dir>/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline_core import (STAGE2_COLS, add_context, assign_one_to_one, decide, load_cands, score_stage1_chunked,
                           stage2_matrix)
from train_full import load_country


def to_lists(pairs: pl.DataFrame, s1_ids: np.ndarray, q_ids: np.ndarray, col: str) -> pl.DataFrame:
    """(a, b) integer pairs -> frame [source1_entity_id, col] with sorted comma-joined ids."""
    p = pairs.select("a", "b").unique()
    if p.shape[0] == 0:
        return pl.DataFrame({"source1_entity_id": pl.Series([], dtype=pl.Utf8), col: pl.Series([], dtype=pl.Utf8)})
    p = p.with_columns(pl.Series("source1_entity_id", s1_ids[p["a"].to_numpy()]),
                       pl.Series("q_id", q_ids[p["b"].to_numpy()]))
    return p.group_by("source1_entity_id").agg(pl.col("q_id").sort().str.join(",").alias(col))


def write_tsv(path: str, s1_order: pl.DataFrame, lists: pl.DataFrame, col: str) -> pl.DataFrame:
    """One row per Source-1 entity, in test_source1 order; empty list when absent."""
    out = s1_order.join(lists, on="source1_entity_id", how="left", maintain_order="left").with_columns(
        pl.col(col).fill_null(""))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid, ids in out.select("source1_entity_id", col).iter_rows():
            f.write(f"{sid}\t{ids}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--decision", default=None, help="override: expected | threshold")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    cfg = json.load(open(os.path.join(a.models_dir, "config.json")))
    decision = a.decision or cfg["decision"]
    threshold = a.threshold if a.threshold is not None else cfg["threshold"]
    m1 = lgb.Booster(model_file=os.path.join(a.models_dir, "stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(a.models_dir, "stage2.txt"))
    nd = os.path.join(a.work_dir, "norm")
    all_s1 = pl.read_parquet(os.path.join(nd, f"{a.split}_s1.parquet"), columns=["entity_id", "country"]).rename(
        {"entity_id": "source1_entity_id"})
    countries = sorted(all_s1["country"].unique().to_list())
    cand_lists, match_lists = [], []
    n_cand = n_acc = 0
    for ctry in countries:
        s1, qs = load_country(a.work_dir, a.split, ctry)
        s1_ids = s1["entity_id"].to_numpy()
        q_ids = qs["entity_id"].to_numpy()
        c = load_cands(a.work_dir, a.split, s1, qs, country=ctry)
        n_cand += c.shape[0]
        cand_lists.append(to_lists(c, s1_ids, q_ids, "candidate_entity_ids"))   # exactly the pairs the model scores
        print(f"[{ctry}] candidates {c.shape[0]} ({c.shape[0]/max(s1.shape[0],1):.2f} per S1), {time.time()-t0:.0f}s",
              flush=True)
        light = score_stage1_chunked(c, s1, qs, m1, chunk=1_000_000, workers=a.workers)
        del c
        gc.collect()
        light = add_context(light)
        light = light.with_columns(pl.Series("p2", m2.predict(stage2_matrix(light), num_threads=a.workers)
                                             .astype(np.float32)))
        light.select(["a", "b", "country", "p2"] + [x for x in STAGE2_COLS if x in light.columns]).write_parquet(
            os.path.join(a.work_dir, f"scored_{a.split}_{ctry}.parquet"))
        assigned = assign_one_to_one(light, "p2")
        pred = decide(assigned, "p2", mode=decision, threshold=threshold)
        n_acc += pred.shape[0]
        match_lists.append(to_lists(pred, s1_ids, q_ids, "matched_entity_ids"))
        print(f"[{ctry}] accepted {pred.shape[0]} ({pred.shape[0]/max(s1.shape[0],1):.3f} per S1), "
              f"{time.time()-t0:.0f}s", flush=True)
        del light, assigned, pred, s1, qs
        gc.collect()
    s1_order = all_s1.select("source1_entity_id")
    write_tsv(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1_order, pl.concat(cand_lists), "candidate_entity_ids")
    out = write_tsv(os.path.join(a.out_dir, "matching_results.tsv"), s1_order, pl.concat(match_lists),
                    "matched_entity_ids")
    n_empty = int((out["matched_entity_ids"] == "").sum())
    print(f"candidate pairs {n_cand} ({n_cand/all_s1.shape[0]:.2f} per S1); accepted {n_acc} "
          f"({n_acc/all_s1.shape[0]:.3f} per S1)", flush=True)
    print(f"matching_results.tsv written: {all_s1.shape[0]} rows, {n_empty} empty ({n_empty/all_s1.shape[0]:.3%}), "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
