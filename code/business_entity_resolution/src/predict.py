"""Step 4: score the test candidates and write the two submission files.

Usage: python predict.py --work-dir <work> --split test --models-dir ../models --out-dir <output>
Writes <out-dir>/matching_results.tsv and <out-dir>/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline_core import (add_context, assign_one_to_one, decide, load_cands, load_norm, score_stage1_chunked,
                           stage2_matrix)


def write_id_lists(path: str, s1_ids: np.ndarray, pairs: pl.DataFrame, s1_index: np.ndarray, q_ids: np.ndarray,
                   header: tuple[str, str]):
    """pairs: frame with integer columns a (S1 row) and b (query row)."""
    lists = (pairs.with_columns(pl.Series("q_id", q_ids[pairs["b"].to_numpy()]))
             .group_by("a").agg(pl.col("q_id").unique().sort().str.join(",").alias("ids")))
    m = dict(zip(lists["a"].to_list(), lists["ids"].to_list()))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for i, sid in enumerate(s1_ids):
            f.write(f"{sid}\t{m.get(i, '')}\n")


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

    s1, qs = load_norm(a.work_dir, a.split)
    c = load_cands(a.work_dir, a.split, s1, qs)
    print("candidates", c.shape, f"per S1={c.shape[0]/s1.shape[0]:.2f}", f"{time.time()-t0:.0f}s", flush=True)
    s1_ids = s1["entity_id"].to_numpy()
    q_ids = qs["entity_id"].to_numpy()

    # candidate_pairs.tsv = exactly the pairs the model scores
    write_id_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1_ids, c.select("a", "b"), None, q_ids,
                   ("source1_entity_id", "candidate_entity_ids"))
    print("candidate_pairs.tsv written", flush=True)

    light = score_stage1_chunked(c, s1, qs, m1, chunk=1_000_000, workers=a.workers)
    light = add_context(light)
    light = light.with_columns(pl.Series("p2", m2.predict(stage2_matrix(light), num_threads=a.workers).astype(np.float32)))
    light.select("a", "b", "country", "p1", "p2", "h_rank").write_parquet(os.path.join(a.work_dir, f"scored_{a.split}.parquet"))
    assigned = assign_one_to_one(light, "p2")
    pred = decide(assigned, "p2", mode=decision, threshold=threshold)
    print("accepted pairs", pred.shape[0], f"per S1={pred.shape[0]/s1.shape[0]:.3f}", flush=True)
    write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, pred, None, q_ids,
                   ("source1_entity_id", "matched_entity_ids"))
    n_empty = s1.shape[0] - pred["a"].n_unique()
    print(f"matching_results.tsv written: {s1.shape[0]} rows, {n_empty} empty ({n_empty/s1.shape[0]:.3%}), "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
