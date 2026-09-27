"""Create a scaled-down replica of the training world for model fitting/validation.

We keep a random fraction of Source-1 entities, ALL of their matched Source-2/3
records, and the same fraction of the unmatched (distractor) Source-2/3 records.
The result has the same records-per-entity and distractor ratios as the full data,
so blocking recall, precision and the macro F0.5 measured on it are realistic.

Writes <work>/norm/trainsub_s{1,2,3}.parquet and <work>/norm/trainsub_gt.parquet
"""
from __future__ import annotations

import argparse
import os

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    nd = os.path.join(a.work_dir, "norm")
    s1 = pl.read_parquet(os.path.join(nd, "train_s1.parquet"))
    gt = pl.read_parquet(os.path.join(nd, "train_gt.parquet"))
    pairs = pl.read_parquet(os.path.join(nd, "train_pairs.parquet"))
    keep_s1 = s1.select("entity_id").sample(fraction=a.frac, seed=a.seed)
    s1s = s1.join(keep_s1, on="entity_id", how="semi")
    gts = gt.join(keep_s1, left_on="source1_entity_id", right_on="entity_id", how="semi")
    keep_mid = pairs.join(keep_s1, left_on="s1", right_on="entity_id", how="semi").select("mid")
    for i in (2, 3):
        s = pl.read_parquet(os.path.join(nd, f"train_s{i}.parquet"))
        matched = s.join(keep_mid, left_on="entity_id", right_on="mid", how="semi")
        unmatched = s.join(pairs.select("mid"), left_on="entity_id", right_on="mid", how="anti").sample(fraction=a.frac, seed=a.seed + i)
        out = pl.concat([matched, unmatched]).sample(fraction=1.0, shuffle=True, seed=a.seed)
        out.write_parquet(os.path.join(nd, f"trainsub_s{i}.parquet"))
        print(f"trainsub_s{i}", out.shape, "matched", matched.shape[0], "unmatched", unmatched.shape[0])
    s1s.write_parquet(os.path.join(nd, "trainsub_s1.parquet"))
    gts.write_parquet(os.path.join(nd, "trainsub_gt.parquet"))
    print("trainsub_s1", s1s.shape, "gt", gts.shape)


if __name__ == "__main__":
    main()
