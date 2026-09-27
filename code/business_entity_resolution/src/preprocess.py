"""Step 1: read the raw TSVs, normalise every record, cache as parquet.

Usage:  python preprocess.py --data-dir <student_resource/dataset> --work-dir <work>
Produces <work>/norm/{train,test}_s{1,2,3}.parquet and <work>/norm/train_pairs.parquet
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import polars as pl

from normalize import norm_address, norm_name

NAME_KEYS = ["name_full", "name_core", "name_sig", "name_legal", "name_nospace", "name_sorted", "name_nonlatin"]
ADDR_KEYS = ["addr_full", "addr_alpha", "addr_nums", "addr_first_num", "addr_state", "addr_empty"]


def _norm_chunk(args):
    names, addrs = args
    cols = {k: [] for k in NAME_KEYS + ADDR_KEYS}
    for n, a in zip(names, addrs):
        dn = norm_name(n)
        da = norm_address(a)
        for k in NAME_KEYS:
            cols[k].append(dn[k])
        for k in ADDR_KEYS:
            cols[k].append(da[k])
    return cols


def normalise_frame(df: pl.DataFrame, procs: int = 8, chunk: int = 50_000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(procs) as pool:
        parts = pool.map(_norm_chunk, jobs)
    merged = {k: [] for k in NAME_KEYS + ADDR_KEYS}
    for p in parts:
        for k in merged:
            merged[k].extend(p[k])
    out = df.with_columns([pl.Series(k, v) for k, v in merged.items()])
    return out


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, empty_string_is_null=False,
                       schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                                         "business_address": pl.Utf8, "country": pl.Utf8})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()
    out_dir = os.path.join(a.work_dir, "norm")
    os.makedirs(out_dir, exist_ok=True)
    for split in a.splits.split(","):
        for s in (1, 2, 3):
            path = os.path.join(a.data_dir, split, f"{split}_source{s}.tsv")
            df = read_tsv(path).with_columns(pl.col("business_name").fill_null(""),
                                             pl.col("business_address").fill_null(""))
            df = normalise_frame(df, a.procs)
            df.write_parquet(os.path.join(out_dir, f"{split}_s{s}.parquet"))
            print(split, s, df.shape, flush=True)
        if split == "train":
            gt = read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")) if False else \
                pl.read_csv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"), separator="\t",
                            quote_char=None, empty_string_is_null=False)
            gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
            gt.write_parquet(os.path.join(out_dir, "train_gt.parquet"))
            pairs = (gt.filter(pl.col("matched_entity_ids") != "")
                     .with_columns(pl.col("matched_entity_ids").str.split(","))
                     .explode("matched_entity_ids")
                     .rename({"matched_entity_ids": "mid", "source1_entity_id": "s1"}))
            pairs.write_parquet(os.path.join(out_dir, "train_pairs.parquet"))
            print("pairs", pairs.shape, flush=True)


if __name__ == "__main__":
    main()
