"""Optional extra blocking pass for the hardest records.

Records whose address is empty, or whose name collapsed to a single token
(domain-style names such as ``summitanimalhospital``), cannot be found by the
word-level indexes when the name also contains a typo.  For that small subset
(~5% of records) we run a character-trigram TF-IDF index over the space-less
Source-1 names and append the top-k neighbours as additional candidates.

Output files sit next to the main blocking output (``<country>_extra.parquet``)
with the same schema; ``cos_name`` carries the character cosine.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from blocking import csr_to_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100_000)
    a = ap.parse_args()
    nd = os.path.join(a.work_dir, "norm")
    out_dir = os.path.join(a.work_dir, "cands", a.split)
    cols = ["entity_id", "country", "name_core", "name_nospace", "addr_empty"]
    s1 = pl.read_parquet(os.path.join(nd, f"{a.split}_s1.parquet"), columns=cols)
    qs = pl.concat([pl.read_parquet(os.path.join(nd, f"{a.split}_s{i}.parquet"), columns=cols) for i in (2, 3)])
    hard = qs.filter(pl.col("addr_empty") | (pl.col("name_core").str.count_matches(" ") == 0))
    print(f"hard records: {hard.shape[0]} of {qs.shape[0]}", flush=True)
    for ctry in sorted(s1["country"].unique().to_list()):
        t0 = time.time()
        s1c = s1.filter(pl.col("country") == ctry)
        qc = hard.filter(pl.col("country") == ctry)
        if qc.shape[0] == 0:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.2, sublinear_tf=True, dtype=np.float32)
        A = vec.fit_transform(s1c["name_nospace"].to_list()).T.tocsr()
        s1_ids = s1c["entity_id"].to_numpy()
        q_ids = qc["entity_id"].to_numpy()
        frames = []
        for start in range(0, qc.shape[0], a.chunk):
            B = vec.transform(qc["name_nospace"].slice(start, a.chunk).to_list())
            C = sp_matmul_topn(B, A, top_n=a.top_k, threshold=0.3, sort=True, n_threads=a.threads).tocsr()
            frames.append(csr_to_frame(C, "cos_name", start))
        f = pl.concat(frames)
        f = f.with_columns(pl.Series("q_id", q_ids[f["q"].to_numpy()]), pl.Series("s1_id", s1_ids[f["s1"].to_numpy()]),
                           pl.lit(ctry).alias("country"), pl.lit(0.0, dtype=pl.Float32).alias("cos_joint"),
                           pl.lit(0.0, dtype=pl.Float32).alias("cos_jskel"), pl.lit(0.0, dtype=pl.Float32).alias("cos_fullskel"))
        f = f.with_columns((0.25 * pl.col("cos_name")).alias("h"), pl.lit(99, dtype=pl.UInt32).alias("h_rank"),
                           pl.lit(1.0, dtype=pl.Float32).alias("h_gap")).drop("q", "s1")
        # drop pairs already produced by the main blockers
        existing = pl.concat([pl.read_parquet(p, columns=["q_id", "s1_id"]) for p in
                              sorted(glob.glob(os.path.join(out_dir, f"{ctry}_[0-9]*.parquet")))])
        f = f.join(existing, on=["q_id", "s1_id"], how="anti")
        f = f.select("q_id", "s1_id", "country", "cos_joint", "cos_jskel", "cos_fullskel", "cos_name", "h", "h_rank", "h_gap")
        f.write_parquet(os.path.join(out_dir, f"{ctry}_extra.parquet"))
        print(f"[{ctry}] hard={qc.shape[0]} extra pairs={f.shape[0]} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
