"""Step 2: candidate generation (blocking).

For every country label present in Source 1 we build several word-level TF-IDF
indexes over the Source-1 records and, for every Source-2/3 record of the same
country, retrieve the top-k most similar Source-1 records from each index with a
sparse matrix product (sparse_dot_topn).  The union of the retrievals is trimmed
with a transparent rule: keep the top ``--keep`` by heuristic score (max of the
blocker cosines + a name bonus) plus any candidate (within the top ``--max-rank``)
that has an identical normalised name, the same leading house number, or a name
token Jaccard >= 0.5 together with a shared number / address-token overlap.

Blockers (all word unigrams, sublinear tf-idf, cosine):
  * joint    : name_core + full address, word uni+bigrams
  * jskel    : phonetic skeleton of name + address (transliteration robust)
  * fullskel : skeleton of name + skeleton of address words + numbers
  * name     : name_core only (rescues empty-address records)

Output: <work>/cands/<split>/<country>_<chunk>.parquet with columns
  q_id, s1_id, cos_joint, cos_jskel, cos_fullskel, cos_name, h, h_rank, h_gap
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from features import skeleton

TOKEN_RE = r"(?u)\b\w+\b"


def skel_doc(s: str) -> str:
    return " ".join(t if t.isdigit() else skeleton(t) for t in s.split())


def build_docs(df: pl.DataFrame) -> pl.DataFrame:
    sk = [skel_doc(x) for x in df["name_core"].to_list()]
    ska = [skel_doc(x) for x in df["addr_alpha"].to_list()]
    return df.with_columns(pl.Series("skel", sk), pl.Series("skela", ska)).with_columns(
        (pl.col("name_core") + " " + pl.col("addr_full")).alias("d_joint"),
        (pl.col("skel") + " " + pl.col("addr_full")).alias("d_jskel"),
        (pl.col("skel") + " " + pl.col("skela") + " " + pl.col("addr_nums")).alias("d_fullskel"),
        pl.col("name_core").alias("d_name"),
    )


BLOCKERS = [
    # (doc column, output column, top_k, max_df, ngram_range)
    ("d_joint", "cos_joint", 30, 0.05, (1, 2)),
    ("d_jskel", "cos_jskel", 15, 0.05, (1, 1)),
    ("d_fullskel", "cos_fullskel", 10, 0.05, (1, 1)),
    ("d_name", "cos_name", 20, 0.5, (1, 1)),
]


MAX_DF_CAP = 30000  # absolute cap on document frequency (bounds cost on very large indexes)


class Blocker:
    def __init__(self, s1_docs: list[str], max_df: float, top_k: int, threads: int = 8, ngram=(1, 1)):
        n = len(s1_docs)
        md = max_df if max_df >= 0.5 else min(max(int(max_df * n), 2), MAX_DF_CAP)
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=ngram, max_df=md, sublinear_tf=True,
                                   dtype=np.float32, token_pattern=TOKEN_RE, min_df=2 if ngram[1] > 1 else 1)
        A = self.vec.fit_transform(s1_docs)
        self.AT = A.T.tocsr()
        self.top_k = top_k
        self.threads = threads

    def query(self, docs: list[str]) -> sp.csr_matrix:
        B = self.vec.transform(docs)
        return sp_matmul_topn(B, self.AT, top_n=self.top_k, threshold=0.01, sort=True, n_threads=self.threads).tocsr()


def csr_to_frame(C: sp.csr_matrix, col: str, q_offset: int) -> pl.DataFrame:
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr)) + q_offset
    return pl.DataFrame({"q": rows, "s1": C.indices.astype(np.int64), col: C.data.astype(np.float32)})


def run_country(country: str, s1: pl.DataFrame, qs: pl.DataFrame, out_dir: str, keep: int, chunk: int, threads: int, max_rank: int = 40):
    t0 = time.time()
    s1 = build_docs(s1)
    qs = build_docs(qs)
    s1_ids = s1["entity_id"].to_numpy()
    q_ids = qs["entity_id"].to_numpy()
    blockers = {}
    for doc_col, out_col, k, max_df, ngram in BLOCKERS:
        blockers[out_col] = (Blocker(s1[doc_col].to_list(), max_df, k, threads, ngram), doc_col)
    print(f"[{country}] S1={len(s1_ids)} queries={len(q_ids)} indexes built in {time.time()-t0:.0f}s", flush=True)
    side_cols = ["name_core", "name_sorted", "addr_first_num", "addr_nums", "addr_alpha"]
    s1_side = s1.select(side_cols).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int64)).rename({c: c + "_a" for c in side_cols})
    q_side = qs.select(side_cols).with_row_index("q").with_columns(pl.col("q").cast(pl.Int64)).rename({c: c + "_b" for c in side_cols})
    n_out = 0
    for ci, start in enumerate(range(0, len(q_ids), chunk)):
        tc = time.time()
        sub = qs.slice(start, chunk)
        frames = []
        for out_col, (bl, doc_col) in blockers.items():
            C = bl.query(sub[doc_col].to_list())
            frames.append(csr_to_frame(C, out_col, start))
        # outer-union of all blockers on (q, s1)
        uni = frames[0]
        for f in frames[1:]:
            uni = uni.join(f, on=["q", "s1"], how="full", coalesce=True)
        uni = uni.with_columns([pl.col(oc).fill_null(0.0) for oc in blockers])
        uni = uni.with_columns(
            (pl.max_horizontal("cos_joint", "cos_jskel", "cos_fullskel") + 0.25 * pl.col("cos_name")).alias("h")
        )
        uni = uni.with_columns(pl.col("h").rank(method="ordinal", descending=True).over("q").alias("h_rank"),
                               (pl.col("h").max().over("q") - pl.col("h")).alias("h_gap"))
        uni = uni.filter(pl.col("h_rank") <= max_rank)
        # ---- transparent rule-based trim ----------------------------------------
        # keep the top ``keep`` by heuristic score, plus any candidate with exact
        # name agreement, same leading house number, or strong token overlap.
        uni = uni.join(s1_side, on="s1", how="left").join(q_side, on="q", how="left")
        tok = lambda col: pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        uni = uni.with_columns(
            (pl.col("name_core_a") == pl.col("name_core_b")).alias("name_eq"),
            (pl.col("name_sorted_a") == pl.col("name_sorted_b")).alias("sorted_eq"),
            ((pl.col("addr_first_num_a") != "") & (pl.col("addr_first_num_a") == pl.col("addr_first_num_b"))).alias("fn_eq"),
            (tok("addr_nums_a").list.set_intersection(tok("addr_nums_b")).list.len() > 0).alias("num_common"),
            (tok("name_core_a").list.set_intersection(tok("name_core_b")).list.len()
             / pl.max_horizontal(tok("name_core_a").list.set_union(tok("name_core_b")).list.len(), 1)).alias("nj"),
            (tok("addr_alpha_a").list.set_intersection(tok("addr_alpha_b")).list.len()
             / pl.max_horizontal(tok("addr_alpha_a").list.set_union(tok("addr_alpha_b")).list.len(), 1)).alias("aj"),
        )
        rule = ((pl.col("h_rank") <= keep) | pl.col("name_eq") | pl.col("sorted_eq") | pl.col("fn_eq")
                | ((pl.col("nj") >= 0.5) & (pl.col("num_common") | (pl.col("aj") >= 0.3))))
        uni = uni.filter(rule)
        uni = uni.with_columns(pl.Series("q_id", q_ids[uni["q"].to_numpy()]),
                               pl.Series("s1_id", s1_ids[uni["s1"].to_numpy()]),
                               pl.lit(country).alias("country")).select(
            "q_id", "s1_id", "country", *[oc for oc in blockers], "h", "h_rank", "h_gap")
        uni.write_parquet(os.path.join(out_dir, f"{country}_{ci:03d}.parquet"))
        n_out += uni.shape[0]
        print(f"[{country}] chunk {ci} queries={sub.shape[0]} pairs={uni.shape[0]} ({time.time()-tc:.0f}s)", flush=True)
    print(f"[{country}] done: {n_out} candidate pairs, {n_out/len(q_ids):.2f} per query, {n_out/len(s1_ids):.2f} per S1, {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--keep", type=int, default=4, help="always keep the top-k by heuristic score")
    ap.add_argument("--max-rank", type=int, default=40, help="never keep candidates ranked below this")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--countries", default="")
    ap.add_argument("--query-frac", type=float, default=1.0, help="subsample queries (for quick experiments)")
    a = ap.parse_args()
    nd = os.path.join(a.work_dir, "norm")
    out_dir = os.path.join(a.work_dir, "cands", a.split)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["entity_id", "country", "name_core", "name_sorted", "addr_full", "addr_alpha", "addr_nums", "addr_first_num"]
    s1 = pl.read_parquet(os.path.join(nd, f"{a.split}_s1.parquet"), columns=cols)
    qs = pl.concat([pl.read_parquet(os.path.join(nd, f"{a.split}_s{i}.parquet"), columns=cols) for i in (2, 3)])
    if a.query_frac < 1.0:
        qs = qs.sample(fraction=a.query_frac, seed=0)
    countries = a.countries.split(",") if a.countries else sorted(s1["country"].unique().to_list())
    for c in countries:
        for f in glob.glob(os.path.join(out_dir, f"{c}_*.parquet")):
            os.remove(f)
        run_country(c, s1.filter(pl.col("country") == c), qs.filter(pl.col("country") == c), out_dir, a.keep, a.chunk, a.threads, a.max_rank)


if __name__ == "__main__":
    main()
