"""Reverse batched TF-IDF retrieval against the complete Source-1 universe.

The four retrieval views and trimming rule are adapted from the Akanksha
branch. For a sealed S1 evaluation sample, the entire S1 index competes in
each sparse top-k query; selected IDs are filtered only after global ranking.
This module is label-blind and emits retrieval candidates, not match decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

TOKEN_RE = r"(?u)\b\w+\b"
VOWELS = re.compile(r"[aeiouyhw]")
REPEATED = re.compile(r"(.)\1+")
VIEWS = (
    ("d_joint", "cos_joint", 30, 0.05, (1, 2)),
    ("d_jskel", "cos_jskel", 15, 0.05, (1, 1)),
    ("d_fullskel", "cos_fullskel", 10, 0.05, (1, 1)),
    ("d_name", "cos_name", 20, 0.5, (1, 1)),
)
NORMALIZED_COLUMNS = (
    "entity_id", "country", "name_core", "name_sorted", "addr_full",
    "addr_alpha", "addr_nums", "addr_first_num",
)


def skeleton(value: str) -> str:
    value = value.replace("ph", "f").replace("v", "w").replace("z", "j")
    value = value.replace("q", "k").replace("c", "k").replace("x", "ks")
    value = value.replace("th", "t").replace("dh", "d").replace("sh", "s")
    return REPEATED.sub(r"\1", VOWELS.sub("", value))


def documents(frame: pl.DataFrame) -> pl.DataFrame:
    name = frame["name_core"].to_list()
    address = frame["addr_alpha"].to_list()
    skel_name = [" ".join(token if token.isdigit() else skeleton(token)
                          for token in value.split()) for value in name]
    skel_address = [" ".join(token if token.isdigit() else skeleton(token)
                             for token in value.split()) for value in address]
    return frame.with_columns(pl.Series("skel_name", skel_name),
                              pl.Series("skel_address", skel_address)).with_columns(
        (pl.col("name_core") + " " + pl.col("addr_full")).alias("d_joint"),
        (pl.col("skel_name") + " " + pl.col("addr_full")).alias("d_jskel"),
        (pl.col("skel_name") + " " + pl.col("skel_address") + " " + pl.col("addr_nums")).alias("d_fullskel"),
        pl.col("name_core").alias("d_name"),
    )


def read_country(paths: list[Path], country: str) -> pl.DataFrame:
    frames = [pl.read_parquet(path, columns=NORMALIZED_COLUMNS).filter(pl.col("country") == country)
              for path in paths]
    return pl.concat(frames) if frames else pl.DataFrame()


class SparseView:
    def __init__(self, docs: list[str], top_k: int, max_df: float,
                 ngrams: tuple[int, int], threads: int):
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=ngrams,
            max_df=max_df if len(docs) >= 40 else 1.0,
            sublinear_tf=True, dtype=np.float32, token_pattern=TOKEN_RE,
            min_df=2 if ngrams[1] > 1 else 1,
        )
        self.index = self.vectorizer.fit_transform(docs).T.tocsr()
        self.top_k = top_k
        self.threads = threads

    def query(self, docs: list[str]) -> sparse.csr_matrix:
        q = self.vectorizer.transform(docs)
        return sp_matmul_topn(q, self.index, top_n=self.top_k,
                             threshold=0.01, sort=True, n_threads=self.threads).tocsr()


def csr_frame(matrix: sparse.csr_matrix, score_column: str) -> pl.DataFrame:
    row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), np.diff(matrix.indptr))
    return pl.DataFrame({"q": row_ids, "s1": matrix.indices.astype(np.int32),
                         score_column: matrix.data.astype(np.float32)})


def candidate_chunk(query: pl.DataFrame, index: pl.DataFrame,
                    views: dict[str, tuple[SparseView, str]],
                    selected_positions: np.ndarray | None, keep: int, max_rank: int) -> pl.DataFrame:
    if query.is_empty():
        return pl.DataFrame()
    query = documents(query)
    frames = [csr_frame(model.query(query[doc_col].to_list()), score_col)
              for score_col, (model, doc_col) in views.items()]
    union = frames[0]
    for frame in frames[1:]:
        union = union.join(frame, on=["q", "s1"], how="full", coalesce=True)
    if union.is_empty():
        return union
    union = union.with_columns(pl.col(score).fill_null(0.0) for score in views)
    union = union.with_columns(
        (pl.max_horizontal("cos_joint", "cos_jskel", "cos_fullskel")
         + 0.25 * pl.col("cos_name")).alias("h")
    ).with_columns(
        pl.col("h").rank(method="ordinal", descending=True).over("q").alias("h_rank"),
        (pl.col("h").max().over("q") - pl.col("h")).alias("h_gap"),
    ).filter(pl.col("h_rank") <= max_rank)
    # Filter after ranking against every S1 competitor. This preserves the
    # selected entities' realistic rank while avoiding expensive side joins.
    if selected_positions is not None:
        union = union.filter(pl.col("s1").is_in(selected_positions))
    if union.is_empty():
        return union
    side = ["name_core", "name_sorted", "addr_first_num", "addr_nums", "addr_alpha"]
    s1_side = index.select(side).with_row_index("s1")
    if selected_positions is not None:
        s1_side = s1_side.filter(pl.col("s1").is_in(selected_positions))
    s1_side = s1_side.rename({x: x + "_a" for x in side})
    q_side = query.select(side).with_row_index("q").rename({x: x + "_b" for x in side})
    union = union.join(s1_side, on="s1", how="left").join(q_side, on="q", how="left")
    tokens = lambda name: pl.col(name).str.split(" ").list.eval(
        pl.element().filter(pl.element() != ""))
    union = union.with_columns(
        (pl.col("name_core_a") == pl.col("name_core_b")).alias("name_eq"),
        (pl.col("name_sorted_a") == pl.col("name_sorted_b")).alias("sorted_eq"),
        ((pl.col("addr_first_num_a") != "")
         & (pl.col("addr_first_num_a") == pl.col("addr_first_num_b"))).alias("fn_eq"),
        (tokens("addr_nums_a").list.set_intersection(tokens("addr_nums_b")).list.len() > 0).alias("num_common"),
        (tokens("name_core_a").list.set_intersection(tokens("name_core_b")).list.len()
         / pl.max_horizontal(tokens("name_core_a").list.set_union(tokens("name_core_b")).list.len(), 1)).alias("nj"),
        (tokens("addr_alpha_a").list.set_intersection(tokens("addr_alpha_b")).list.len()
         / pl.max_horizontal(tokens("addr_alpha_a").list.set_union(tokens("addr_alpha_b")).list.len(), 1)).alias("aj"),
    )
    rule = ((pl.col("h_rank") <= keep) | pl.col("name_eq") | pl.col("sorted_eq") | pl.col("fn_eq")
            | ((pl.col("nj") >= 0.5) & (pl.col("num_common") | (pl.col("aj") >= 0.3))))
    union = union.filter(rule)
    if union.is_empty():
        return union
    return union.with_columns(
        pl.Series("q_id", query["entity_id"].to_numpy()[union["q"].to_numpy()]),
        pl.Series("s1_id", index["entity_id"].to_numpy()[union["s1"].to_numpy()]),
    ).select("q_id", "s1_id", *views.keys(), "h", "h_rank", "h_gap")


def retrieve(index_files: list[Path], query_files: list[Path], out_dir: Path,
             country: str, selected_ids: set[str] | None = None,
             rows_per_chunk: int = 50_000, threads: int = 4,
             keep: int = 4, max_rank: int = 40) -> dict:
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = {"country": country, "index_files": [str(path) for path in index_files],
                "query_files": [str(path) for path in query_files],
                "selected_ids": sorted(selected_ids) if selected_ids is not None else None,
                "rows_per_chunk": rows_per_chunk, "keep": keep, "max_rank": max_rank}
    input_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("input_hash") != input_hash:
            raise ValueError(f"Completed shard has different inputs: {out_dir}")
        return manifest
    # An interrupted run has no completion manifest. Recompute its unique shard
    # from the beginning; no other worker writes to this directory.
    for partial in out_dir.glob("candidates-*.parquet"):
        partial.unlink()
    index = documents(read_country(index_files, country))
    if index.is_empty():
        raise ValueError(f"No Source-1 index rows for {country}")
    views = {score: (SparseView(index[doc].to_list(), top_k, max_df, ngrams, threads), doc)
             for doc, score, top_k, max_df, ngrams in VIEWS}
    selected_positions = None if selected_ids is None else np.flatnonzero(
        np.isin(index["entity_id"].to_numpy(), list(selected_ids))).astype(np.int32)
    print(f"{country}: indexed {index.height:,} S1 rows in {time.monotonic()-started:.1f}s", flush=True)
    query_count = 0
    candidate_count = 0
    part = 0
    for query_file in query_files:
        frame = pl.read_parquet(query_file, columns=NORMALIZED_COLUMNS).filter(pl.col("country") == country)
        for offset in range(0, frame.height, rows_per_chunk):
            batch = frame.slice(offset, rows_per_chunk)
            found = candidate_chunk(batch, index, views, selected_positions, keep, max_rank)
            if not found.is_empty():
                destination = out_dir / f"candidates-{part:05d}.parquet"
                temporary = destination.with_suffix(".parquet.partial")
                found.write_parquet(temporary, compression="zstd")
                temporary.replace(destination)
                candidate_count += found.height
            query_count += batch.height
            part += 1
            print(f"{country}: queried {query_count:,}, retained {candidate_count:,}", flush=True)
    manifest = {"country": country, "index_s1": index.height,
                "queries": query_count, "candidates": candidate_count,
                "selected_s1_count": len(selected_ids) if selected_ids is not None else None,
                "seconds": time.monotonic()-started,
                "label_blind": True, "full_s1_competition": True,
                "input_hash": input_hash,
                "views": [view[1] for view in VIEWS], "keep": keep, "max_rank": max_rank}
    temporary_manifest = out_dir / "manifest.json.partial"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--selected-ids", type=Path)
    parser.add_argument("--rows-per-chunk", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=40)
    args = parser.parse_args()
    selected = None
    if args.selected_ids:
        selected = set(pl.read_csv(args.selected_ids, separator="\t")["entity_id"].to_list())
    index_files = sorted(args.index_dir.glob("part-*.parquet"))
    query_files = sorted(args.query_dir.glob("part-*.parquet"))
    if not index_files or not query_files:
        parser.error("Index and query directories must contain Parquet parts")
    print(json.dumps(retrieve(index_files, query_files, args.out_dir, args.country,
                              selected, args.rows_per_chunk, args.threads,
                              args.keep, args.max_rank), indent=2))


if __name__ == "__main__":
    main()
