"""Shared, label-blind features for core-name frequency calibration."""

import csv
import hashlib
import json
import pathlib

import numpy as np
import polars as pl

FEATURE_NAMES = (
    "frozen_probability", "core_equal", "target_address_empty", "india",
    "log_target_core_frequency", "log_source_core_frequency",
)
MODEL_PARAMS = {
    "n_estimators": 160, "max_depth": 3, "learning_rate": .05,
    "min_child_weight": 20, "subsample": .85, "colsample_bytree": .9,
    "reg_lambda": 5, "tree_method": "hist", "n_jobs": 4,
    "random_state": 20260927, "eval_metric": "logloss",
}
GROUP_FEATURE_NAMES = FEATURE_NAMES + (
    "log_candidate_count", "group_max_probability",
    "probability_gap_to_best", "log_high_probability_count",
)


def read_source(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def digest(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(source1: pathlib.Path, pair_scores: pathlib.Path,
         core_cache: pathlib.Path) -> dict:
    sources = read_source(source1)
    source_ids = [row["entity_id"] for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate Source-1 ID")
    frequency_meta = json.loads((core_cache / "manifest.json").read_text())
    score_meta = json.loads(pair_scores.with_suffix(".json").read_text())
    if frequency_meta["source1_sha256"] != digest(source1):
        raise ValueError("Core-frequency cache belongs to another Source-1 file")
    if score_meta["candidate_sha256"] != frequency_meta["candidate_sha256"]:
        raise ValueError("Scores and frequency cache use different candidates")
    if score_meta["index_meta_sha256"] != frequency_meta["index_meta_sha256"]:
        raise ValueError("Scores and frequency cache use different target indexes")
    frame = pl.read_parquet(pair_scores).with_row_index("pair_index")
    frame = (frame.join(pl.read_parquet(core_cache / "target_core.parquet"),
                        on="target_id", how="left", maintain_order="left")
                  .join(pl.read_parquet(core_cache / "source_core.parquet"),
                        on="s1_id", how="left", maintain_order="left")
                  .sort("pair_index"))
    if frame.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("Missing pair score or core-frequency record")
    if frame.height != score_meta["candidate_pairs"] or frame.unique(["s1_id", "target_id"]).height != frame.height:
        raise ValueError("Pair-score count or uniqueness mismatch")
    groups = {sid: i for i, sid in enumerate(source_ids)}
    group = np.array([groups[sid] for sid in frame["s1_id"].to_list()], dtype=np.int32)
    if np.any(np.diff(group) < 0):
        raise ValueError("Pair-score groups out of Source-1 order")
    country = np.array([row["country"] for row in sources])
    p = frame["frozen_probability"].to_numpy().astype(np.float32)
    core_equal = (frame["target_core"] == frame["source_core"]).to_numpy().astype(np.float32)
    addr_missing = frame["target_address_empty"].to_numpy().astype(np.float32)
    india = (country[group] == "India").astype(np.float32)
    target_freq = np.log1p(frame["target_core_frequency"].to_numpy()).astype(np.float32)
    source_freq = np.log1p(frame["source_core_frequency"].to_numpy()).astype(np.float32)
    columns = [p, core_equal, addr_missing, india, target_freq, source_freq]
    counts = np.bincount(group, minlength=len(source_ids))
    maximum = np.zeros(len(source_ids), dtype=np.float32)
    np.maximum.at(maximum, group, p)
    high = np.bincount(group[p >= .74], minlength=len(source_ids))
    group_columns = [
        np.log1p(counts[group]).astype(np.float32),
        maximum[group], maximum[group] - p,
        np.log1p(high[group]).astype(np.float32),
    ]
    return {
        "sources": sources, "source_ids": source_ids, "frame": frame,
        "group": group, "country": country, "frozen_probability": p,
        "feature_matrices": {
            "probability_only": np.column_stack(columns[:1]).astype(np.float32),
            "context": np.column_stack(columns[:4]).astype(np.float32),
            "context_plus_core_frequency": np.column_stack(columns).astype(np.float32),
            "context_frequency_group": np.column_stack(columns + group_columns).astype(np.float32),
        },
    }
