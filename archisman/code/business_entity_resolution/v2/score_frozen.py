"""Apply the immutable Archisman XGBoost blend to v2 candidate shards.

This is a compatibility benchmark. It uses the original feature definitions,
model assets, 0.74 decision threshold, and 16-candidate cap. The new retriever
changes the candidate distribution, so the old holdout score does not transfer.
This script never reads ground truth.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import FEATURE_NAMES, features, represent
from inference_features import EXTRA_NAMES, GROUP_NAMES, extra_vector, group_matrix
from infer import load_models


def read_source(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_candidates(root: pathlib.Path, cap: int) -> dict[str, list[str]]:
    paths = sorted(root.glob("*/candidates-*.parquet"))
    if not paths:
        raise FileNotFoundError("No retrieval candidate shards")
    frame = pl.concat([pl.read_parquet(path, columns=["s1_id", "q_id", "h"])
                       for path in paths])
    if frame.unique(["s1_id", "q_id"]).height != frame.height:
        raise ValueError("Duplicate candidate pairs")
    frame = frame.sort(["s1_id", "h", "q_id"], descending=[False, True, False])
    grouped: dict[str, list[str]] = {}
    for sid, mid in frame.select("s1_id", "q_id").iter_rows():
        group = grouped.setdefault(sid, [])
        if len(group) < cap:
            group.append(mid)
    return grouped


def score_batch(queries: list[dict[str, str]], lists: list[list[str]],
                targets: dict[str, tuple[str, str, str]], target_cache: dict,
                models: tuple) -> list[list[str]]:
    first, group, augmented, (nw, aw, default), (cn, ca, cdefault) = models
    pair_features = []
    extra_features = []
    reps = []
    offsets = [0]
    for query, candidate_ids in zip(queries, lists):
        source_rep = represent(query["business_name"], query["business_address"], query["country"])
        current = []
        for mid in candidate_ids:
            if mid not in target_cache:
                target_cache[mid] = represent(*targets[mid])
            target_rep = target_cache[mid]
            pair_features.append(features(source_rep, target_rep, nw, aw, default))
            extra_features.append(extra_vector(source_rep, target_rep, cn, ca, cdefault))
            current.append(target_rep)
        reps.append(current)
        offsets.append(len(pair_features))
    if not pair_features:
        return [[] for _ in queries]
    pair_features = np.asarray(pair_features, dtype=np.float32)
    extra_features = np.asarray(extra_features, dtype=np.float32)
    assert pair_features.shape[1] == len(FEATURE_NAMES)
    assert extra_features.shape[1] == len(EXTRA_NAMES)
    full_features = np.concatenate((pair_features, extra_features), axis=1)
    first_probability = first.predict_proba(pair_features)[:, 1].astype(np.float32)
    context = np.concatenate([
        group_matrix(query, current, first_probability[offsets[i]:offsets[i + 1]], aw, default)
        for i, (query, current) in enumerate(zip(queries, reps))
    ], axis=0)
    assert context.shape == (len(pair_features), len(GROUP_NAMES))
    group_input = np.concatenate((full_features, context[:, 1:]), axis=1)
    probability = (0.4 * group.predict_proba(group_input)[:, 1]
                   + 0.6 * augmented.predict_proba(full_features)[:, 1])
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite model probability")
    return [[mid for mid, p in zip(mids, probability[offsets[i]:offsets[i + 1]]) if p >= 0.74]
            for i, mids in enumerate(lists)]


def run(source1: pathlib.Path, target_records: pathlib.Path,
        candidate_root: pathlib.Path, model_dir: pathlib.Path,
        output_dir: pathlib.Path, cap: int = 16, batch_size: int = 128) -> dict:
    if cap < 1 or batch_size < 1:
        raise ValueError("cap and batch_size must be positive")
    started = time.monotonic()
    source = read_source(source1)
    grouped = load_candidates(candidate_root, cap)
    source_ids = {row["entity_id"] for row in source}
    if len(source_ids) != len(source):
        raise ValueError("Duplicate Source-1 ID")
    if set(grouped) - source_ids:
        raise ValueError("Candidate shards contain an S1 ID outside the selected batch")
    target_frame = pl.read_parquet(target_records)
    if target_frame["entity_id"].n_unique() != target_frame.height:
        raise ValueError("Duplicate target record ID")
    targets = {mid: (name, address, country)
               for mid, name, address, country in target_frame.select(
                   "entity_id", "business_name", "business_address", "country").iter_rows()}
    missing = set(itertools.chain.from_iterable(grouped.values())) - targets.keys()
    if missing:
        raise ValueError(f"{len(missing)} candidate target records unavailable")
    models = load_models(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = output_dir / "matching_results.tsv"
    candidates = output_dir / "candidate_pairs.tsv"
    if prediction.exists() or candidates.exists():
        raise FileExistsError("Output already exists")
    predicted_pairs = 0
    candidate_pairs = 0
    target_cache = {}
    with prediction.open("w", newline="", encoding="utf-8") as pf, \
            candidates.open("w", newline="", encoding="utf-8") as cf:
        pw = csv.writer(pf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")
        pw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        for offset in range(0, len(source), batch_size):
            batch = source[offset:offset + batch_size]
            lists = [grouped.get(row["entity_id"], []) for row in batch]
            predictions = score_batch(batch, lists, targets, target_cache, models)
            for query, candidate_ids, matched_ids in zip(batch, lists, predictions):
                pw.writerow([query["entity_id"], ",".join(matched_ids)])
                cw.writerow([query["entity_id"], ",".join(candidate_ids)])
                predicted_pairs += len(matched_ids)
                candidate_pairs += len(candidate_ids)
            print(f"Scored {min(offset + batch_size, len(source)):,}/{len(source):,} S1", flush=True)
    result = {"source1_rows": len(source), "candidate_pairs_after_cap": candidate_pairs,
              "predicted_pairs": predicted_pairs, "candidate_cap": cap,
              "seconds": time.monotonic() - started,
              "frozen_model": str(model_dir), "label_blind": True}
    (output_dir / "score_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type=pathlib.Path, required=True)
    parser.add_argument("--target-records", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path,
                        default=ROOT / "code/business_entity_resolution/models")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args.source1, args.target_records, args.candidate_root,
                         args.model_dir, args.output_dir, args.cap, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
