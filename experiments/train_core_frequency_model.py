"""Freeze the selected frequency calibrator before opening fresh validation."""

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np
from xgboost import XGBClassifier

from core_frequency_features import FEATURE_NAMES, MODEL_PARAMS, digest, load


def truth_sets(path: pathlib.Path, source_ids: list[str]) -> list[set[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [row["source1_entity_id"] for row in rows] != source_ids:
        raise ValueError("Source/truth order mismatch")
    return [set(filter(None, row["matched_entity_ids"].split(","))) for row in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--truth", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--screen-report", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Use a new model directory")
    screen = json.loads(args.screen_report.read_text())
    config = screen["configurations"]["context_plus_core_frequency"]
    threshold = config["best_exposed_threshold"]
    if threshold != .7 or screen["pair_count"] != 503_552:
        raise ValueError("Unexpected selected screen configuration")
    data = load(args.source1, args.pair_scores, args.core_cache)
    if len(data["source_ids"]) != 10_000:
        raise ValueError("Expected exposed first fresh 10k")
    truth = truth_sets(args.truth, data["source_ids"])
    labels = np.array([
        target in truth[group] for target, group in
        zip(data["frame"]["target_id"], data["group"])
    ], dtype=np.int8)
    x = data["feature_matrices"]["context_plus_core_frequency"]
    if len(labels) != 503_552 or int(labels.sum()) != 34_379:
        raise ValueError("Candidate labels differ from the frozen baseline")
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(x, labels)
    args.out_dir.mkdir(parents=True)
    model_path = args.out_dir / "core_frequency_model.json"
    model.save_model(model_path)
    manifest = {
        "status": "frozen_before_fresh_v2_truth",
        "selection_scope": "exposed fresh-v1 10k, two entity-disjoint development folds",
        "threshold": threshold, "feature_names": FEATURE_NAMES,
        "model_params": MODEL_PARAMS, "training_s1": len(data["source_ids"]),
        "training_pairs": len(labels), "positive_pairs": int(labels.sum()),
        "source1_sha256": digest(args.source1),
        "truth_sha256": digest(args.truth),
        "pair_scores_sha256": digest(args.pair_scores),
        "core_cache_manifest_sha256": digest(args.core_cache / "manifest.json"),
        "screen_report_sha256": digest(args.screen_report),
        "model_sha256": digest(model_path),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
