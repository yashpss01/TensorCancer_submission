"""Apply a previously frozen frequency calibrator without reading labels."""

import argparse
import csv
import json
import pathlib

import numpy as np
from xgboost import XGBClassifier

from core_frequency_features import FEATURE_NAMES, digest, load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--model-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Use a new output directory")
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    model_path = args.model_dir / "core_frequency_model.json"
    if manifest["status"] != "frozen_before_fresh_v2_truth":
        raise ValueError("Model was not frozen for fresh validation")
    if manifest["feature_names"] != list(FEATURE_NAMES) or manifest["threshold"] != .7:
        raise ValueError("Unexpected feature or threshold contract")
    if manifest["model_sha256"] != digest(model_path):
        raise ValueError("Frozen model digest mismatch")
    data = load(args.source1, args.pair_scores, args.core_cache)
    x = data["feature_matrices"]["context_plus_core_frequency"]
    model = XGBClassifier()
    model.load_model(model_path)
    scores = model.predict_proba(x)[:, 1].astype(np.float32)
    if len(scores) != len(data["group"]) or not np.isfinite(scores).all():
        raise ValueError("Invalid prediction probabilities")
    chosen = [[] for _ in data["source_ids"]]
    for target, group, score in zip(data["frame"]["target_id"], data["group"], scores):
        if score >= .7:
            chosen[group].append(target)
    args.out_dir.mkdir(parents=True)
    out = args.out_dir / "matching_results.tsv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid, matches in zip(data["source_ids"], chosen):
            writer.writerow([sid, ",".join(matches)])
    result = {
        "source1_rows": len(chosen), "candidate_pairs": len(scores),
        "predicted_pairs": sum(map(len, chosen)),
        "selection": "frozen core-frequency XGBoost probability >= 0.70",
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "matching_sha256": digest(out),
        "pair_scores_sha256": digest(args.pair_scores),
        "core_cache_manifest_sha256": digest(args.core_cache / "manifest.json"),
        "scope": "label-blind prediction; no quality score here",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
