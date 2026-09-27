"""Apply a frozen rich reranker without loading truth labels."""

import argparse
import csv
import json
import pathlib
import time

import numpy as np
from xgboost import XGBClassifier

from core_frequency_features import digest, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--features", type=pathlib.Path, required=True)
    p.add_argument("--model-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Use a new output directory")
    started = time.monotonic()
    checkpoint = json.loads((args.model_dir / "manifest.json").read_text())
    if checkpoint["status"] != "frozen_before_fresh_final_truth":
        raise ValueError("Checkpoint status mismatch")
    model_path = args.model_dir / "rich_pair_model.json"
    if checkpoint["model_sha256"] != digest(model_path):
        raise ValueError("Checkpoint model digest mismatch")
    data = load(args.source1, args.pair_scores, args.core_cache)
    export = json.loads(args.features.with_suffix(".json").read_text())
    if export["feature_names"] != checkpoint["feature_names"]:
        raise ValueError("Rich feature layout differs from frozen checkpoint")
    if export["source1_sha256"] != digest(args.source1) or export["pair_scores_sha256"] != digest(args.pair_scores):
        raise ValueError("Rich features belong to another source or pair file")
    if export["core_cache_manifest_sha256"] != digest(args.core_cache / "manifest.json"):
        raise ValueError("Rich features belong to another core cache")
    x = np.load(args.features, mmap_mode="r")
    if x.shape != (len(data["frame"]), len(checkpoint["feature_names"])):
        raise ValueError("Rich feature matrix has wrong shape")
    model = XGBClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(x)[:, 1]
    selected = probabilities >= checkpoint["threshold"]
    chosen = [[] for _ in data["source_ids"]]
    for mid, group, keep in zip(data["frame"]["target_id"], data["group"], selected):
        if keep:
            chosen[group].append(mid)
    args.out_dir.mkdir(parents=True)
    out = args.out_dir / "matching_results.tsv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid, matches in zip(data["source_ids"], chosen):
            writer.writerow([sid, ",".join(matches)])
    meta = {
        "scope": "label-blind frozen rich-pair prediction; no truth read",
        "source1_rows": len(chosen), "candidate_pairs": len(selected),
        "predicted_pairs": int(selected.sum()),
        "threshold": checkpoint["threshold"],
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "rich_features_manifest_sha256": digest(args.features.with_suffix(".json")),
        "matching_sha256": digest(out), "seconds": time.monotonic()-started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
