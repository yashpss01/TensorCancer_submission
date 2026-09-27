"""Label-blind target-owner conflict resolution on exposed cohorts only.

Source 1 is deduplicated. For a target accepted by more than one sampled S1,
keep the highest-scoring owner, breaking ties by S1 ID. This development
screen does not alter the packaged inference or open a new holdout.
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def entity_scores(chosen, data):
    labels, group, truth_count = data["labels"], data["group"], data["truth_count"]
    count = len(truth_count)
    tp = np.bincount(group[chosen & labels], minlength=count)
    fp = np.bincount(group[chosen & ~labels], minlength=count)
    return np.where(truth_count == 0, (fp == 0).astype(float),
                    1.25 * tp / np.maximum(tp + .25 * truth_count + fp, 1e-12))


def interval(base, changed, mask):
    gain = (changed-base)[mask]
    rng = np.random.default_rng(20260927)
    samples = [float(gain[rng.integers(0, len(gain), len(gain))].mean())
               for _ in range(2000)]
    return {"gain_percentage_points": float(100*gain.mean()),
            "paired_entity_bootstrap_95_pp":
                [float(100*v) for v in np.quantile(samples, [.025, .975])]}


def screen(data, probability, target_ids, threshold):
    labels, group = data["labels"], data["group"]
    if len(probability) != len(target_ids) or len(probability) != len(labels):
        raise ValueError("Pair scores, candidates, and truth differ")
    accepted = probability >= threshold
    owners = collections.defaultdict(list)
    for pair_index in np.flatnonzero(accepted):
        owners[target_ids[pair_index]].append(int(pair_index))
    conflicts = {mid: js for mid, js in owners.items() if len(js) > 1}
    resolved = accepted.copy()
    removed = []
    for indices in conflicts.values():
        ranked = sorted(indices, key=lambda j: (-float(probability[j]),
                                                 data["source_ids"][group[j]]))
        for pair_index in ranked[1:]:
            resolved[pair_index] = False
            removed.append(pair_index)
    base = metrics(accepted, labels, group, data["truth_count"], data["country"])
    changed = metrics(resolved, labels, group, data["truth_count"], data["country"])
    scores_before = entity_scores(accepted, data)
    scores_after = entity_scores(resolved, data)
    countries = {name: interval(scores_before, scores_after, mask)
                 for name, mask in (("overall", np.ones(len(data["source_ids"]), bool)),
                                    ("India", data["country"] == "India"),
                                    ("US", data["country"] == "US"))}
    # Truth is used only here, after label-blind conflict resolution.
    true_owner_count = collections.Counter(target_ids[j]
                                           for j in np.flatnonzero(labels))
    return {
        "candidate_pairs": len(probability), "threshold": threshold,
        "true_candidate_targets_with_multiple_s1_owners":
            sum(count > 1 for count in true_owner_count.values()),
        "predicted_target_conflicts": len(conflicts),
        "removed_predictions": len(removed),
        "removed_true": int(labels[removed].sum()),
        "removed_false": int((~labels[removed]).sum()),
        "baseline": base, "resolved": changed, "paired_gains": countries,
    }


def verify_training_ownership(path):
    seen = set()
    repeated = set()
    rows = links = 0
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            rows += 1
            ids = set(filter(None, row["matched_entity_ids"].split(",")))
            links += len(ids)
            for target_id in ids:
                if target_id in seen:
                    repeated.add(target_id)
                else:
                    seen.add(target_id)
    return {"ground_truth_sha256": digest.hexdigest(),
            "source1_rows": rows, "true_links": links,
            "distinct_target_ids": len(seen),
            "targets_with_multiple_s1_owners": len(repeated)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--full-training-truth", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path")
    data = {name: load_cohort(args.root / "work" / f"fresh_10k_{name}")
            for name in ("v1", "v2", "final")}
    frozen = XGBClassifier()
    frozen.load_model(args.root / "experiments/checkpoints/rich_pair_v1v2/rich_pair_model.json")
    comparisons = (("v1", "v2", .7), ("v2", "v1", .825),
                   ("v1+v2 frozen", "final", .775))
    result = {"scope": "three exposed reduced-pool 10k cohorts; development screen only",
              "rule": "each accepted target assigned to highest rich probability S1; tie by S1 ID",
              "training_ownership": verify_training_ownership(args.full_training_truth),
              "directions": {}}
    for train_name, test_name, threshold in comparisons:
        test = data[test_name]
        model = frozen if test_name == "final" else fit(data[train_name]["x"],
                                                        data[train_name]["labels"])
        probability = model.predict_proba(test["x"])[:, 1]
        target_ids = pl.read_parquet(args.root / "work" / f"fresh_10k_{test_name}" /
                                     "frozen_pair_scores.parquet")["target_id"].to_list()
        part = screen(test, probability, target_ids, threshold)
        result["directions"][f"{train_name}_to_{test_name}"] = part
        print(test_name, "conflicts", part["predicted_target_conflicts"],
              "removed true/false", part["removed_true"], part["removed_false"],
              "F0.5", part["baseline"]["overall"]["macro_f05"],
              "to", part["resolved"]["overall"]["macro_f05"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
