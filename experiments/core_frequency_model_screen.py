"""Entity-disjoint development screen for a name-frequency decision feature.

The exposed first fresh 10k is development data here. No sealed truth is read.
The two-fold result guides selection but is not an unbiased final estimate.
"""

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np
from xgboost import XGBClassifier
from core_frequency_features import MODEL_PARAMS, load


def read_rows(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def metrics(chosen: np.ndarray, labels: np.ndarray, group: np.ndarray,
            truth_count: np.ndarray, country: np.ndarray) -> dict:
    n = len(truth_count)
    tp = np.bincount(group[chosen & labels], minlength=n)
    fp = np.bincount(group[chosen & ~labels], minlength=n)
    fn = truth_count - tp
    score = np.where(truth_count == 0, (fp == 0).astype(float),
                     1.25 * tp / np.maximum(tp + .25 * truth_count + fp, 1e-12))
    def part(mask: np.ndarray) -> dict:
        return {
            "s1": int(mask.sum()), "macro_f05": float(score[mask].mean()),
            "tp": int(tp[mask].sum()), "fp": int(fp[mask].sum()),
            "fn": int(fn[mask].sum()),
        }
    return {
        "overall": part(np.ones(n, dtype=bool)),
        "India": part(country == "India"),
        "US": part(country == "US"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--truth", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    data = load(args.source1, args.pair_scores, args.core_cache)
    source = data["sources"]
    truth = read_rows(args.truth)
    if len(source) != len(truth) or len(source) != 10_000:
        raise ValueError("Expected aligned, exposed first fresh 10k")
    s1_ids = [row["entity_id"] for row in source]
    if s1_ids != [row["source1_entity_id"] for row in truth]:
        raise ValueError("Source/truth order differs")
    frame = data["frame"]
    group = data["group"]
    truth_sets = [set(filter(None, row["matched_entity_ids"].split(","))) for row in truth]
    labels = np.array([mid in truth_sets[i] for mid, i in zip(frame["target_id"], group)], dtype=bool)
    truth_count = np.array([len(values) for values in truth_sets], dtype=np.int32)
    country = data["country"]
    probabilities = data["frozen_probability"]
    baseline = metrics(probabilities >= .74, labels, group, truth_count, country)
    if abs(baseline["overall"]["macro_f05"] - 0.9841388449392869) > 1e-10:
        raise ValueError("Frozen score does not reproduce the saved baseline")
    feature_sets = data["feature_matrices"]
    folds = np.array([
        hashlib.sha256(sid.encode()).digest()[0] % 2 for sid in s1_ids
    ], dtype=np.int8)
    result = {
        "scope": "exposed fresh-v1 10k, 442904-target reduced pool; entity-disjoint two-fold development screen",
        "pair_count": len(frame), "positive_candidate_pairs": int(labels.sum()),
        "baseline": baseline, "configurations": {},
    }
    for name, x in feature_sets.items():
        crossfit = np.empty(len(frame), dtype=np.float32)
        for fold in (0, 1):
            train = folds[group] != fold
            valid = ~train
            model = XGBClassifier(**MODEL_PARAMS)
            model.fit(x[train], labels[train].astype(np.int8))
            crossfit[valid] = model.predict_proba(x[valid])[:, 1].astype(np.float32)
        fixed = metrics(crossfit >= .5, labels, group, truth_count, country)
        thresholds = [round(float(v), 3) for v in np.arange(.25, .951, .025)]
        sweep = [
            {"threshold": threshold,
             "macro_f05": metrics(crossfit >= threshold, labels, group, truth_count, country)["overall"]["macro_f05"]}
            for threshold in thresholds
        ]
        best = max(sweep, key=lambda row: row["macro_f05"])
        best_metrics = metrics(crossfit >= best["threshold"], labels, group, truth_count, country)
        result["configurations"][name] = {
            "features": x.shape[1], "fixed_threshold_0p5": fixed,
            "best_exposed_threshold": best["threshold"], "best_exposed_metrics": best_metrics,
            "threshold_sweep": sweep,
        }
        print(name, "fixed", fixed["overall"]["macro_f05"],
              "best_exposed", best_metrics["overall"]["macro_f05"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baseline": baseline,
                      "configurations": {name: {
                          "fixed": value["fixed_threshold_0p5"]["overall"]["macro_f05"],
                          "best_exposed": value["best_exposed_metrics"]["overall"]["macro_f05"],
                      } for name, value in result["configurations"].items()}}, indent=2))


if __name__ == "__main__":
    main()
