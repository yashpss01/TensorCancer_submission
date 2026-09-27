"""Evaluate completed v2 10k predictions against locally sealed labels.

Do not run until retrieval, frozen scoring and all model/threshold choices are
fixed. The resulting score is for a fresh labeled training batch, not Portal
test performance or France performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path: Path, id_column: str, values_column: str | None = None) -> dict:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            key = row[id_column]
            if key in result:
                raise ValueError(f"Duplicate ID in {path}: {key}")
            if values_column is None:
                result[key] = row
            else:
                values = row[values_column].split(",") if row[values_column] else []
                if len(values) != len(set(values)):
                    raise ValueError(f"Duplicate match/candidate ID for {key} in {path}")
                result[key] = set(values)
    return result


def evaluate(source: Path, truth: Path, predictions: Path, candidates: Path) -> dict:
    entities = rows(source, "entity_id")
    true_sets = rows(truth, "source1_entity_id", "matched_entity_ids")
    pred_sets = rows(predictions, "source1_entity_id", "matched_entity_ids")
    candidate_sets = rows(candidates, "source1_entity_id", "candidate_entity_ids")
    expected = set(entities)
    if any(set(group) != expected for group in (true_sets, pred_sets, candidate_sets)):
        raise ValueError("Source, truth, predictions and candidates must cover the same S1 IDs")
    results = []
    for sid, record in entities.items():
        truth_set = true_sets[sid]
        prediction = pred_sets[sid]
        candidate_set = candidate_sets[sid]
        if not prediction <= candidate_set:
            raise ValueError(f"Predicted ID outside candidates for {sid}")
        tp = len(prediction & truth_set)
        fp = len(prediction - truth_set)
        fn = len(truth_set - prediction)
        denom = 1.25 * tp + fp + 0.25 * fn
        f05 = 1.25 * tp / denom if denom else 1.0
        results.append({"country": record["country"], "tp": tp, "fp": fp, "fn": fn,
                        "f0_5": f05, "singleton": not truth_set,
                        "singleton_correct": not truth_set and not prediction,
                        "candidate_count": len(candidate_set)})

    def summarize(group: list[dict]) -> dict:
        tp = sum(item["tp"] for item in group)
        fp = sum(item["fp"] for item in group)
        fn = sum(item["fn"] for item in group)
        singleton = sum(item["singleton"] for item in group)
        counts = np.array([item["candidate_count"] for item in group])
        return {"s1_count": len(group), "tp": tp, "fp": fp, "fn": fn,
                "macro_f0_5": float(np.mean([item["f0_5"] for item in group])),
                "micro_precision": tp / (tp + fp) if tp + fp else None,
                "micro_recall": tp / (tp + fn) if tp + fn else None,
                "singleton_count": singleton,
                "singleton_accuracy": sum(item["singleton_correct"] for item in group) / singleton if singleton else None,
                "candidates_per_s1_mean": float(counts.mean()),
                "candidates_per_s1_p95": float(np.percentile(counts, 95)),
                "candidates_per_s1_max": int(counts.max())}

    return {"scope": "fresh labeled S1 10k, full training target universe, frozen model on new candidate distribution",
            "overall": summarize(results),
            "countries": {country: summarize([item for item in results if item["country"] == country])
                          for country in sorted({item["country"] for item in results})}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.source, args.truth, args.predictions, args.candidates)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
