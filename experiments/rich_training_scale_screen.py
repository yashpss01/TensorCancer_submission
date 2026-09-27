"""Exposed-cohort screen for increasing rich matcher training examples.

All three cohorts here are already exposed. A fixed 0.775 threshold keeps the
comparison focused on training volume; this is not a fresh validation claim.
"""

import argparse
import json
import pathlib
import time

import numpy as np

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def score(model, test, threshold):
    probability = model.predict_proba(test["x"])[:, 1]
    return metrics(probability >= threshold, test["labels"], test["group"],
                   test["truth_count"], test["country"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    names = ("fresh_10k_v1", "fresh_10k_v2", "fresh_10k_final")
    cohorts = {name: load_cohort(args.root / "work" / name) for name in names}
    if len({tuple(value["names"]) for value in cohorts.values()}) != 1:
        raise ValueError("Rich feature layouts differ")
    source_sets = [set(value["source_ids"]) for value in cohorts.values()]
    if any(source_sets[i] & source_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Source-1 IDs overlap across cohorts")
    threshold = .775
    result = {"scope": "three already exposed 10k training cohorts with positive-enriched reduced target pools",
              "fixed_threshold": threshold, "directions": {}}
    for test_name, single_name, added_name in (
            ("fresh_10k_v1", "fresh_10k_v2", "fresh_10k_final"),
            ("fresh_10k_v2", "fresh_10k_v1", "fresh_10k_final"),
            ("fresh_10k_final", "fresh_10k_v1", "fresh_10k_v2")):
        test = cohorts[test_name]
        single, added = cohorts[single_name], cohorts[added_name]
        single_model = fit(single["x"], single["labels"])
        single_score = score(single_model, test, threshold)
        combined_x = np.concatenate((single["x"], added["x"]), axis=0)
        combined_y = np.concatenate((single["labels"], added["labels"]))
        pooled_model = fit(combined_x, combined_y)
        pooled_score = score(pooled_model, test, threshold)
        result["directions"][test_name] = {
            "single_train": single_name, "added_train": added_name,
            "single_train_pairs": len(single["labels"]),
            "pooled_train_pairs": len(combined_y),
            "single_train_test_score": single_score,
            "pooled_train_test_score": pooled_score,
        }
        print(test_name,
              {label: {country: round(score[country]["macro_f05"]*100, 4)
                       for country in ("overall", "India", "US")}
               for label, score in (("single", single_score), ("pooled", pooled_score))},
              flush=True)
        del combined_x, combined_y, single_model, pooled_model
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
