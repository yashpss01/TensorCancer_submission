"""Screen an inference-available target-source feature on exposed cohorts."""

import argparse
import json
import pathlib
import time

import numpy as np
import polars as pl

from core_frequency_model_screen import metrics
from rich_pair_model_screen import choose_threshold, fit, load_cohort


def source_counts(data, probabilities, threshold, source3):
    chosen = probabilities >= threshold
    labels = data["labels"]
    result = {}
    for name, mask in (("S2", ~source3), ("S3", source3)):
        result[name] = {
            "pairs": int(mask.sum()), "true": int((labels & mask).sum()),
            "tp": int((chosen & labels & mask).sum()),
            "fp": int((chosen & ~labels & mask).sum()),
            "fn": int((~chosen & labels & mask).sum()),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    cohorts = {}
    for name in ("fresh_10k_v1", "fresh_10k_v2"):
        root = args.root / "work" / name
        data = load_cohort(root)
        ids = pl.read_parquet(root / "frozen_pair_scores.parquet")["target_id"]
        source3 = np.asarray(ids.str.starts_with("S3-").to_numpy(), dtype=bool)
        source2 = np.asarray(ids.str.starts_with("S2-").to_numpy(), dtype=bool)
        if len(source3) != len(data["x"]) or not np.all(source2 ^ source3):
            raise ValueError("Target-source IDs do not align with feature rows")
        data["x"] = np.column_stack((data["x"], source3.astype(np.float32)))
        cohorts[name] = (data, source3)
    result = {"scope": "exposed v1/v2 reduced-pool cross-cohort development only",
              "candidate_sets_changed": False, "directions": {}}
    for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                  ("fresh_10k_v2", "fresh_10k_v1")):
        train, _ = cohorts[train_name]
        test, test_source3 = cohorts[test_name]
        direction = {}
        for name, columns in (("rich", list(range(train["x"].shape[1]-1))),
                              ("rich_plus_target_source", list(range(train["x"].shape[1])))):
            threshold, crossfit = choose_threshold(train, columns)
            model = fit(train["x"][:, columns], train["labels"])
            probabilities = model.predict_proba(test["x"][:, columns])[:, 1]
            outcome = metrics(probabilities >= threshold, test["labels"],
                              test["group"], test["truth_count"], test["country"])
            direction[name] = {
                "selected_threshold": threshold,
                "train_crossfit": crossfit,
                "test": outcome,
                "test_target_source_pair_counts": source_counts(test, probabilities,
                                                               threshold, test_source3),
            }
        result["directions"][f"{train_name}_to_{test_name}"] = direction
        print(train_name, "to", test_name,
              {name: {country: round(item["test"][country]["macro_f05"]*100, 4)
                      for country in ("overall", "India", "US")}
               for name, item in direction.items()}, flush=True)
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
