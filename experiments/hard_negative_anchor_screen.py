"""Development screen: give a graph reranker more hard negatives per tree.

Both training and test cohorts are already exposed. The holdout cohort here is
entity-disjoint from fitting within a direction, but this is not fresh evidence.
"""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
from xgboost import XGBClassifier

from core_frequency_model_screen import metrics
from rich_pair_model_screen import load_cohort


def model():
    return XGBClassifier(n_estimators=450, max_depth=5, learning_rate=.045,
                         min_child_weight=8, subsample=.85, colsample_bytree=.9,
                         reg_lambda=5, tree_method="hist", n_jobs=4,
                         random_state=20260927, eval_metric="logloss")


def selected_pairs(data, hard_threshold, easy_fraction):
    p0 = data["x"][:, data["names"].index("frozen_probability")]
    easy_draw = np.random.default_rng(20260927).random(len(p0)) < easy_fraction
    return data["labels"] | (p0 >= hard_threshold) | easy_draw


def choose(data, selected):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in data["source_ids"]], dtype=np.int8)
    crossfit = np.empty(len(data["labels"]), dtype=np.float32)
    for heldout in (0, 1):
        fit_mask = selected & (fold[data["group"]] != heldout)
        valid = fold[data["group"]] == heldout
        classifier = model()
        classifier.fit(data["x"][fit_mask], data["labels"][fit_mask])
        crossfit[valid] = classifier.predict_proba(data["x"][valid])[:, 1]
    grid = []
    for threshold in np.arange(.1, .991, .025):
        result = metrics(crossfit >= threshold, data["labels"], data["group"],
                         data["truth_count"], data["country"])
        grid.append((result["overall"]["macro_f05"], float(round(threshold, 3)), result))
    _, threshold, result = max(grid, key=lambda item: item[0])
    return threshold, result


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
        graph_path = root / "cross_source_anchor.npy"
        graph = np.load(graph_path, mmap_mode="r")
        if graph.shape[0] != len(data["x"]):
            raise ValueError("Anchor feature/pair mismatch")
        data["x"] = np.column_stack((data["x"], graph)).astype(np.float32)
        data["names"] = data["names"] + json.loads(graph_path.with_suffix(".json").read_text())["feature_names"]
        cohorts[name] = data
    configs = {"hard_0p01_easy_10pct": (.01, .1),
               "hard_0p001_easy_05pct": (.001, .05)}
    result = {"scope": "two exposed 10k reduced-pool cohorts, cross-cohort development",
              "method": "all positives + frozen-score hard negatives + seeded easy-negative sample",
              "configurations": {}}
    for label, (hard, easy) in configs.items():
        result["configurations"][label] = {}
        for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                      ("fresh_10k_v2", "fresh_10k_v1")):
            train, test = cohorts[train_name], cohorts[test_name]
            selected = selected_pairs(train, hard, easy)
            threshold, crossfit = choose(train, selected)
            classifier = model()
            classifier.fit(train["x"][selected], train["labels"][selected])
            scores = classifier.predict_proba(test["x"])[:, 1]
            outcome = metrics(scores >= threshold, test["labels"], test["group"],
                              test["truth_count"], test["country"])
            result["configurations"][label][f"{train_name}_to_{test_name}"] = {
                "train_selected_pairs": int(selected.sum()),
                "train_selected_negatives": int((selected & ~train["labels"]).sum()),
                "threshold_from_train_crossfit": threshold,
                "train_crossfit": crossfit, "test": outcome,
            }
            print(label, train_name, "to", test_name, "threshold", threshold,
                  "F0.5", outcome["overall"]["macro_f05"],
                  "India", outcome["India"]["macro_f05"], flush=True)
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
