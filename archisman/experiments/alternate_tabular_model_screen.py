"""Exposed-cohort LightGBM/CatBoost screen on rich + graph evidence."""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from core_frequency_model_screen import metrics
from rich_pair_model_screen import load_cohort


def factory(name):
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=450, num_leaves=31, max_depth=-1,
                              learning_rate=.04, min_child_samples=30,
                              colsample_bytree=.85, subsample=.85,
                              reg_lambda=5, n_jobs=4, verbosity=-1,
                              random_state=20260927)
    return CatBoostClassifier(iterations=550, depth=6, learning_rate=.05,
                              l2_leaf_reg=5, loss_function="Logloss",
                              thread_count=4, verbose=False,
                              random_seed=20260927)


def choose(train, name):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in train["source_ids"]], dtype=np.int8)
    crossfit = np.empty(len(train["labels"]), dtype=np.float32)
    for holdout in (0, 1):
        fit_mask = fold[train["group"]] != holdout
        model = factory(name)
        model.fit(train["x"][fit_mask], train["labels"][fit_mask])
        crossfit[~fit_mask] = model.predict_proba(train["x"][~fit_mask])[:, 1]
    grid = []
    for threshold in np.arange(.4, .951, .025):
        score = metrics(crossfit >= threshold, train["labels"], train["group"],
                        train["truth_count"], train["country"])
        grid.append((score["overall"]["macro_f05"], float(round(threshold, 3)), score))
    _, threshold, score = max(grid, key=lambda row: row[0])
    return threshold, score


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
        graph = np.load(root / "cross_source_anchor.npy", mmap_mode="r")
        if len(graph) != len(data["x"]):
            raise ValueError("Anchor/rich pair mismatch")
        data["x"] = np.column_stack((data["x"], graph)).astype(np.float32)
        cohorts[name] = data
    result = {"scope": "exposed v1/v2 cross-cohort development only; frozen candidates",
              "configurations": {}}
    for name in ("lightgbm", "catboost"):
        result["configurations"][name] = {}
        for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                      ("fresh_10k_v2", "fresh_10k_v1")):
            train, test = cohorts[train_name], cohorts[test_name]
            threshold, dev = choose(train, name)
            model = factory(name)
            model.fit(train["x"], train["labels"])
            prob = model.predict_proba(test["x"])[:, 1]
            score = metrics(prob >= threshold, test["labels"], test["group"],
                            test["truth_count"], test["country"])
            result["configurations"][name][f"{train_name}_to_{test_name}"] = {
                "selected_threshold": threshold, "train_crossfit": dev, "test": score,
            }
            print(name, train_name, "to", test_name, "threshold", threshold,
                  "F0.5", score["overall"]["macro_f05"],
                  "India", score["India"]["macro_f05"], flush=True)
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
