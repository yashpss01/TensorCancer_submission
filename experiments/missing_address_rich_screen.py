"""Cross-cohort development screen for a missing-target-address matcher.

Both v1 and v2 cohorts are exposed. No result from this script is a fresh
confirmation, full-corpus estimate, or Portal result. The candidate sets are
fixed, and the gate uses only an inference-time target-address feature.
"""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
from xgboost import XGBClassifier

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def expert_model():
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=.045,
        min_child_weight=8, subsample=.85, colsample_bytree=.85,
        reg_lambda=6, tree_method="hist", n_jobs=4,
        random_state=20260927, eval_metric="logloss",
    )


def crossfit(train, missing):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in train["source_ids"]], dtype=np.int8)
    global_p = np.empty(len(train["labels"]), dtype=np.float32)
    expert_p = np.full(len(train["labels"]), np.nan, dtype=np.float32)
    for holdout in (0, 1):
        fit_rows = fold[train["group"]] != holdout
        test_rows = ~fit_rows
        model = fit(train["x"][fit_rows], train["labels"][fit_rows])
        global_p[test_rows] = model.predict_proba(train["x"][test_rows])[:, 1]
        expert = expert_model()
        expert.fit(train["x"][fit_rows & missing],
                   train["labels"][fit_rows & missing])
        selected = test_rows & missing
        expert_p[selected] = expert.predict_proba(train["x"][selected])[:, 1]
    if not np.all(np.isfinite(global_p)) or not np.all(np.isfinite(expert_p[missing])):
        raise ValueError("Incomplete out-of-fold probabilities")
    return global_p, expert_p


def evaluate(data, probabilities, threshold):
    return metrics(probabilities >= threshold, data["labels"], data["group"],
                   data["truth_count"], data["country"])


def select(train, missing, global_p, expert_p):
    thresholds = np.arange(.45, .951, .025)
    baseline = max(((evaluate(train, global_p, threshold)["overall"]["macro_f05"],
                     float(round(threshold, 3))) for threshold in thresholds),
                   key=lambda row: row[0])
    # Specialist decisions replace only candidates with an empty target
    # address. Retain a single global threshold for all other candidates.
    configurations = {}
    for label, specialist in (("stratified_threshold", global_p),
                              ("specialist", expert_p)):
        best = (-1.0, None, None)
        for missing_threshold in np.arange(.20, .951, .025):
            chosen = global_p >= baseline[1]
            chosen[missing] = specialist[missing] >= missing_threshold
            result = metrics(chosen, train["labels"], train["group"],
                             train["truth_count"], train["country"])
            row = (result["overall"]["macro_f05"],
                   float(round(missing_threshold, 3)), result)
            if row[0] > best[0]:
                best = row
        configurations[label] = {
            "global_threshold": baseline[1],
            "missing_threshold": best[1],
            "train_crossfit": best[2],
        }
    return {"global_threshold": baseline[1],
            "train_crossfit_global": evaluate(train, global_p, baseline[1]),
            "configurations": configurations}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    cohorts = {name: load_cohort(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    if cohorts["fresh_10k_v1"]["names"] != cohorts["fresh_10k_v2"]["names"]:
        raise ValueError("Rich feature layouts differ")
    address_column = cohorts["fresh_10k_v1"]["names"].index("target_address_empty")
    result = {"scope": "exposed v1/v2 reduced-pool cross-cohort development only",
              "candidate_sets_changed": False, "directions": {}}
    for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                  ("fresh_10k_v2", "fresh_10k_v1")):
        train, test = cohorts[train_name], cohorts[test_name]
        train_missing = train["x"][:, address_column] > .5
        test_missing = test["x"][:, address_column] > .5
        global_oof, expert_oof = crossfit(train, train_missing)
        selection = select(train, train_missing, global_oof, expert_oof)
        global_model = fit(train["x"], train["labels"])
        global_p = global_model.predict_proba(test["x"])[:, 1]
        expert = expert_model()
        expert.fit(train["x"][train_missing], train["labels"][train_missing])
        expert_p = expert.predict_proba(test["x"][test_missing])[:, 1]
        direction = {
            "train_missing_pairs": int(train_missing.sum()),
            "train_missing_positives": int(train["labels"][train_missing].sum()),
            "test_missing_pairs": int(test_missing.sum()),
            "test_missing_positives": int(test["labels"][test_missing].sum()),
            "selection": selection,
            "test_global": evaluate(test, global_p, selection["global_threshold"]),
            "test_configs": {},
        }
        for label, choice in selection["configurations"].items():
            chosen = global_p >= choice["global_threshold"]
            selected = expert_p if label == "specialist" else global_p[test_missing]
            chosen[test_missing] = selected >= choice["missing_threshold"]
            direction["test_configs"][label] = metrics(
                chosen, test["labels"], test["group"],
                test["truth_count"], test["country"])
        result["directions"][f"{train_name}_to_{test_name}"] = direction
        print(train_name, "to", test_name,
              {key: round(value["overall"]["macro_f05"] * 100, 4)
               for key, value in {"global": direction["test_global"],
                                  **direction["test_configs"]}.items()}, flush=True)
    result["seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
