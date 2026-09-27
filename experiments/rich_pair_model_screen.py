"""Cross-cohort development screen; neither cohort is still sealed.

Each direction fits on one exposed 10k, calibrates a threshold on entity-
disjoint folds of that same cohort, and evaluates on the other exposed 10k.
Only a later frozen-model run on a new disjoint cohort can confirm a gain.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import time

import numpy as np
from xgboost import XGBClassifier

from core_frequency_features import load
from core_frequency_model_screen import metrics


def load_cohort(root):
    data = load(root / "source1.tsv", root / "frozen_pair_scores.parquet",
                root / "core_freq_full")
    manifest = json.loads((root / "rich_pair_features.json").read_text())
    x = np.load(root / "rich_pair_features.npy", mmap_mode="r")
    if x.shape != (len(data["frame"]), len(manifest["feature_names"])):
        raise ValueError("Feature matrix/manifest mismatch")
    with (root / "truth.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [r["source1_entity_id"] for r in rows] != data["source_ids"]:
        raise ValueError("Truth/source order mismatch")
    sets = [set(filter(None, r["matched_entity_ids"].split(","))) for r in rows]
    group = data["group"]
    labels = np.array([mid in sets[i] for mid, i in
                       zip(data["frame"]["target_id"], group)], dtype=bool)
    truth_count = np.array([len(s) for s in sets], dtype=np.int32)
    country = data["country"]
    expected = {"fresh_10k_v1": .9841388449392869,
                "fresh_10k_v2": .9831077662024144,
                "fresh_10k_final": .9835376330949781}[root.name]
    baseline = metrics(data["frozen_probability"] >= .74, labels, group,
                       truth_count, country)
    if abs(baseline["overall"]["macro_f05"] - expected) > 1e-10:
        raise ValueError("Frozen baseline no longer reproduces saved metric")
    return {"x": x, "names": manifest["feature_names"], "labels": labels,
            "group": group, "truth_count": truth_count, "country": country,
            "source_ids": data["source_ids"], "baseline": baseline}


def fit(x, labels):
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=.045,
        min_child_weight=16, subsample=.85, colsample_bytree=.85,
        reg_lambda=6, tree_method="hist", n_jobs=4,
        random_state=20260927, eval_metric="logloss")
    model.fit(x, labels)
    return model


def choose_threshold(train, columns):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in train["source_ids"]], dtype=np.int8)
    x = train["x"][:, columns]
    crossfit = np.empty(len(train["labels"]), dtype=np.float32)
    for holdout in (0, 1):
        fit_mask = fold[train["group"]] != holdout
        model = fit(x[fit_mask], train["labels"][fit_mask])
        crossfit[~fit_mask] = model.predict_proba(x[~fit_mask])[:, 1]
    tested = []
    for threshold in np.arange(.4, .951, .025):
        m = metrics(crossfit >= threshold, train["labels"], train["group"],
                    train["truth_count"], train["country"])
        tested.append((m["overall"]["macro_f05"], float(round(threshold, 3)), m))
    _, selected, result = max(tested, key=lambda v: v[0])
    return selected, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    cohorts = {name: load_cohort(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    names = cohorts["fresh_10k_v1"]["names"]
    if names != cohorts["fresh_10k_v2"]["names"]:
        raise ValueError("Feature layouts differ")
    core = [i for i, name in enumerate(names) if name in {
        "frozen_probability", "core_equal", "target_address_empty", "india",
        "log_target_core_frequency", "log_source_core_frequency"}]
    if len(core) != 6:
        raise ValueError("Core feature count mismatch")
    rapid_start = names.index("unicode_core_ratio")
    configurations = {"core6": core,
                      "rich_existing": list(range(rapid_start)),
                      "rich_plus_rapid": list(range(len(names)))}
    result = {"scope": "two exposed 10k reduced-pool cohorts, cross-cohort development only",
              "feature_names": names, "baselines": {
                  name: value["baseline"] for name, value in cohorts.items()},
              "configurations": {}}
    for label, columns in configurations.items():
        result["configurations"][label] = {}
        for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                      ("fresh_10k_v2", "fresh_10k_v1")):
            train = cohorts[train_name]
            test = cohorts[test_name]
            threshold, crossfit = choose_threshold(train, columns)
            model = fit(train["x"][:, columns], train["labels"])
            scores = model.predict_proba(test["x"][:, columns])[:, 1]
            outcome = metrics(scores >= threshold, test["labels"], test["group"],
                              test["truth_count"], test["country"])
            result["configurations"][label][f"{train_name}_to_{test_name}"] = {
                "threshold_from_train_crossfit": threshold,
                "train_crossfit": crossfit, "test": outcome,
            }
            print(label, train_name, "to", test_name, "threshold", threshold,
                  "F0.5", outcome["overall"]["macro_f05"],
                  "India", outcome["India"]["macro_f05"], flush=True)
    result["seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
