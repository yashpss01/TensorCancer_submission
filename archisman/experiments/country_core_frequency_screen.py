"""Pre-fresh-validation country-specialist screen on exposed v1 only."""

import argparse
import hashlib
import json
import pathlib

import numpy as np
from xgboost import XGBClassifier

from core_frequency_features import MODEL_PARAMS, load
from core_frequency_model_screen import metrics, read_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--truth", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    data = load(args.source1, args.pair_scores, args.core_cache)
    source_ids = data["source_ids"]
    truth_rows = read_rows(args.truth)
    if [row["source1_entity_id"] for row in truth_rows] != source_ids:
        raise ValueError("Source/truth order mismatch")
    truth = [set(filter(None, row["matched_entity_ids"].split(","))) for row in truth_rows]
    group = data["group"]
    labels = np.array([target in truth[i] for target, i in
                       zip(data["frame"]["target_id"], group)], dtype=bool)
    truth_count = np.array([len(values) for values in truth], dtype=np.int32)
    country = data["country"]
    folds = np.array([hashlib.sha256(sid.encode()).digest()[0] % 2
                      for sid in source_ids], dtype=np.int8)
    x = data["feature_matrices"]["context_plus_core_frequency"]
    crossfit = np.empty(len(labels), dtype=np.float32)
    for fold in (0, 1):
        for place in ("India", "US"):
            pool = country[group] == place
            train = pool & (folds[group] != fold)
            valid = pool & (folds[group] == fold)
            model = XGBClassifier(**MODEL_PARAMS)
            model.fit(x[train], labels[train].astype(np.int8))
            crossfit[valid] = model.predict_proba(x[valid])[:, 1].astype(np.float32)
    thresholds = [round(float(value), 3) for value in np.arange(.25, .951, .025)]
    sweep = [{"threshold": threshold,
              "score": metrics(crossfit >= threshold, labels, group, truth_count, country)["overall"]["macro_f05"]}
             for threshold in thresholds]
    best = max(sweep, key=lambda value: value["score"])
    report = {
        "scope": "exposed v1 10k; entity-disjoint two-fold development screen",
        "model": "separate India and US models with identical six features and hyperparameters",
        "fixed_threshold_0p7": metrics(crossfit >= .7, labels, group, truth_count, country),
        "best_exposed_threshold": best["threshold"],
        "best_exposed_metrics": metrics(crossfit >= best["threshold"], labels, group, truth_count, country),
        "threshold_sweep": sweep,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "fixed_threshold_0p7": report["fixed_threshold_0p7"],
        "best_exposed_threshold": best["threshold"],
        "best_exposed_metrics": report["best_exposed_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
