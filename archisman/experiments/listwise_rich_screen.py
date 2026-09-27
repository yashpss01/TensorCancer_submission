"""Development-only group-listwise ranker on unchanged rich candidate sets."""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
from xgboost import XGBRanker

from core_frequency_model_screen import metrics
from rich_country_threshold_screen import exact_threshold_ceiling
from rich_pair_model_screen import fit, load_cohort


def ranker():
    return XGBRanker(
        objective="rank:ndcg", n_estimators=300, max_depth=4,
        learning_rate=.045, min_child_weight=16, subsample=.85,
        colsample_bytree=.85, reg_lambda=6, tree_method="hist", n_jobs=4,
        random_state=20260927, lambdarank_num_pair_per_sample=8,
    )


def fit_ranker(data, mask):
    model = ranker()
    model.fit(data["x"][mask], data["labels"][mask].astype(np.int8),
              qid=data["group"][mask])
    return model


def out_of_fold(data):
    group_fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                           for s in data["source_ids"]], dtype=np.int8)
    pair_fold = group_fold[data["group"]]
    scores = np.empty(len(data["labels"]), dtype=np.float32)
    for held in (0, 1):
        train_mask = pair_fold != held
        model = fit_ranker(data, train_mask)
        scores[~train_mask] = model.predict(data["x"][~train_mask])
    return scores


def direction(train, test, baseline_threshold):
    oof = out_of_fold(train)
    chosen = exact_threshold_ceiling(train, oof)["overall"]["threshold"]
    if not isinstance(chosen, float):
        raise ValueError("Ranker selected no predictions")
    model = fit_ranker(train, np.ones(len(train["labels"]), dtype=bool))
    test_scores = model.predict(test["x"])
    base = fit(train["x"], train["labels"])
    base_scores = base.predict_proba(test["x"])[:, 1]
    arguments = (test["labels"], test["group"], test["truth_count"],
                 test["country"])
    return {"ranker_objective": "rank:ndcg", "ranker_trees": 300,
            "ranker_threshold_from_training_oof": chosen,
            "training_oof_ranker": metrics(oof >= chosen,
                                           train["labels"], train["group"],
                                           train["truth_count"], train["country"]),
            "test_rich_baseline": metrics(base_scores >= baseline_threshold,
                                          *arguments),
            "test_listwise_ranker": metrics(test_scores >= chosen, *arguments)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path")
    cohorts = {name: load_cohort(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    result = {"scope": "two exposed 10k reduced-pool cross-cohort screens",
              "candidate_lists": "unchanged", "directions": {}}
    started = time.monotonic()
    for fit_name, test_name, baseline_threshold in (
            ("fresh_10k_v1", "fresh_10k_v2", .7),
            ("fresh_10k_v2", "fresh_10k_v1", .825)):
        result["directions"][f"{fit_name}_to_{test_name}"] = direction(
            cohorts[fit_name], cohorts[test_name], baseline_threshold)
        row = result["directions"][f"{fit_name}_to_{test_name}"]
        print(fit_name, "to", test_name,
              "rich", row["test_rich_baseline"]["overall"]["macro_f05"],
              "ranker", row["test_listwise_ranker"]["overall"]["macro_f05"],
              "India", row["test_listwise_ranker"]["India"]["macro_f05"],
              flush=True)
    result["seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
