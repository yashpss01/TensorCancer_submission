"""Freeze a rich pair reranker using only the two exposed 10k cohorts."""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Use a new model directory")
    started = time.monotonic()
    first = load_cohort(args.root / "work/fresh_10k_v1")
    second = load_cohort(args.root / "work/fresh_10k_v2")
    if first["names"] != second["names"]:
        raise ValueError("Feature layouts differ")
    x = np.vstack((first["x"], second["x"]))
    y = np.r_[first["labels"], second["labels"]]
    group = np.r_[first["group"], second["group"] + len(first["source_ids"])]
    truth_count = np.r_[first["truth_count"], second["truth_count"]]
    country = np.r_[first["country"], second["country"]]
    ids = first["source_ids"] + second["source_ids"]
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2 for s in ids], dtype=np.int8)
    crossfit = np.empty(len(y), dtype=np.float32)
    for holdout in (0, 1):
        train = fold[group] != holdout
        model = fit(x[train], y[train])
        crossfit[~train] = model.predict_proba(x[~train])[:, 1]
    tested = []
    for threshold in np.arange(.4, .951, .025):
        summary = metrics(crossfit >= threshold, y, group, truth_count, country)
        tested.append((summary["overall"]["macro_f05"], float(round(threshold, 3)), summary))
    _, threshold, crossfit_metrics = max(tested, key=lambda row: row[0])
    final = fit(x, y)
    args.out_dir.mkdir(parents=True)
    model_path = args.out_dir / "rich_pair_model.json"
    final.save_model(model_path)
    from core_frequency_features import digest
    manifest = {
        "status": "frozen_before_fresh_final_truth",
        "scope": "exposed fresh-v1 and fresh-v2, 20k S1, 1,003,680 candidate pairs; no final cohort used",
        "feature_names": first["names"], "threshold": threshold,
        "model_params": {
            "n_estimators": 300, "max_depth": 4, "learning_rate": .045,
            "min_child_weight": 16, "subsample": .85, "colsample_bytree": .85,
            "reg_lambda": 6, "tree_method": "hist", "n_jobs": 4,
            "random_state": 20260927, "eval_metric": "logloss",
        },
        "development_crossfit": crossfit_metrics,
        "source1_rows": len(ids), "candidate_pairs": len(y),
        "positive_candidate_pairs": int(y.sum()),
        "cohort_manifests_sha256": {
            cohort: digest(args.root / "work" / cohort / "rich_pair_features.json")
            for cohort in ("fresh_10k_v1", "fresh_10k_v2")
        },
        "model_sha256": digest(model_path),
        "seconds": time.monotonic() - started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in
                      ("feature_names", "development_crossfit")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
