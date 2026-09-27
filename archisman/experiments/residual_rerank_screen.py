"""Exploratory entity-disjoint residual reranker on exposed development pairs.

The two outer test folds are never used for fitting or threshold selection.
This is a development screen, not an untouched or full-corpus result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from xgboost import XGBClassifier


def metrics(prob, threshold, labels, offsets, truth, group_mask):
    chosen = prob >= threshold
    cumulative_pred = np.r_[0, np.cumsum(chosen, dtype=np.int64)]
    cumulative_hit = np.r_[0, np.cumsum(chosen & labels, dtype=np.int64)]
    pred = np.diff(cumulative_pred[offsets])
    hit = np.diff(cumulative_hit[offsets])
    fp = pred - hit
    entity_f05 = np.where(
        truth == 0, (pred == 0).astype(float),
        1.25 * hit / np.maximum(hit + 0.25 * truth + fp, 1e-12),
    )
    return {
        "groups": int(group_mask.sum()),
        "macro_f0_5": float(entity_f05[group_mask].mean()),
        "tp": int(hit[group_mask].sum()),
        "fp": int(fp[group_mask].sum()),
        "fn": int((truth[group_mask] - hit[group_mask]).sum()),
    }


def make_features(x, e, g, group_prob, augmented_prob):
    # Two independent fitted match scores, plus narrowly selected evidence.
    # All columns were computed without exposed validation labels.
    selected_x = [1, 2, 7, 10, 12, 13, 15, 20, 21, 25, 27,
                  28, 30, 31, 32, 34, 35, 36, 38]
    selected_e = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
    selected_g = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15,
                  16, 17, 18, 22, 23, 24, 25, 26]
    return np.column_stack((
        group_prob, augmented_prob,
        np.asarray(x[:, selected_x]),
        np.asarray(e[:, selected_e]),
        np.asarray(g[:, selected_g]),
    )).astype(np.float32)


def new_model():
    return XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.05,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=5, tree_method="hist", n_jobs=4,
        objective="binary:logistic", eval_metric="logloss",
        random_state=20260927,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--fit-full-model", type=pathlib.Path)
    args = p.parse_args()
    start = time.monotonic()
    root = args.source_root
    sys.path.insert(0, str(root / "code/business_entity_resolution/round5"))
    import train

    art = root / "artifacts/matching_round5"
    val = art / "validation"
    x = np.load(val / "X.npy", mmap_mode="r")
    e = np.load(val / "extra_pair.npy", mmap_mode="r")
    g = np.load(val / "group_features.npy", mmap_mode="r")
    labels = np.load(val / "y.npy", mmap_mode="r").astype(bool)
    offsets = np.load(val / "offsets.npy")
    group_prob = np.load(art / "validation_group_extra_prob.npy")
    augmented_prob = np.load(art / "validation_augmented_0p15_prob.npy")
    baseline_prob = 0.4 * group_prob + 0.6 * augmented_prob
    truth, countries = train.group_truth("validation")
    ids = [row["entity_id"] for row in json.loads(
        (root / "artifacts/blocking_round4/fresh/queries.json").read_text())]
    assert len(ids) == len(truth) == len(offsets) - 1
    keys = np.asarray([list(hashlib.sha256(s.encode()).digest()[:2]) for s in ids], dtype=np.uint8)
    outer_fold = keys[:, 0] & 1
    calibration = keys[:, 1] % 5 == 0
    features = make_features(x, e, g, group_prob, augmented_prob)
    assert len(features) == len(labels)
    report = {
        "scope": "exposed 10k development split; paired, entity-disjoint outer crossfit",
        "features": int(features.shape[1]),
        "fitting": "each outer fold trains on ~40% of groups, calibrates on ~10%, tests on ~50%",
        "folds": [],
    }
    thresholds = np.arange(0.40, 0.961, 0.02)
    for test_fold in (0, 1):
        outer_train = outer_fold != test_fold
        fit_groups = outer_train & ~calibration
        cal_groups = outer_train & calibration
        test_groups = outer_fold == test_fold
        fit_pairs = np.repeat(fit_groups, np.diff(offsets))
        model = new_model()
        model.fit(features[fit_pairs], labels[fit_pairs], verbose=False)
        predicted = model.predict_proba(features)[:, 1].astype(np.float32)
        variants = {
            "residual_only": predicted,
            "blend_20pct_residual": 0.8 * baseline_prob + 0.2 * predicted,
            "blend_50pct_residual": 0.5 * baseline_prob + 0.5 * predicted,
        }
        calibrated_variants = {}
        for variant, probabilities in variants.items():
            calibrated = max(
                ((metrics(probabilities, float(t), labels, offsets, truth, cal_groups)["macro_f0_5"], float(t))
                 for t in thresholds),
                key=lambda item: item[0],
            )
            calibrated_variants[variant] = {
                "threshold": calibrated[1],
                "calibration_f0_5": calibrated[0],
                "test": metrics(probabilities, calibrated[1], labels, offsets, truth, test_groups),
                "conservative_0p74_test": metrics(probabilities, 0.74, labels, offsets, truth, test_groups),
            }
        threshold = calibrated_variants["residual_only"]["threshold"]
        test = calibrated_variants["residual_only"]["test"]
        baseline = metrics(baseline_prob, 0.74, labels, offsets, truth, test_groups)
        fold = {
            "outer_test_fold": test_fold,
            "fit_groups": int(fit_groups.sum()),
            "calibration_groups": int(cal_groups.sum()),
            "threshold": threshold,
            "calibration_f0_5": calibrated_variants["residual_only"]["calibration_f0_5"],
            "test": test,
            "baseline_test": baseline,
            "variants": calibrated_variants,
            "countries": {
                c: {
                    "changed": metrics(predicted, threshold, labels, offsets, truth, test_groups & mask),
                    "baseline": metrics(baseline_prob, 0.74, labels, offsets, truth, test_groups & mask),
                } for c, mask in countries.items()
            },
            "country_fixed_threshold_variants": {
                country: {
                    variant: metrics(probabilities, 0.74, labels, offsets, truth, test_groups & mask)
                    for variant, probabilities in variants.items()
                } for country, mask in countries.items()
            },
            "seconds_elapsed": round(time.monotonic() - start, 1),
        }
        report["folds"].append(fold)
        print(json.dumps(fold), flush=True)
    report["seconds_total"] = round(time.monotonic() - start, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output, flush=True)
    if args.fit_full_model:
        full_model = new_model()
        full_model.fit(features, labels, verbose=False)
        args.fit_full_model.parent.mkdir(parents=True, exist_ok=True)
        full_model.save_model(args.fit_full_model)
        manifest = {
            "scope": "experimental reranker trained on exposed 10k development pairs",
            "model": args.fit_full_model.name,
            "features": int(features.shape[1]),
            "decision": "0.8 * frozen_probability + 0.2 * residual_probability >= 0.74",
            "not_fresh_validated": True,
        }
        args.fit_full_model.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Saved experimental model", args.fit_full_model, flush=True)


if __name__ == "__main__":
    main()
