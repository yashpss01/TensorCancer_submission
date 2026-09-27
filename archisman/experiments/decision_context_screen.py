"""Development-only crossfit of contextual thresholds on cached candidates.

No fresh 10k or final holdout labels are read. Each half of the exposed 10k
development entities tunes thresholds for the other half. A positive result is
only a proposal for later untouched validation, not a deployable score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np


def score(pred: np.ndarray, hit: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    fp = pred - hit
    value = np.where(truth == 0, (pred == 0).astype(float),
                     1.25 * hit / np.maximum(hit + 0.25 * truth + fp, 1e-12))
    return {"macro_f0_5": float(value[mask].mean()),
            "tp": int(hit[mask].sum()), "fp": int(fp[mask].sum()),
            "fn": int((truth[mask] - hit[mask]).sum()),
            "groups": int(mask.sum())}


def bucket_counts(prob: np.ndarray, labels: np.ndarray, group_ids: np.ndarray,
                  pair_mask: np.ndarray, thresholds: np.ndarray, groups: int):
    pred = []
    hit = []
    for threshold in thresholds:
        chosen = pair_mask & (prob >= threshold)
        pred.append(np.bincount(group_ids[chosen], minlength=groups).astype(np.int32))
        hit.append(np.bincount(group_ids[chosen & labels], minlength=groups).astype(np.int32))
    return np.stack(pred), np.stack(hit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    sys.path.insert(0, str(root / "code/business_entity_resolution/round5"))
    import train

    art = root / "artifacts/matching_round5"
    val = art / "validation"
    x = np.load(val / "X.npy", mmap_mode="r")
    e = np.load(val / "extra_pair.npy", mmap_mode="r")
    group = np.load(val / "group_features.npy", mmap_mode="r")
    labels = np.load(val / "y.npy", mmap_mode="r").astype(bool)
    offsets = np.load(val / "offsets.npy")
    prob = (0.4 * np.load(art / "validation_group_extra_prob.npy")
            + 0.6 * np.load(art / "validation_augmented_0p15_prob.npy"))
    truth, countries = train.group_truth("validation")
    n_groups = len(truth)
    group_ids = np.repeat(np.arange(n_groups, dtype=np.int32), np.diff(offsets))
    assert len(group_ids) == len(prob) == len(labels)
    source_ids = [row["entity_id"] for row in json.loads(
        (root / "artifacts/blocking_round4/fresh/queries.json").read_text())]
    assert len(source_ids) == n_groups
    fold = np.array([hashlib.sha256(sid.encode()).digest()[0] & 1 for sid in source_ids], dtype=np.int8)
    contexts = {
        "missing_address": (x[:, 34] > 0.5) | (e[:, 11] > 0.5),
        "unreliable_anchor": group[:, 1] < 0.9,
        "address_number_conflict": (x[:, 30] > 0.5) | (e[:, 15] > 0.5),
        "India_country": np.repeat(countries["India"], np.diff(offsets)),
        "US_country": np.repeat(countries["US"], np.diff(offsets)),
    }
    # A small predetermined grid prevents unbounded search on the exposed set.
    thresholds = np.unique(np.concatenate((np.arange(0.50, 0.951, 0.025), [0.74])))
    baseline_chosen = prob >= 0.74
    baseline_pred = np.bincount(group_ids[baseline_chosen], minlength=n_groups).astype(np.int32)
    baseline_hit = np.bincount(group_ids[baseline_chosen & labels], minlength=n_groups).astype(np.int32)
    results = {"scope": "exposed 10k development split only; paired entity-level two-fold crossfit",
               "baseline_threshold": 0.74,
               "baseline": score(baseline_pred, baseline_hit, truth, np.ones(n_groups, dtype=bool)),
               "context_prevalence": {name: int(mask.sum()) for name, mask in contexts.items()},
               "rules": {}}
    trials = {name: [mask, ~mask] for name, mask in contexts.items()}
    missing = contexts["missing_address"]
    ambiguous = contexts["unreliable_anchor"] & ~missing
    trials["missing_plus_ambiguous"] = [missing, ambiguous,
                                         ~(missing | ambiguous)]
    for name, buckets in trials.items():
        matrices = []
        for index, bucket in enumerate(buckets):
            candidates = thresholds if index < len(buckets) - 1 else np.array([0.74])
            matrices.append((candidates, *bucket_counts(prob, labels, group_ids,
                                                        bucket, candidates, n_groups)))
        fold_reports = []
        for test_fold in (0, 1):
            tune_mask = fold != test_fold
            test_mask = fold == test_fold
            best = None
            for choice in np.ndindex(*(len(item[0]) for item in matrices)):
                pred = sum(item[1][idx] for item, idx in zip(matrices, choice))
                hit = sum(item[2][idx] for item, idx in zip(matrices, choice))
                training_score = score(pred, hit, truth, tune_mask)
                objective = (training_score["macro_f0_5"], -training_score["fp"])
                if best is None or objective > best[0]:
                    best = (objective, choice, pred, hit, training_score)
            _, choice, pred, hit, tuned = best
            test_score = score(pred, hit, truth, test_mask)
            fold_reports.append({"test_fold": test_fold,
                                 "thresholds": [float(item[0][idx]) for item, idx in zip(matrices, choice)],
                                 "tune": tuned, "test": test_score,
                                 "baseline_test": score(baseline_pred, baseline_hit, truth, test_mask),
                                 "countries": {country: score(pred, hit, truth, test_mask & mask)
                                               for country, mask in countries.items()}})
        results["rules"][name] = {"folds": fold_reports,
                                  "mean_test_macro_f0_5": float(np.mean([r["test"]["macro_f0_5"] for r in fold_reports])),
                                  "mean_baseline_macro_f0_5": float(np.mean([r["baseline_test"]["macro_f0_5"] for r in fold_reports]))}
        print(name, results["rules"][name]["mean_test_macro_f0_5"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
