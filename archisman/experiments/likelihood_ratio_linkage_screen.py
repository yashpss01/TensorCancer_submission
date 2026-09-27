"""Small Fellegi-Sunter-style comparator on exposed development candidates.

Comparison levels and smoothing are fixed before either direction is scored.
Training-entity crossfit chooses a threshold; opposite-cohort scores use only
the training cohort's labels. This does not retrain or change the blocker.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np

from core_frequency_model_screen import metrics
from rich_pair_model_screen import load_cohort

ALPHA = 5.0
LEVEL_COUNTS = (11, 8, 3, 3)


def levels(cohort):
    x = cohort["x"]
    names = cohort["names"]
    def col(name):
        return np.asarray(x[:, names.index(name)])
    exact_core = col("name_core_exact") > .5
    name = np.where(exact_core,
                    np.digitize(col("log_target_core_frequency"), [1, 3, 5, 7]),
                    5 + np.digitize(col("name_core_dice"), [.2, .4, .6, .8, .95]))
    missing = col("address_missing_either") > .5
    exact_address = col("address_exact") > .5
    address = np.where(missing, 0,
                       np.where(exact_address, 1,
                                2 + np.digitize(col("address_token_dice"),
                                                [.2, .4, .6, .8, .95])))
    numbers = np.where(col("address_number_conflict") > .5, 2,
                       np.where(col("address_number_count_shared") > 0, 1, 0))
    postal = np.where(col("address_postal_conflict") > .5, 2,
                      np.where(col("address_postal_equal") > .5, 1, 0))
    result = np.column_stack((name, address, numbers, postal)).astype(np.int16)
    if np.any(result < 0) or any(np.any(result[:, j] >= k)
                                 for j, k in enumerate(LEVEL_COUNTS)):
        raise ValueError("Comparison level outside fixed range")
    return result


def fit_weights(x, labels):
    weights = []
    for j, categories in enumerate(LEVEL_COUNTS):
        matched = np.bincount(x[labels, j], minlength=categories).astype(float)
        unmatched = np.bincount(x[~labels, j], minlength=categories).astype(float)
        m = (matched + ALPHA) / (matched.sum() + ALPHA * categories)
        u = (unmatched + ALPHA) / (unmatched.sum() + ALPHA * categories)
        weights.append(np.log(m / u))
    return weights


def score(x, weights):
    result = np.zeros(len(x), dtype=np.float64)
    for j, w in enumerate(weights):
        result += w[x[:, j]]
    return result


def screen(train, test):
    train_x, test_x = levels(train), levels(test)
    folds = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                      for s in train["source_ids"]], dtype=np.int8)
    crossfit = np.empty(len(train_x), dtype=np.float64)
    for fold in (0, 1):
        fit_mask = folds[train["group"]] != fold
        crossfit[~fit_mask] = score(train_x[~fit_mask],
                                    fit_weights(train_x[fit_mask],
                                                train["labels"][fit_mask]))
    cutoffs = np.unique(np.quantile(crossfit, np.linspace(.8, .9999, 201)))
    trials = [(metrics(crossfit >= cutoff, train["labels"], train["group"],
                       train["truth_count"], train["country"])["overall"]["macro_f05"],
               float(cutoff)) for cutoff in cutoffs]
    crossfit_f05, chosen = max(trials)
    weights = fit_weights(train_x, train["labels"])
    test_score = score(test_x, weights)
    result = metrics(test_score >= chosen, test["labels"], test["group"],
                     test["truth_count"], test["country"])
    return {"train_crossfit_threshold": chosen,
            "train_crossfit_macro_f05": crossfit_f05,
            "test": result,
            "level_weights": [w.tolist() for w in weights]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path")
    cohorts = {name: load_cohort(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    result = {
        "scope": "exposed 10k reduced-pool cross-cohort development only",
        "method": "smoothed match/nonmatch log likelihood ratios for four fixed comparison levels",
        "smoothing_alpha": ALPHA,
        "level_counts": LEVEL_COUNTS,
        "directions": {},
    }
    for left, right in (("fresh_10k_v1", "fresh_10k_v2"),
                        ("fresh_10k_v2", "fresh_10k_v1")):
        part = screen(cohorts[left], cohorts[right])
        result["directions"][f"{left}_to_{right}"] = part
        print(left, "to", right,
              "overall", part["test"]["overall"]["macro_f05"],
              "India", part["test"]["India"]["macro_f05"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
