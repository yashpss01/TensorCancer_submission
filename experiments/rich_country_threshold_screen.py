"""Country-specific rich thresholds with entity-disjoint training crossfit.

Both 10k cohorts are already exposed development data. Candidate lists and
model features remain unchanged. Neither opposite-cohort check is fresh.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
from xgboost import XGBClassifier

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


GRID = np.round(np.arange(.4, .951, .025), 3)


def crossfit_probabilities(train):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in train["source_ids"]], dtype=np.int8)
    pair_fold = fold[train["group"]]
    out = np.empty(len(train["labels"]), dtype=np.float32)
    for held_out in (0, 1):
        fit_mask = pair_fold != held_out
        model = fit(train["x"][fit_mask], train["labels"][fit_mask])
        out[~fit_mask] = model.predict_proba(train["x"][~fit_mask])[:, 1]
    return out


def chosen_thresholds(train, probability):
    arguments = (train["labels"], train["group"], train["truth_count"],
                 train["country"])
    score_by_threshold = [(float(t), metrics(probability >= t, *arguments))
                          for t in GRID]
    return {part: max(score_by_threshold,
                      key=lambda row: row[1][part]["macro_f05"])[0]
            for part in ("overall", "India", "US")}


def run_direction(train, test):
    crossfit = crossfit_probabilities(train)
    thresholds = chosen_thresholds(train, crossfit)
    model = fit(train["x"], train["labels"])
    probability = model.predict_proba(test["x"])[:, 1]
    pair_country = test["country"][test["group"]]
    country_threshold = np.where(pair_country == "India",
                                 thresholds["India"], thresholds["US"])
    arguments = (test["labels"], test["group"], test["truth_count"],
                 test["country"])
    return {
        "thresholds_selected_on_training_crossfit": thresholds,
        "test_global_threshold": metrics(probability >= thresholds["overall"],
                                         *arguments),
        "test_country_thresholds": metrics(probability >= country_threshold,
                                           *arguments),
    }


def exact_threshold_ceiling(cohort, probability):
    """Optimistic exhaustive threshold search on an exposed labeled cohort."""
    order = np.argsort(-probability, kind="stable")
    score = probability[order]
    group = cohort["group"][order]
    label = cohort["labels"][order]
    truth = cohort["truth_count"]
    country = cohort["country"]
    n = len(truth)
    tp = np.zeros(n, dtype=np.int16)
    fp = np.zeros(n, dtype=np.int16)
    value = (truth == 0).astype(np.float64)
    total = float(value.sum())
    india = country == "India"
    india_total = float(value[india].sum())
    us_total = total - india_total
    best = {part: {"macro_f05": current / count,
                   "threshold": ">max_probability"}
            for part, current, count in (("overall", total, n),
                                         ("India", india_total, int(india.sum())),
                                         ("US", us_total, int((~india).sum())))}
    for i, (p, g, is_true) in enumerate(zip(score, group, label)):
        old = value[g]
        if is_true:
            tp[g] += 1
        else:
            fp[g] += 1
        if truth[g] == 0:
            value[g] = float(fp[g] == 0)
        else:
            value[g] = 1.25 * tp[g] / (tp[g] + .25 * truth[g] + fp[g])
        delta = value[g] - old
        total += delta
        if india[g]:
            india_total += delta
        else:
            us_total += delta
        if i + 1 < len(score) and score[i + 1] == p:
            continue  # >= threshold selects every pair sharing this score.
        for part, current, count in (("overall", total, n),
                                     ("India", india_total, int(india.sum())),
                                     ("US", us_total, int((~india).sum()))):
            candidate = current / count
            if candidate > best[part]["macro_f05"]:
                best[part] = {"macro_f05": candidate, "threshold": float(p)}
    best["country_specific_combined_macro_f05"] = (
        best["India"]["macro_f05"] * india.sum()
        + best["US"]["macro_f05"] * (~india).sum()) / n
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path")
    cohorts = {name: load_cohort(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    result = {"scope": "two already exposed 10k reduced-pool cohorts",
              "candidate_lists": "unchanged", "directions": {}}
    for fit_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                ("fresh_10k_v2", "fresh_10k_v1")):
        result["directions"][f"{fit_name}_to_{test_name}"] = run_direction(
            cohorts[fit_name], cohorts[test_name])
        outcome = result["directions"][f"{fit_name}_to_{test_name}"]
        print(fit_name, "to", test_name, "overall",
              outcome["test_global_threshold"]["overall"]["macro_f05"],
              "to", outcome["test_country_thresholds"]["overall"]["macro_f05"],
              flush=True)
    exposed_final = load_cohort(args.root / "work" / "fresh_10k_final")
    frozen = XGBClassifier()
    frozen.load_model(args.root / "experiments/checkpoints/rich_pair_v1v2/rich_pair_model.json")
    probability = frozen.predict_proba(exposed_final["x"])[:, 1]
    arguments = (exposed_final["labels"], exposed_final["group"],
                 exposed_final["truth_count"], exposed_final["country"])
    frozen_metrics = metrics(probability >= .775, *arguments)
    if abs(frozen_metrics["overall"]["macro_f05"] - .9877071019095269) > 1e-10:
        raise ValueError("Frozen confirmation score mismatch")
    ceiling = exact_threshold_ceiling(exposed_final, probability)
    for part in ("overall", "India", "US"):
        threshold = ceiling[part]["threshold"]
        if isinstance(threshold, float):
            checked = metrics(probability >= threshold, *arguments)
            if abs(checked[part]["macro_f05"] - ceiling[part]["macro_f05"]) > 1e-10:
                raise ValueError(f"Exact threshold sweep mismatch: {part}")
    result["exposed_final_threshold_oracle_diagnostic"] = {
        "scope": "selection-optimistic ceiling for this fixed score ordering only",
        "frozen_0p775": frozen_metrics,
        "best_exact_thresholds_using_exposed_labels": ceiling,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
