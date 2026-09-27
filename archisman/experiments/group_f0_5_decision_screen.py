"""Test per-S1 expected-F0.5 decisions on exposed development cohorts.

The rule is selected only on v1/v2 entity-disjoint crossfit probabilities.
The third cohort is already exposed after its frozen confirmation and is used
here as a development diagnostic, not a new sealed validation claim.
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


def group_choice(prob, group, n_groups, odds_scale, zero_multiplier):
    counts = np.bincount(group, minlength=n_groups)
    offsets = np.r_[0, np.cumsum(counts)]
    chosen = np.zeros(len(prob), dtype=bool)
    for i in range(n_groups):
        lo, hi = offsets[i:i+2]
        if hi == lo:
            continue
        raw = np.clip(prob[lo:hi], 1e-6, 1-1e-6)
        p = raw / (raw + odds_scale * (1-raw))
        order = np.argsort(-p, kind="stable")
        sorted_p = p[order]
        k = np.arange(1, len(p)+1)
        utility = 1.25 * np.cumsum(sorted_p) / (k + .25 * p.sum())
        best_k = int(np.argmax(utility)) + 1
        empty_utility = np.exp(np.log1p(-p).sum()) * zero_multiplier
        if utility[best_k-1] >= empty_utility:
            chosen[lo + order[:best_k]] = True
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    v1 = load_cohort(args.root / "work/fresh_10k_v1")
    v2 = load_cohort(args.root / "work/fresh_10k_v2")
    x = np.vstack((v1["x"], v2["x"]))
    y = np.r_[v1["labels"], v2["labels"]]
    group = np.r_[v1["group"], v2["group"]+len(v1["source_ids"])]
    truth = np.r_[v1["truth_count"], v2["truth_count"]]
    country = np.r_[v1["country"], v2["country"]]
    ids = v1["source_ids"]+v2["source_ids"]
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2 for s in ids], dtype=np.int8)
    crossfit = np.empty(len(y), dtype=np.float32)
    for holdout in (0, 1):
        train = fold[group] != holdout
        classifier = fit(x[train], y[train])
        crossfit[~train] = classifier.predict_proba(x[~train])[:, 1]
    fixed = metrics(crossfit >= .775, y, group, truth, country)
    expected = .9880341136844041
    if abs(fixed["overall"]["macro_f05"] - expected) > 1e-10:
        raise ValueError("Crossfit probabilities do not reproduce the frozen selection screen")
    variants = []
    for scale in (.3, .5, .7, 1., 1.5, 2., 3.):
        for zero in (.5, 1., 2.):
            selected = group_choice(crossfit, group, len(ids), scale, zero)
            score = metrics(selected, y, group, truth, country)
            variants.append((score["overall"]["macro_f05"], scale, zero, score))
    best, scale, zero, dev_score = max(variants, key=lambda row: row[0])
    final = load_cohort(args.root / "work/fresh_10k_final")
    checkpoint = args.root / "experiments/checkpoints/rich_pair_v1v2/rich_pair_model.json"
    classifier = XGBClassifier()
    classifier.load_model(checkpoint)
    scores = classifier.predict_proba(final["x"])[:, 1]
    final_fixed = metrics(scores >= .775, final["labels"], final["group"],
                          final["truth_count"], final["country"])
    if abs(final_fixed["overall"]["macro_f05"] - .9877071019095269) > 1e-10:
        raise ValueError("Final frozen score does not reproduce prior confirmation")
    decision = group_choice(scores, final["group"], len(final["source_ids"]), scale, zero)
    final_changed = metrics(decision, final["labels"], final["group"],
                            final["truth_count"], final["country"])
    report = {
        "scope": "v1/v2 exposed crossfit selection; final cohort already exposed, used only as a development diagnostic",
        "development_fixed": fixed,
        "development_selected_rule": {"odds_scale": scale,
                                      "empty_utility_multiplier": zero,
                                      "metrics": dev_score},
        "retrospective_final_fixed": final_fixed,
        "retrospective_final_selected_rule": final_changed,
        "seconds": time.monotonic()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"dev_fixed": fixed["overall"], "dev_rule": dev_score["overall"],
                      "scale": scale, "zero": zero, "final_fixed": final_fixed["overall"],
                      "final_rule": final_changed["overall"],
                      "final_india": final_changed["India"],
                      "seconds": report["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
