"""Development-only joint promotion using frozen opposite-source anchors.

The anchor matrix was generated without labels from baseline-accepted targets.
Model fitting and rule selection use one exposed cohort; evaluation uses the
other exposed cohort. No fresh confirmation or full-corpus claim follows.
"""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def load(root):
    data = load_cohort(root)
    meta = json.loads((root / "cross_source_anchor.json").read_text())
    rich_meta = json.loads((root / "rich_pair_features.json").read_text())
    if (meta["source1_sha256"] != rich_meta["source1_sha256"] or
            meta["pair_scores_sha256"] != rich_meta["pair_scores_sha256"]):
        raise ValueError("Anchor and rich features belong to different candidate pairs")
    data["anchor"] = np.load(root / "cross_source_anchor.npy", mmap_mode="r")
    if len(data["anchor"]) != len(data["x"]):
        raise ValueError("Anchor and rich rows differ")
    data["anchor_names"] = meta["feature_names"]
    return data


def out_of_fold(data):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in data["source_ids"]], dtype=np.int8)
    probability = np.empty(len(data["labels"]), dtype=np.float32)
    for held in (0, 1):
        fit_rows = fold[data["group"]] != held
        test_rows = ~fit_rows
        model = fit(data["x"][fit_rows], data["labels"][fit_rows])
        probability[test_rows] = model.predict_proba(data["x"][test_rows])[:, 1]
    return probability


def score(data, chosen):
    return metrics(chosen, data["labels"], data["group"],
                   data["truth_count"], data["country"])


def promoted(data, probability, base_threshold, rule):
    chosen = probability >= base_threshold
    if rule is None:
        return chosen
    base_min, similarity, address_min = rule
    a = data["anchor"]
    names = data["anchor_names"]
    core = a[:, names.index("cross_anchor_core_wratio")]
    address = a[:, names.index("cross_anchor_address_ratio")]
    anchor_count = a[:, names.index("cross_anchor_count")]
    anchor_probability = a[:, names.index("cross_anchor_max_probability")]
    missing = data["x"][:, data["names"].index("target_address_empty")] > .5
    return chosen | ((probability >= base_min) & (anchor_count >= 1)
                     & (anchor_probability >= .9) & (core >= similarity)
                     & (missing | (address >= address_min)))


def choose(data, probability):
    threshold_grid = np.arange(.5, .951, .025)
    base_threshold = max(threshold_grid,
                         key=lambda threshold: score(data, probability >= threshold)
                         ["overall"]["macro_f05"])
    base_threshold = float(round(base_threshold, 3))
    baseline = score(data, probability >= base_threshold)
    best = (baseline["overall"]["macro_f05"], None, baseline)
    for base_min in (.3, .45, .6, .7):
        for similarity in (.8, .88, .94, .98):
            for address_min in (.0, .6, .85):
                rule = (base_min, similarity, address_min)
                value = score(data, promoted(data, probability, base_threshold, rule))
                if value["overall"]["macro_f05"] > best[0]:
                    best = (value["overall"]["macro_f05"], rule, value)
    return base_threshold, baseline, best[1], best[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    cohorts = {name: load(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    result = {"scope": "exposed v1/v2 reduced-pool cross-cohort development only",
              "candidate_sets_changed": False, "directions": {}}
    for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                  ("fresh_10k_v2", "fresh_10k_v1")):
        train, test = cohorts[train_name], cohorts[test_name]
        oof = out_of_fold(train)
        threshold, train_baseline, rule, train_changed = choose(train, oof)
        model = fit(train["x"], train["labels"])
        test_probability = model.predict_proba(test["x"])[:, 1]
        test_baseline = score(test, promoted(test, test_probability, threshold, None))
        test_changed = score(test, promoted(test, test_probability, threshold, rule))
        result["directions"][f"{train_name}_to_{test_name}"] = {
            "selected_base_threshold": threshold,
            "selected_rule": rule,
            "train_crossfit_baseline": train_baseline,
            "train_crossfit_changed": train_changed,
            "test_baseline": test_baseline,
            "test_changed": test_changed,
        }
        print(train_name, "to", test_name, "rule", rule,
              "baseline", test_baseline["overall"]["macro_f05"],
              "changed", test_changed["overall"]["macro_f05"],
              "India", test_changed["India"]["macro_f05"], flush=True)
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
