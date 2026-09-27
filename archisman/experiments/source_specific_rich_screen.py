"""Cross-cohort test of separate S2/S3 rich matchers on unchanged candidates.

Source prefixes are inference-visible. The threshold for each direction is the
previously selected global rich threshold from training-entity crossfit; no
test labels choose it. Both test cohorts are already exposed development data.
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def source_masks(root):
    frame = pl.read_parquet(root / "frozen_pair_scores.parquet")
    ids = frame["target_id"].to_list()
    s2 = np.fromiter((value.startswith("S2-") for value in ids),
                     dtype=bool, count=len(ids))
    if any(not value.startswith(("S2-", "S3-")) for value in ids):
        raise ValueError("Unexpected target source prefix")
    return s2, ~s2


def outcome(train, test, train_masks, test_masks, threshold):
    global_model = fit(train["x"], train["labels"])
    global_probability = global_model.predict_proba(test["x"])[:, 1]
    separate_probability = np.empty(len(test["labels"]), dtype=np.float64)
    for train_mask, test_mask in zip(train_masks, test_masks):
        model = fit(train["x"][train_mask], train["labels"][train_mask])
        separate_probability[test_mask] = model.predict_proba(test["x"][test_mask])[:, 1]
    kwargs = (test["labels"], test["group"], test["truth_count"], test["country"])
    return {
        "threshold_from_global_training_crossfit": threshold,
        "train_s2_pairs": int(train_masks[0].sum()),
        "train_s3_pairs": int(train_masks[1].sum()),
        "test_s2_pairs": int(test_masks[0].sum()),
        "test_s3_pairs": int(test_masks[1].sum()),
        "global_rich": metrics(global_probability >= threshold, *kwargs),
        "separate_source_models": metrics(separate_probability >= threshold, *kwargs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path")
    cohorts = {name: load_cohort(args.root / "work" / name)
               for name in ("fresh_10k_v1", "fresh_10k_v2")}
    masks = {name: source_masks(args.root / "work" / name)
             for name in cohorts}
    result = {"scope": "two exposed 10k reduced-pool cross-cohort screens only",
              "candidate_lists": "unchanged", "directions": {}}
    for left, right, threshold in (("fresh_10k_v1", "fresh_10k_v2", .7),
                                   ("fresh_10k_v2", "fresh_10k_v1", .825)):
        part = outcome(cohorts[left], cohorts[right], masks[left], masks[right], threshold)
        result["directions"][f"{left}_to_{right}"] = part
        print(left, "to", right,
              "overall", part["global_rich"]["overall"]["macro_f05"], "to",
              part["separate_source_models"]["overall"]["macro_f05"],
              "India", part["global_rich"]["India"]["macro_f05"], "to",
              part["separate_source_models"]["India"]["macro_f05"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
