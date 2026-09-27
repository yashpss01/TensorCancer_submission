"""Loss attribution for exposed development data, with overlapping strata."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def aggregate_group(mask, truth, retrieved, hit, fp, actual, oracle, categories):
    selected = np.asarray(mask, dtype=bool)
    n = int(selected.sum())
    return {
        "groups": n,
        "truth_pairs": int(truth[selected].sum()),
        "candidate_tp": int(retrieved[selected].sum()),
        "candidate_fn": int((truth[selected] - retrieved[selected]).sum()),
        "matcher_tp": int(hit[selected].sum()),
        "matcher_fp": int(fp[selected].sum()),
        "matcher_fn": int((truth[selected] - hit[selected]).sum()),
        "macro_f0_5": float(actual[selected].mean()) if n else None,
        "oracle_macro_f0_5": float(oracle[selected].mean()) if n else None,
        "total_score_loss_sum": float((1 - actual[selected]).sum()),
        "retrieval_loss_sum": float((1 - oracle[selected]).sum()),
        "matcher_loss_sum": float((oracle[selected] - actual[selected]).sum()),
        "entities_with_retrieval_miss": int(np.count_nonzero(selected & (truth > retrieved))),
        "entities_with_rejected_true_candidate": int(np.count_nonzero(selected & (retrieved > hit))),
        "entities_with_false_accept": int(np.count_nonzero(selected & (fp > 0))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    root = args.source_root
    sys.path.insert(0, str(root / "code/business_entity_resolution/round5"))
    import train

    art = root / "artifacts/matching_round5"
    val = art / "validation"
    x = np.load(val / "X.npy", mmap_mode="r")
    e = np.load(val / "extra_pair.npy", mmap_mode="r")
    g = np.load(val / "group_features.npy", mmap_mode="r")
    y = np.load(val / "y.npy", mmap_mode="r").astype(bool)
    offsets = np.load(val / "offsets.npy")
    prob = (0.4 * np.load(art / "validation_group_extra_prob.npy")
            + 0.6 * np.load(art / "validation_augmented_0p15_prob.npy"))
    truth, countries = train.group_truth("validation")
    queries = json.loads((root / "artifacts/blocking_round4/fresh/queries.json").read_text())
    assert len(queries) == len(truth) == len(offsets) - 1
    n = len(queries)
    group_ids = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
    chosen = prob >= 0.74
    retrieved = np.bincount(group_ids[y], minlength=n)
    hit = np.bincount(group_ids[y & chosen], minlength=n)
    fp = np.bincount(group_ids[~y & chosen], minlength=n)
    pred = hit + fp
    actual = np.where(truth == 0, (pred == 0).astype(float),
                      1.25 * hit / np.maximum(hit + 0.25 * truth + fp, 1e-12))
    oracle = np.where(truth == 0, 1.0,
                      1.25 * retrieved / np.maximum(retrieved + 0.25 * truth, 1e-12))
    missing_pair = (x[:, 34] > 0.5) | (e[:, 11] > 0.5)
    script_pair = x[:, 20] > 0.5
    conflict_pair = (x[:, 30] > 0.5) | (x[:, 32] > 0.5) | (e[:, 15] > 0.5)
    exact_core_pair = x[:, 2] > 0.5
    high = prob >= 0.5
    exact_count = np.bincount(group_ids[exact_core_pair], minlength=n)
    high_count = np.bincount(group_ids[high], minlength=n)
    group_any = lambda pair_mask: np.bincount(group_ids[pair_mask], minlength=n) > 0
    categories = {
        "India": countries["India"],
        "US": countries["US"],
        "singleton": truth == 0,
        "query_missing_address": np.array([not q["business_address"].strip() for q in queries]),
        "true_candidate_missing_address": group_any(y & missing_pair),
        "true_candidate_script_mismatch": group_any(y & script_pair),
        "true_candidate_address_conflict": group_any(y & conflict_pair),
        "multiple_exact_core_candidates": exact_count >= 2,
        "multiple_high_score_candidates": high_count >= 2,
        "positive_group_without_true_candidate": (truth > 0) & (retrieved == 0),
        "one_true_link": truth == 1,
        "two_true_links": truth == 2,
        "three_or_more_true_links": truth >= 3,
        "at_least_60_candidates": np.diff(offsets) >= 60,
    }
    for country in ("India", "US"):
        for condition in ("true_candidate_missing_address", "true_candidate_script_mismatch",
                          "true_candidate_address_conflict", "multiple_exact_core_candidates"):
            categories[f"{country}_and_{condition}"] = countries[country] & categories[condition]
    overall = aggregate_group(np.ones(n, dtype=bool), truth, retrieved, hit, fp,
                              actual, oracle, categories)
    strata = {name: aggregate_group(mask, truth, retrieved, hit, fp, actual, oracle, categories)
              for name, mask in categories.items()}
    pair_strata = {}
    for name, mask in {
        "missing_address": missing_pair,
        "script_mismatch": script_pair,
        "address_conflict": conflict_pair,
        "exact_core_name": exact_core_pair,
    }.items():
        pair_strata[name] = {
            "retrieved_true": int((y & mask).sum()),
            "retrieved_true_rejected": int((y & ~chosen & mask).sum()),
            "false_accepted": int((~y & chosen & mask).sum()),
            "true_accepted": int((y & chosen & mask).sum()),
        }
    output = {
        "scope": "exposed 10k validation, 409141-target reduced pool; overlapping categories",
        "decision_threshold": 0.74,
        "overall": overall,
        "strata": strata,
        "pair_strata": pair_strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"overall": overall, "pair_strata": pair_strata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
