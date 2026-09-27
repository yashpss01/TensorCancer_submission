"""Exposed-cohort macro-F0.5 screen for a neural score on all uncertain pairs.

The neural scorer sees only input.json. This evaluator separately opens labels,
checks sample identity, then restores scores into the complete candidate list.
This is development selection, never a fresh confirmation claim.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
from scipy.special import expit

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def entity_scores(chosen, test):
    group = test["group"]
    labels = test["labels"]
    truth_count = test["truth_count"]
    count = len(truth_count)
    tp = np.bincount(group[chosen & labels], minlength=count)
    fp = np.bincount(group[chosen & ~labels], minlength=count)
    return np.where(truth_count == 0, (fp == 0).astype(float),
                    1.25 * tp / np.maximum(tp + .25 * truth_count + fp, 1e-12))


def paired_bootstrap(chosen, base_scores, test):
    difference = entity_scores(chosen, test) - base_scores
    result = {}
    for name, mask in (("overall", np.ones(len(difference), dtype=bool)),
                       ("India", test["country"] == "India"),
                       ("US", test["country"] == "US")):
        values = difference[mask]
        rng = np.random.default_rng(20260927)
        samples = [float(values[rng.integers(0, len(values), len(values))].mean())
                   for _ in range(1000)]
        result[name] = {
            "gain_percentage_points": float(100 * values.mean()),
            "paired_entity_bootstrap_95_pp":
                [float(100 * x) for x in np.quantile(samples, [.025, .975])],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--train", choices=("fresh_10k_v1", "fresh_10k_v2"), required=True)
    parser.add_argument("--test", choices=("fresh_10k_v1", "fresh_10k_v2"), required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--scores", type=pathlib.Path, required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Rich threshold selected on training cohort crossfit")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.train == args.test or args.output.exists():
        raise ValueError("Use distinct exposed cohorts and a new output path")
    label_payload = json.loads(args.labels.read_text())
    score_payload = json.loads(args.scores.read_text())
    input_path = args.labels.parent / "input.json"
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if (label_payload["input_sha256"] != input_hash or
            score_payload["evaluation_input_sha256"] != input_hash or
            label_payload["train_cohort"] != args.train or
            label_payload["test_cohort"] != args.test or
            score_payload["training_cohort"] != args.train or
            score_payload["test_cohort"] != args.test):
        raise ValueError("Neural scores, labels, and cohort direction differ")
    train = load_cohort(args.root / "work" / args.train)
    test = load_cohort(args.root / "work" / args.test)
    if train["names"] != test["names"]:
        raise ValueError("Rich feature layouts differ")
    rich = fit(train["x"], train["labels"]).predict_proba(test["x"])[:, 1]
    selected = np.asarray(label_payload["pair_indices"], dtype=np.int64)
    neural = expit(np.asarray(score_payload["logits"], dtype=np.float64))
    if (len(selected) != len(neural) or len(set(selected.tolist())) != len(selected) or
            np.any(selected < 0) or np.any(selected >= len(rich)) or
            not np.allclose(rich[selected], label_payload["rich_probability"], atol=1e-6) or
            not np.array_equal(test["labels"][selected], label_payload["labels"])):
        raise ValueError("Candidate ordering, neural score count, or labels differ")
    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be in (0,1)")
    baseline = metrics(rich >= args.threshold, test["labels"], test["group"],
                       test["truth_count"], test["country"])
    base_scores = entity_scores(rich >= args.threshold, test)
    variants = {}
    paired_gains = {}
    for alpha in (0.05, 0.1, 0.2, 0.3, 0.5):
        blend = rich.copy()
        blend[selected] = (1-alpha)*rich[selected] + alpha*neural
        chosen = blend >= args.threshold
        variants[str(alpha)] = metrics(chosen, test["labels"], test["group"],
                                       test["truth_count"], test["country"])
        paired_gains[str(alpha)] = paired_bootstrap(chosen, base_scores, test)
    result = {
        "scope": "two exposed 10k reduced-pool cohorts; development screen only",
        "train": args.train, "test": args.test,
        "candidate_pairs": len(rich), "neural_scored_uncertain_pairs": len(selected),
        "selection": "rich probability 0.05–0.95; all eligible pairs",
        "decision_threshold": args.threshold,
        "baseline": baseline, "variants": variants,
        "paired_gains": paired_gains,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("baseline", baseline["overall"]["macro_f05"],
          "India", baseline["India"]["macro_f05"], flush=True)
    for alpha, item in variants.items():
        print("blend", alpha, item["overall"]["macro_f05"],
              "India", item["India"]["macro_f05"], flush=True)


if __name__ == "__main__":
    main()
