"""Score complete paired outputs against one untouched bounded S1 batch."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

import numpy as np


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def ids(value):
    return value.split(",") if value else []


def summarize(items, target_count):
    sizes = sorted(item["candidate_count"] for item in items)
    n = len(items)
    candidate_pairs = sum(sizes)
    candidate_tp = sum(item["candidate_tp"] for item in items)
    candidate_fn = sum(item["candidate_fn"] for item in items)
    candidate_fp = candidate_pairs - candidate_tp
    comparison_space = n * target_count
    candidate_tn = comparison_space - candidate_tp - candidate_fn - candidate_fp
    return {
        "groups": n,
        "macro_f0_5": statistics.mean(item["f05"] for item in items),
        "tp": sum(item["tp"] for item in items),
        "fp": sum(item["fp"] for item in items),
        "fn": sum(item["fn"] for item in items),
        "candidate_comparison_space": comparison_space,
        "candidate_pairs": candidate_pairs,
        "candidate_tp": candidate_tp,
        "candidate_fn": candidate_fn,
        "candidate_fp": candidate_fp,
        "candidate_tn": candidate_tn,
        "candidate_recall": candidate_tp / (candidate_tp + candidate_fn),
        "candidate_precision": candidate_tp / candidate_pairs,
        "candidate_reduction_ratio": 1 - candidate_pairs / comparison_space,
        "candidate_specificity": candidate_tn / (candidate_tn + candidate_fp),
        "candidate_f1": 2 * candidate_tp / (2 * candidate_tp + candidate_fp + candidate_fn),
        "candidate_oracle_macro_f0_5": statistics.mean(item["candidate_oracle_f05"] for item in items),
        "candidate_mean": statistics.mean(sizes),
        "candidate_p95": sizes[(95 * n + 99) // 100 - 1],
        "candidate_max": sizes[-1],
        "singletons": sum(item["singleton"] for item in items),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--truth", type=pathlib.Path, required=True)
    p.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--baseline-tsv", type=pathlib.Path, required=True)
    p.add_argument("--changed-tsv", type=pathlib.Path, required=True)
    p.add_argument("--pool-meta", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    source = list(rows(args.source1))
    truth = list(rows(args.truth))
    candidate = list(rows(args.candidate_tsv))
    baseline = list(rows(args.baseline_tsv))
    changed = list(rows(args.changed_tsv))
    if not all(len(data) == len(source) == 10_000 for data in (truth, candidate, baseline, changed)):
        raise ValueError("Every input must cover the same complete 10k S1 batch")
    scope = json.loads(args.pool_meta.read_text())
    output = {"scope": scope, "results": {"baseline": [], "changed": []}}
    for s, t, c, b, a in zip(source, truth, candidate, baseline, changed):
        sid = s["entity_id"]
        if not sid == t["source1_entity_id"] == c["source1_entity_id"] == b["source1_entity_id"] == a["source1_entity_id"]:
            raise ValueError("S1 IDs/order mismatch")
        actual = set(ids(t["matched_entity_ids"]))
        candidates = set(ids(c["candidate_entity_ids"]))
        if len(candidates) != len(ids(c["candidate_entity_ids"])):
            raise ValueError("Duplicate candidates")
        for name, prediction in (("baseline", b), ("changed", a)):
            selected = set(ids(prediction["matched_entity_ids"]))
            if len(selected) != len(ids(prediction["matched_entity_ids"])) or not selected <= candidates:
                raise ValueError("Duplicate or out-of-candidate prediction")
            tp = len(selected & actual)
            fp = len(selected - actual)
            fn = len(actual - selected)
            f05 = (1.0 if not selected else 0.0) if not actual else 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
            output["results"][name].append({
                "country": s["country"], "f05": f05,
                "tp": tp, "fp": fp, "fn": fn,
                "candidate_tp": len(candidates & actual),
                "candidate_fn": len(actual - candidates),
                "candidate_oracle_f05": (1.0 if not actual else
                                         1.25 * len(candidates & actual) /
                                         (len(candidates & actual) + 0.25 * len(actual))),
                "candidate_count": len(candidates),
                "singleton": int(not actual),
            })
    report = {"target_pool": scope, "batch_rows": len(source), "pipelines": {}}
    for name, data in output["results"].items():
        report["pipelines"][name] = {
            "overall": summarize(data, scope["target_count"]),
            "countries": {
                country: summarize([item for item in data if item["country"] == country],
                                   scope["target_count"])
                for country in ("India", "US")
            },
        }
    country_by_row = [row["country"] for row in source]
    delta = np.asarray([changed["f05"] - baseline["f05"]
                        for baseline, changed in zip(output["results"]["baseline"],
                                                     output["results"]["changed"])])
    rng = np.random.default_rng(20260927)
    report["paired_delta"] = {}
    for label, mask in (("overall", np.ones(len(delta), dtype=bool)),
                        ("India", np.asarray(country_by_row) == "India"),
                        ("US", np.asarray(country_by_row) == "US")):
        values = delta[mask]
        draws = np.asarray([values[rng.integers(0, len(values), len(values))].mean()
                            for _ in range(3000)])
        report["paired_delta"][label] = {
            "macro_f0_5_delta": float(values.mean()),
            "paired_entity_bootstrap_95pct": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "improved_entities": int((values > 0).sum()),
            "worsened_entities": int((values < 0).sum()),
            "unchanged_entities": int((values == 0).sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pipelines"], indent=2), flush=True)


if __name__ == "__main__":
    main()
