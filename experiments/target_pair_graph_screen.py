"""Cross-cohort target-to-target graph classifier and joint-decision screen.

Both cohorts are exposed development data. Graph features are label-blind;
pair labels are derived only here for training. No target ID is a feature.
"""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from core_frequency_model_screen import metrics
from rich_pair_model_screen import fit, load_cohort


def graph_model():
    return XGBClassifier(
        n_estimators=240, max_depth=4, learning_rate=.05,
        min_child_weight=12, subsample=.85, colsample_bytree=.9,
        reg_lambda=6, tree_method="hist", n_jobs=4,
        eval_metric="logloss", random_state=20260927,
    )


def load(root):
    data = load_cohort(root)
    graph_root = root / "target_pair_graph"
    graph_meta = json.loads((graph_root / "manifest.json").read_text())
    rich_meta = json.loads((root / "rich_pair_features.json").read_text())
    if (graph_meta["source1_sha256"] != rich_meta["source1_sha256"] or
            graph_meta["pair_scores_sha256"] != rich_meta["pair_scores_sha256"]):
        raise ValueError("Graph and rich features belong to different candidate pairs")
    candidate = np.load(graph_root / "candidate_row.npy", mmap_mode="r")
    anchor = np.load(graph_root / "anchor_row.npy", mmap_mode="r")
    matrix = np.load(graph_root / "features.npy", mmap_mode="r")
    if (len(candidate) != len(anchor) or len(candidate) != len(matrix) or
            np.any(candidate >= len(data["x"])) or np.any(anchor >= len(data["x"])) or
            np.any(data["group"][candidate] != data["group"][anchor])):
        raise ValueError("Graph candidate/anchor rows are not aligned")
    names = graph_meta["feature_names"]
    columns = [i for i, name in enumerate(names) if name != "candidate_frozen_probability"]
    graph = {"candidate": candidate, "anchor": anchor,
             "x": matrix[:, columns],
             "labels": data["labels"][candidate] & data["labels"][anchor],
             "group": data["group"][candidate]}
    return data, graph


def support(pair_prob, graph, candidate_count):
    values = np.zeros(candidate_count, dtype=np.float32)
    np.maximum.at(values, graph["candidate"], pair_prob)
    return values


def crossfit(data, graph):
    fold = np.array([hashlib.sha256(s.encode()).digest()[0] % 2
                     for s in data["source_ids"]], dtype=np.int8)
    rich_prob = np.empty(len(data["labels"]), dtype=np.float32)
    graph_prob = np.empty(len(graph["labels"]), dtype=np.float32)
    for held in (0, 1):
        train_rich = fold[data["group"]] != held
        test_rich = ~train_rich
        rich = fit(data["x"][train_rich], data["labels"][train_rich])
        rich_prob[test_rich] = rich.predict_proba(data["x"][test_rich])[:, 1]
        train_graph = fold[graph["group"]] != held
        test_graph = ~train_graph
        model = graph_model()
        model.fit(graph["x"][train_graph], graph["labels"][train_graph])
        graph_prob[test_graph] = model.predict_proba(graph["x"][test_graph])[:, 1]
    return rich_prob, graph_prob


def score(data, chosen):
    return metrics(chosen, data["labels"], data["group"],
                   data["truth_count"], data["country"])


def decision(data, rich_prob, graph_support, threshold, rule):
    chosen = rich_prob >= threshold
    if rule is None:
        return chosen
    minimum_rich, minimum_graph, missing_only = rule
    promoted = ((rich_prob >= minimum_rich) & (graph_support >= minimum_graph))
    if missing_only:
        empty_column = data["names"].index("target_address_empty")
        promoted &= data["x"][:, empty_column] > .5
    return chosen | promoted


def choose(data, rich_prob, graph_support):
    base_threshold = max(np.arange(.5, .951, .025),
                         key=lambda t: score(data, rich_prob >= t)["overall"]["macro_f05"])
    base_threshold = float(round(base_threshold, 3))
    baseline = score(data, rich_prob >= base_threshold)
    best = (baseline["overall"]["macro_f05"], None, baseline)
    for minimum_rich in (.3, .45, .6, .7):
        for minimum_graph in (.75, .85, .92, .97):
            for missing_only in (False, True):
                rule = (minimum_rich, minimum_graph, missing_only)
                value = score(data, decision(data, rich_prob, graph_support,
                                             base_threshold, rule))
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
              "candidate_sets_changed": False,
              "graph_feature_excludes_candidate_frozen_probability": True,
              "directions": {}}
    for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                  ("fresh_10k_v2", "fresh_10k_v1")):
        train, train_graph = cohorts[train_name]
        test, test_graph = cohorts[test_name]
        rich_oof, graph_oof = crossfit(train, train_graph)
        support_oof = support(graph_oof, train_graph, len(train["labels"]))
        base_threshold, dev_base, rule, dev_changed = choose(train, rich_oof, support_oof)
        rich = fit(train["x"], train["labels"])
        graph = graph_model()
        graph.fit(train_graph["x"], train_graph["labels"])
        rich_test = rich.predict_proba(test["x"])[:, 1]
        graph_test = graph.predict_proba(test_graph["x"])[:, 1]
        support_test = support(graph_test, test_graph, len(test["labels"]))
        direction = {
            "train_graph_pairs": len(train_graph["labels"]),
            "train_graph_positive_pairs": int(train_graph["labels"].sum()),
            "test_graph_auc": float(roc_auc_score(test_graph["labels"], graph_test)),
            "test_graph_average_precision": float(average_precision_score(test_graph["labels"], graph_test)),
            "selected_base_threshold": base_threshold,
            "selected_joint_rule": rule,
            "train_crossfit_baseline": dev_base,
            "train_crossfit_joint": dev_changed,
            "test_baseline": score(test, decision(test, rich_test, support_test,
                                                   base_threshold, None)),
            "test_joint": score(test, decision(test, rich_test, support_test,
                                                base_threshold, rule)),
        }
        result["directions"][f"{train_name}_to_{test_name}"] = direction
        print(train_name, "to", test_name, "rule", rule,
              "base", direction["test_baseline"]["overall"]["macro_f05"],
              "joint", direction["test_joint"]["overall"]["macro_f05"],
              "India", direction["test_joint"]["India"]["macro_f05"], flush=True)
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
