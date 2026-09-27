"""Development-only cross-cohort screen of cross-source anchor evidence."""

import argparse
import json
import pathlib
import time

import numpy as np

from core_frequency_model_screen import metrics
from rich_pair_model_screen import choose_threshold, fit, load_cohort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--pseudo", action="store_true", help="append frozen pseudo-query scores")
    args = p.parse_args()
    started = time.monotonic()
    cohorts = {}
    for cohort in ("fresh_10k_v1", "fresh_10k_v2"):
        root = args.root / "work" / cohort
        data = load_cohort(root)
        graph_path = root / "cross_source_anchor.npy"
        graph_meta = json.loads(graph_path.with_suffix(".json").read_text())
        rich_meta = json.loads((root / "rich_pair_features.json").read_text())
        if graph_meta["source1_sha256"] != rich_meta["source1_sha256"] or graph_meta["pair_scores_sha256"] != rich_meta["pair_scores_sha256"]:
            raise ValueError("Anchor and rich features belong to different pairs")
        graph = np.load(graph_path, mmap_mode="r")
        if graph.shape != (len(data["x"]), len(graph_meta["feature_names"])):
            raise ValueError("Anchor feature layout mismatch")
        extras = [graph]
        if args.pseudo:
            pseudo_path = root / "pseudo_query_scores.npy"
            pseudo_meta = json.loads(pseudo_path.with_suffix(".json").read_text())
            if pseudo_meta["source1_sha256"] != rich_meta["source1_sha256"] or pseudo_meta["pair_scores_sha256"] != rich_meta["pair_scores_sha256"]:
                raise ValueError("Pseudo-query scores belong to different pairs")
            pseudo = np.load(pseudo_path, mmap_mode="r")
            if pseudo.shape != (len(data["x"]), len(pseudo_meta["feature_names"])):
                raise ValueError("Pseudo-query feature layout mismatch")
            extras.append(pseudo)
        data["x"] = np.column_stack((data["x"], *extras)).astype(np.float32)
        cohorts[cohort] = data
    result = {"scope": "exposed first and second fresh 10k only; cross-cohort development, no sealed confirmation",
              "pseudo_query_scores": args.pseudo,
              "anchor_feature_names": graph_meta["feature_names"], "directions": {}}
    for train_name, test_name in (("fresh_10k_v1", "fresh_10k_v2"),
                                  ("fresh_10k_v2", "fresh_10k_v1")):
        train, test = cohorts[train_name], cohorts[test_name]
        columns = list(range(train["x"].shape[1]))
        threshold, crossfit = choose_threshold(train, columns)
        model = fit(train["x"], train["labels"])
        scores = model.predict_proba(test["x"])[:, 1]
        outcome = metrics(scores >= threshold, test["labels"], test["group"],
                          test["truth_count"], test["country"])
        result["directions"][f"{train_name}_to_{test_name}"] = {
            "threshold_from_train_crossfit": threshold,
            "train_crossfit": crossfit, "test": outcome,
        }
        print(train_name, "to", test_name, "F0.5", outcome["overall"]["macro_f05"],
              "India", outcome["India"]["macro_f05"], flush=True)
    result["seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
