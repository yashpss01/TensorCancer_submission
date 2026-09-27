"""Compare hard-pair discrimination after neural scoring has finished."""

import argparse
import json
import pathlib

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(labels, rich, neural):
    if len(set(labels)) < 2:
        return {"pairs": len(labels), "positives": int(sum(labels)),
                "rich_auc": None, "reranker_auc": None}
    return {
        "pairs": len(labels), "positives": int(sum(labels)),
        "rich_auc": float(roc_auc_score(labels, rich)),
        "reranker_auc": float(roc_auc_score(labels, neural)),
        "rich_average_precision": float(average_precision_score(labels, rich)),
        "reranker_average_precision": float(average_precision_score(labels, neural)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=pathlib.Path, required=True)
    p.add_argument("--scores", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    label_payload = json.loads(args.labels.read_text())
    score_payload = json.loads(args.scores.read_text())
    if label_payload["input_sha256"] != score_payload["input_sha256"]:
        raise ValueError("Scores and labels belong to different samples")
    labels = np.asarray(label_payload["labels"], dtype=bool)
    rich = np.asarray(label_payload["rich_probability"], dtype=float)
    neural = np.asarray(score_payload["logits"], dtype=float)
    country = np.asarray(label_payload["country"])
    if not len(labels) == len(rich) == len(neural) == len(country) == score_payload["sample_pairs"]:
        raise ValueError("Sample score/label lengths differ")
    result = {
        "scope": "sampled exposed-development pair discrimination only; not per-S1 macro F0.5",
        "train_cohort": label_payload["train_cohort"],
        "test_cohort": label_payload["test_cohort"],
        "model": score_payload["model"],
        "model_revision": score_payload["model_revision"],
        "device": score_payload["device"],
        "scoring_seconds": score_payload["seconds"],
        "overall": metrics(labels, rich, neural),
        "countries": {name: metrics(labels[country == name], rich[country == name],
                                    neural[country == name])
                      for name in ("India", "US")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
