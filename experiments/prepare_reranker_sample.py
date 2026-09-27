"""Prepare a label-blind hard-pair sample and separate local truth file.

The sample is selected by a rich model fitted on the other exposed cohort
and a stable pair hash. The scoring process reads only input.json, never the
separate labels.json. Neither cohort is an untouched confirmation batch.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
import polars as pl

from rich_pair_model_screen import fit, load_cohort

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from infer import target_records  # noqa: E402


def read_source(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--train", choices=("fresh_10k_v1", "fresh_10k_v2"), required=True)
    p.add_argument("--test", choices=("fresh_10k_v1", "fresh_10k_v2"), required=True)
    p.add_argument("--max-pairs", type=int, default=1000)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.train == args.test or args.max_pairs < 1:
        raise ValueError("Use different exposed train/test cohorts and positive sample size")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    started = time.monotonic()
    train = load_cohort(args.root / "work" / args.train)
    test_root = args.root / "work" / args.test
    test = load_cohort(test_root)
    model = fit(train["x"], train["labels"])
    rich_prob = model.predict_proba(test["x"])[:, 1]
    frame = pl.read_parquet(test_root / "frozen_pair_scores.parquet")
    mids = frame["target_id"].to_list()
    sids = frame["s1_id"].to_list()
    if len(mids) != len(rich_prob):
        raise ValueError("Score/candidate rows differ")
    eligible = np.flatnonzero((rich_prob >= .05) & (rich_prob <= .95))
    ranked = sorted(eligible, key=lambda j: hashlib.sha256(
        (sids[j] + "\0" + mids[j]).encode()).digest())
    selected = np.asarray(ranked[:args.max_pairs], dtype=np.int32)
    if len(selected) < args.max_pairs:
        raise ValueError("Fewer eligible pairs than requested")
    sources = read_source(test_root / "source1.tsv")
    db = sqlite3.connect(f"file:{test_root/'bounded_index/index.sqlite'}?mode=ro", uri=True)
    targets = target_records(db, {mids[j] for j in selected})
    db.close()
    pairs = []
    for j in selected:
        q = sources[test["group"][j]]
        name, addr, country = targets[mids[j]]
        pairs.append({
            "query": f"Business name: {q['business_name']}. Address: {q['business_address']}. Country: {q['country']}.",
            "target": f"Business name: {name}. Address: {addr}. Country: {country}.",
        })
    payload = {
        "scope": "exposed development hard-pair sample; no labels in this scoring input",
        "train_cohort": args.train, "test_cohort": args.test,
        "selection": "rich probability 0.05–0.95 then stable pair-hash sample",
        "eligible_pairs": len(eligible), "sample_pairs": len(selected),
        "pairs": pairs,
    }
    labels = test["labels"][selected]
    label_payload = {
        "scope": "local evaluation only; separate from neural scoring input",
        "train_cohort": args.train, "test_cohort": args.test,
        "labels": labels.astype(int).tolist(),
        "rich_probability": rich_prob[selected].astype(float).tolist(),
        "country": test["country"][test["group"][selected]].tolist(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.out_dir / "input.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False)+"\n")
    label_payload["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    (args.out_dir / "labels.json").write_text(json.dumps(label_payload)+"\n")
    print("prepared", len(selected), "of", len(eligible), "hard pairs;",
          int(labels.sum()), "known positives, in", round(time.monotonic()-started, 1),
          "seconds", flush=True)


if __name__ == "__main__":
    main()
