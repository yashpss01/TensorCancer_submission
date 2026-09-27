"""Prepare balanced hard-pair training text from one exposed cohort only."""

import argparse
import csv
import hashlib
import json
import pathlib
import sqlite3
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from infer import target_records  # noqa: E402


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def serialize(name, address, country, style):
    if style == "fields":
        return (f"[COL] name [VAL] {name} [COL] address [VAL] {address} "
                f"[COL] country [VAL] {country}")
    return f"Business name: {name}. Address: {address}. Country: {country}."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=pathlib.Path, required=True)
    p.add_argument("--max-per-class", type=int, default=5000)
    p.add_argument("--style", choices=("plain", "fields"), default="plain")
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.max_per_class < 1 or args.output.exists():
        raise ValueError("Use a positive class cap and new output path")
    source = read(args.cohort / "source1.tsv")
    truth = read(args.cohort / "truth.tsv")
    if [r["entity_id"] for r in source] != [r["source1_entity_id"] for r in truth]:
        raise ValueError("Source/truth rows differ")
    known = [set(filter(None, r["matched_entity_ids"].split(","))) for r in truth]
    frame = pl.read_parquet(args.cohort / "frozen_pair_scores.parquet")
    group_map = {r["entity_id"]: i for i, r in enumerate(source)}
    groups = np.asarray([group_map[sid] for sid in frame["s1_id"].to_list()], dtype=np.int32)
    mids = frame["target_id"].to_list()
    sids = frame["s1_id"].to_list()
    frozen_p = frame["frozen_probability"].to_numpy()
    labels = np.asarray([mid in known[groups[j]] for j, mid in enumerate(mids)], dtype=bool)
    eligible = np.flatnonzero(frozen_p >= .01)
    ordered = sorted(eligible, key=lambda j: hashlib.sha256(
        (sids[j] + "\0" + mids[j]).encode()).digest())
    positive = [j for j in ordered if labels[j]][:args.max_per_class]
    negative = [j for j in ordered if not labels[j]][:args.max_per_class]
    if len(positive) < args.max_per_class or len(negative) < args.max_per_class:
        raise ValueError("Insufficient hard positives or negatives for requested cap")
    selected = sorted(positive + negative)
    db = sqlite3.connect(f"file:{args.cohort/'bounded_index/index.sqlite'}?mode=ro", uri=True)
    targets = target_records(db, {mids[j] for j in selected})
    db.close()
    records = []
    for j in selected:
        q = source[groups[j]]
        name, addr, country = targets[mids[j]]
        records.append({
            "query": serialize(q["business_name"], q["business_address"], q["country"], args.style),
            "target": serialize(name, addr, country, args.style),
            "label": int(labels[j]),
        })
    payload = {
        "scope": "balanced hard-pair training from exposed cohort only",
        "cohort": args.cohort.name,
        "style": args.style,
        "selection": "frozen probability >=0.01, stable pair hash, up to class cap per label",
        "positive_pairs": len(positive), "negative_pairs": len(negative),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False)+"\n")
    print("prepared", len(records), "training pairs from", args.cohort.name, flush=True)


if __name__ == "__main__":
    main()
