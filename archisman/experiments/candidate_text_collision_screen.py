"""Count mixed-label candidate records with identical text on exposed cohorts.

This is diagnostic only: a mixed-label text collision does not prove that the
labels are wrong or that a model using non-text context cannot separate them.
"""

import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import represent  # noqa: E402
from infer import target_records  # noqa: E402


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def ids(value):
    return set(filter(None, value.split(",")))


def cohort(root):
    source = list(read(root / "source1.tsv"))
    truth = list(read(root / "truth.tsv"))
    candidates = list(read(root / "baseline_10k/candidate_pairs.tsv"))
    if not len(source) == len(truth) == len(candidates):
        raise ValueError("Source/truth/candidate row counts differ")
    for q, t, c in zip(source, truth, candidates):
        if q["entity_id"] != t["source1_entity_id"] or q["entity_id"] != c["source1_entity_id"]:
            raise ValueError("Source/truth/candidate IDs differ")
    target_ids = {mid for c in candidates for mid in ids(c["candidate_entity_ids"])}
    db = sqlite3.connect(f"file:{root/'bounded_index/index.sqlite'}?mode=ro", uri=True)
    targets = target_records(db, target_ids)
    db.close()
    counts = {name: collections.Counter() for name in ("raw", "normalized")}
    for q, t, c in zip(source, truth, candidates):
        known = ids(t["matched_entity_ids"])
        mids = ids(c["candidate_entity_ids"])
        for name, normalized in (("raw", False), ("normalized", True)):
            groups = collections.defaultdict(list)
            for mid in mids:
                target = targets[mid]
                if normalized:
                    rep = represent(*target)
                    key = (rep["name"], rep["addr"], rep["country"])
                else:
                    key = target
                groups[key].append(mid)
            mixed = [members for members in groups.values()
                     if any(mid in known for mid in members)
                     and any(mid not in known for mid in members)]
            bucket = counts[name]
            bucket["groups"] += 1
            bucket["candidate_pairs"] += len(mids)
            bucket["mixed_groups"] += bool(mixed)
            bucket["mixed_classes"] += len(mixed)
            bucket["mixed_true_pairs"] += sum(mid in known for members in mixed for mid in members)
            bucket["mixed_false_pairs"] += sum(mid not in known for members in mixed for mid in members)
    return {name: dict(bucket) for name, bucket in counts.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    result = {"scope": "exposed v1/v2 training cohorts, reduced target pools; diagnostic only",
              "cohorts": {}}
    for name in ("fresh_10k_v1", "fresh_10k_v2"):
        result["cohorts"][name] = cohort(args.root / "work" / name)
        print(name, result["cohorts"][name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
