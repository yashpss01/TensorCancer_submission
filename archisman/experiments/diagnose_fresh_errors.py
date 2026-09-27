"""Post-freeze error strata on the completed fresh 10k bounded evaluation."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import sys


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def ids(value):
    return set(value.split(",")) if value else set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--truth", type=pathlib.Path, required=True)
    p.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--baseline-tsv", type=pathlib.Path, required=True)
    p.add_argument("--changed-tsv", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    code = pathlib.Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"
    sys.path.insert(0, str(code))
    from features import represent

    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    records = {mid: (name, addr, country) for mid, name, addr, country in
               db.execute("SELECT id,name,address,country FROM records")}
    db.close()
    buckets = {
        "baseline": collections.defaultdict(collections.Counter),
        "changed": collections.defaultdict(collections.Counter),
    }
    for q, t, c, b, a in zip(rows(args.source1), rows(args.truth), rows(args.candidate_tsv),
                              rows(args.baseline_tsv), rows(args.changed_tsv), strict=True):
        sid = q["entity_id"]
        assert sid == t["source1_entity_id"] == c["source1_entity_id"] == b["source1_entity_id"] == a["source1_entity_id"]
        qr = represent(q["business_name"], q["business_address"], q["country"])
        truth = ids(t["matched_entity_ids"])
        candidates = ids(c["candidate_entity_ids"])
        for name, prediction in (("baseline", b), ("changed", a)):
            accepted = ids(prediction["matched_entity_ids"])
            assert accepted <= candidates
            for country in ("overall", q["country"]):
                counter = buckets[name][country]
                counter["groups"] += 1
                counter["retrieval_fn"] += len(truth - candidates)
                counter["matcher_fn"] += len((truth & candidates) - accepted)
                counter["false_accepted"] += len(accepted - truth)
                for kind, pair_ids in (("rejected_true", (truth & candidates) - accepted),
                                       ("false_accepted", accepted - truth)):
                    for mid in pair_ids:
                        tr = represent(*records[mid])
                        conditions = {
                            "target_address_empty": not records[mid][1].strip(),
                            "unicode_status_mismatch": qr["unicode_name"] != tr["unicode_name"],
                            "address_number_conflict": bool(qr["nums"] and tr["nums"] and not qr["nums"] & tr["nums"]),
                            "exact_core_name": bool(qr["core"] and qr["core"] == tr["core"]),
                        }
                        for condition, present in conditions.items():
                            if present:
                                counter[f"{kind}_{condition}"] += 1
    output = {
        "scope": "fresh 10k, 442904-target reduced pool; post-freeze overlapping error strata",
        "pipelines": {name: {country: dict(counter) for country, counter in groups.items()}
                      for name, groups in buckets.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["pipelines"]["changed"], indent=2), flush=True)


if __name__ == "__main__":
    main()
