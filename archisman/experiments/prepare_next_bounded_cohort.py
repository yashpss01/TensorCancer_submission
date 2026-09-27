"""Seal the next disjoint training-S1 cohort before batched retrieval scoring.

The selected positive labels are used solely to construct and later score a
positive-enriched reduced target pool. Never treat its result as a full-test
or Portal measurement.
"""

import argparse
import csv
import hashlib
import itertools
import json
import pathlib
import time


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, fieldnames, records):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=pathlib.Path, required=True)
    parser.add_argument("--ground-truth", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--start", type=int, default=310_000)
    parser.add_argument("--stop", type=int, default=320_000)
    parser.add_argument("--retriever-code", type=pathlib.Path, required=True)
    parser.add_argument("--rich-manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.stop <= args.start:
        raise ValueError("Invalid zero-based cohort bounds")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Refusing to modify an existing sealed cohort")
    chosen = list(itertools.islice(rows(args.source1), args.start, args.stop))
    if len(chosen) != args.stop-args.start:
        raise ValueError("Source-1 ended before requested stop")
    ids = [record["entity_id"] for record in chosen]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate selected Source-1 ID")
    selected = set(ids)
    truth = {}
    for record in rows(args.ground_truth):
        sid = record["source1_entity_id"]
        if sid in selected:
            if sid in truth:
                raise ValueError("Duplicate selected truth row")
            truth[sid] = record["matched_entity_ids"]
    if set(truth) != selected:
        raise ValueError("Missing selected Source-1 truth rows")
    owner = {}
    for sid, value in truth.items():
        for mid in value.split(",") if value else ():
            if mid in owner:
                raise ValueError("One positive target belongs to two selected S1")
            owner[mid] = sid
    for record in rows(args.ground_truth):
        sid = record["source1_entity_id"]
        if sid in selected:
            continue
        for mid in record["matched_entity_ids"].split(",") if record["matched_entity_ids"] else ():
            if mid in owner:
                raise ValueError("Selected positive target also belongs to another S1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.out_dir / "source1.tsv"
    truth_path = args.out_dir / "truth.tsv"
    write(source_path, ["entity_id", "business_name", "business_address", "country"], chosen)
    write(truth_path, ["source1_entity_id", "matched_entity_ids"],
          ({"source1_entity_id": sid, "matched_entity_ids": truth[sid]} for sid in ids))
    manifest = {
        "scope": "zero-based original training Source-1 rows; sealed reduced-pool confirmation",
        "start": args.start, "stop": args.stop,
        "source1_rows": len(chosen), "positive_links": len(owner),
        "source1_sha256": sha(source_path), "truth_sha256": sha(truth_path),
        "retriever_code_sha256": sha(args.retriever_code),
        "rich_manifest_sha256": sha(args.rich_manifest),
        "candidate_rule": "forward six-view sparse top-64 union per S1; lexically sorted target IDs",
        "target_policy": "copy prior bounded pool then add all positives for this cohort",
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "seal.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
