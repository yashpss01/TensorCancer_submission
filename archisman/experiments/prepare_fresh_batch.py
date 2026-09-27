"""Select a disjoint labeled S1 batch without opening its outcomes for tuning.

The known exposed region covers original S1 rows 1–280,000, including the
supplemental positives. The selected batch excludes shared target ownership
and exact raw name/address/country signatures with that region. This is an
exposure audit, not a guarantee against every semantic duplicate.
"""

import argparse
import csv
import hashlib
import json
import pathlib


def ids(cell):
    return set(cell.split(",")) if cell else set()


def signature(row):
    return tuple(" ".join(row[field].casefold().split()) for field in
                 ("business_name", "business_address", "country"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type=pathlib.Path, required=True)
    parser.add_argument("--truth", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--exposed-rows", type=int, default=280000)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--scan-after", type=int, default=100000)
    args = parser.parse_args()

    records = []
    with args.source1.open(newline="", encoding="utf-8") as sf:
        source = csv.DictReader(sf, delimiter="\t")
        assert source.fieldnames == ["entity_id", "business_name", "business_address", "country"]
        for row_number, record in enumerate(source, 1):
            records.append((row_number, record))
            if row_number == args.exposed_rows + args.scan_after:
                break
    needed_ids = {record["entity_id"] for _, record in records}
    assert len(needed_ids) == len(records), "Duplicate Source 1 ID in the scanned region"
    labels = {}
    with args.truth.open(newline="", encoding="utf-8") as tf:
        truth = csv.DictReader(tf, delimiter="\t")
        assert truth.fieldnames == ["source1_entity_id", "matched_entity_ids"]
        for label in truth:
            sid = label["source1_entity_id"]
            if sid in needed_ids:
                assert sid not in labels, "Duplicate ground-truth Source 1 ID"
                labels[sid] = label
    assert len(labels) == len(needed_ids), "Missing labels for scanned Source 1 records"

    exposed_targets = set()
    exposed_s1 = set()
    exposed_signatures = set()
    selected_targets = set()
    selected_s1 = set()
    selected_signatures = set()
    selected_rows = []
    rejected = {"shared_target": 0, "shared_signature": 0, "duplicate_s1": 0}
    countries = {}
    singleton_count = 0

    for row_number, record in records:
        sid = record["entity_id"]
        label = labels[sid]
        targets = ids(label["matched_entity_ids"])
        key = signature(record)
        if row_number <= args.exposed_rows:
            assert sid not in exposed_s1
            exposed_s1.add(sid)
            exposed_targets.update(targets)
            exposed_signatures.add(key)
            continue
        if sid in exposed_s1 or sid in selected_s1:
            rejected["duplicate_s1"] += 1
            continue
        if targets & exposed_targets or targets & selected_targets:
            rejected["shared_target"] += 1
            continue
        if key in exposed_signatures or key in selected_signatures:
            rejected["shared_signature"] += 1
            continue
        selected_rows.append((row_number, record, label))
        selected_s1.add(sid)
        selected_targets.update(targets)
        selected_signatures.add(key)
        countries[record["country"]] = countries.get(record["country"], 0) + 1
        singleton_count += not targets
        if len(selected_rows) == args.count:
            break

    assert len(selected_rows) == args.count, "Not enough disjoint rows after exposed region"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_out = args.output_dir / "source1.tsv"
    truth_out = args.output_dir / "truth.tsv"
    with source_out.open("w", newline="", encoding="utf-8") as sf, truth_out.open("w", newline="", encoding="utf-8") as tf:
        sw = csv.DictWriter(sf, fieldnames=["entity_id", "business_name", "business_address", "country"], delimiter="\t", lineterminator="\n")
        tw = csv.DictWriter(tf, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter="\t", lineterminator="\n")
        sw.writeheader(); tw.writeheader()
        for _, record, label in selected_rows:
            sw.writerow(record); tw.writerow(label)
    manifest = {
        "status": "selected_not_scored",
        "exposed_original_s1_rows": [1, args.exposed_rows],
        "selected_original_s1_rows": [selected_rows[0][0], selected_rows[-1][0]],
        "selected_count": len(selected_rows),
        "countries": countries,
        "singletons": singleton_count,
        "rejected": rejected,
        "shared_target_ids_with_exposed": 0,
        "shared_raw_signatures_with_exposed": 0,
        "target_ids_in_selected_truth": len(selected_targets),
        "selected_s1_ids_sha256": hashlib.sha256("\n".join(r[1]["entity_id"] for r in selected_rows).encode()).hexdigest(),
        "source1_path": str(args.source1),
        "truth_path": str(args.truth),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
