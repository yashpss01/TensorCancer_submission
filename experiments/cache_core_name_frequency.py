"""Cache label-blind normalized-name frequencies for candidate experiments."""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sqlite3
import sys
import time

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from blocking import tokens  # noqa: E402
from features import LEGAL  # noqa: E402


def digest(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core(name: str) -> str:
    return " ".join(token for token in tokens(name) if token not in LEGAL)


def ids(value: str) -> list[str]:
    return value.split(",") if value else []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--frequency-tsv", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Use a new output directory")
    started = time.monotonic()
    with args.source1.open(newline="", encoding="utf-8") as stream:
        sources = list(csv.DictReader(stream, delimiter="\t"))
    candidate_ids = set()
    with args.candidate_tsv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(sources) != len(candidate_rows):
        raise ValueError("Source/candidate row count differs")
    for source, candidate in zip(sources, candidate_rows):
        if source["entity_id"] != candidate["source1_entity_id"]:
            raise ValueError("Source/candidate row order differs")
        row_ids = ids(candidate["candidate_entity_ids"])
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("Duplicate candidate")
        candidate_ids.update(row_ids)
    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    target_core = {}
    target_address_empty = {}
    ordered_ids = sorted(candidate_ids)
    for start in range(0, len(ordered_ids), 900):
        batch = ordered_ids[start:start + 900]
        query = "SELECT id,name,address FROM records WHERE id IN (" + ",".join("?" * len(batch)) + ")"
        for mid, name, address in db.execute(query, batch):
            target_core[mid] = core(name)
            target_address_empty[mid] = not address.strip()
    db.close()
    if len(target_core) != len(candidate_ids):
        raise ValueError("Candidate missing from target index")
    source_core = {row["entity_id"]: core(row["business_name"]) for row in sources}
    wanted = set(target_core.values()) | set(source_core.values())
    frequency = collections.Counter()
    read_count = 0
    for path in args.frequency_tsv:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                value = core(row["business_name"])
                if value in wanted:
                    frequency[value] += 1
                read_count += 1
                if read_count % 500_000 == 0:
                    print(f"counted {read_count:,} target names in {time.monotonic()-started:.1f}s", flush=True)
    args.out_dir.mkdir(parents=True)
    target = pl.DataFrame({
        "target_id": ordered_ids,
        "target_core": [target_core[mid] for mid in ordered_ids],
        "target_core_frequency": [frequency[target_core[mid]] for mid in ordered_ids],
        "target_address_empty": [target_address_empty[mid] for mid in ordered_ids],
    })
    target.write_parquet(args.out_dir / "target_core.parquet", compression="zstd")
    source = pl.DataFrame({
        "s1_id": [row["entity_id"] for row in sources],
        "source_core": [source_core[row["entity_id"]] for row in sources],
        "source_core_frequency": [frequency[source_core[row["entity_id"]]] for row in sources],
    })
    source.write_parquet(args.out_dir / "source_core.parquet", compression="zstd")
    metadata = {
        "source1_rows": len(sources), "candidate_target_ids": len(candidate_ids),
        "frequency_target_rows": read_count, "wanted_core_names": len(wanted),
        "source1_sha256": digest(args.source1),
        "candidate_sha256": digest(args.candidate_tsv),
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "frequency_files": [
            {"path": str(path), "sha256": digest(path)} for path in args.frequency_tsv
        ],
        "seconds": time.monotonic() - started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
