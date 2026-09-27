"""Extend an old reduced target pool with every positive for a fresh S1 batch.

Labels only choose which targets must exist in the evaluation pool. They do
not influence retrieval ranking or prediction. This pool is intentionally
reduced and cannot establish full-corpus performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sqlite3
import sys
import time


def tsv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-index", type=pathlib.Path, required=True)
    parser.add_argument("--old-extra", type=pathlib.Path, required=True)
    parser.add_argument("--truth", type=pathlib.Path, required=True)
    parser.add_argument("--source2", type=pathlib.Path, required=True)
    parser.add_argument("--source3", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"
    sys.path.insert(0, str(src))
    from blocking import fields, tokens
    from normalize import address, bigrams
    from phonetic import grams

    started = time.monotonic()
    required = {
        match for row in tsv_rows(args.truth)
        for match in row["matched_entity_ids"].split(",") if match
    }
    shutil.copy2(args.old_index, args.out_dir / "index.sqlite")
    shutil.copy2(args.old_extra, args.out_dir / "extra.sqlite")
    db = sqlite3.connect(args.out_dir / "index.sqlite")
    extra = sqlite3.connect(args.out_dir / "extra.sqlite")
    old_count = db.execute("SELECT count(*) FROM records").fetchone()[0]
    present = {row[0] for row in db.execute("SELECT id FROM records")}
    missing = required - present
    found = set()
    next_id = db.execute("SELECT max(rowid) FROM records").fetchone()[0] + 1
    batch_records, batch_search, batch_extra = [], [], []
    for source in (args.source2, args.source3):
        for row in tsv_rows(source):
            mid = row["entity_id"]
            if mid not in missing:
                continue
            if mid in found:
                raise ValueError(f"Duplicate required target: {mid}")
            found.add(mid)
            rid = next_id
            next_id += 1
            name, addr, country = row["business_name"], row["business_address"], row["country"]
            batch_records.append((rid, mid, name, addr, country))
            batch_search.append((rid, *fields(row), " ".join(grams(name))))
            batch_extra.append((rid, " ".join(tokens(name)), " ".join(address(addr)),
                                " ".join(sorted(bigrams(name)))))
            if len(batch_records) >= 5000:
                db.executemany("INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)", batch_records)
                db.executemany("INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)", batch_search)
                extra.executemany("INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)", batch_extra)
                db.commit()
                extra.commit()
                batch_records, batch_search, batch_extra = [], [], []
    if found != missing:
        raise ValueError(f"Missing {len(missing - found)} required targets")
    if batch_records:
        db.executemany("INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)", batch_records)
        db.executemany("INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)", batch_search)
        extra.executemany("INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)", batch_extra)
        db.commit()
        extra.commit()
    total = db.execute("SELECT count(*) FROM records").fetchone()[0]
    assert total == old_count + len(missing)
    assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert extra.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    db.close()
    extra.close()
    meta = {
        "target_count": total,
        "old_pool_target_count": old_count,
        "fresh_required_targets": len(required),
        "fresh_required_preexisting": len(required) - len(missing),
        "fresh_required_added": len(missing),
        "target_policy": "old sealed 409141-target pool plus all fresh batch true targets",
        "scope": "reduced pool; may be optimistic vs full 10.32M-target corpus",
        "seconds": time.monotonic() - started,
    }
    (args.out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
