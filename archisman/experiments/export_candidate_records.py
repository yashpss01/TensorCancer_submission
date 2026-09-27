"""Export only retrieved target records from a read-only bounded index."""

import argparse
import json
import pathlib
import sqlite3

import polars as pl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-root", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    paths = sorted(a.candidate_root.glob("*/candidates-*.parquet"))
    if not paths:
        raise FileNotFoundError("No candidate Parquet shards")
    wanted = sorted(set().union(*(set(pl.read_parquet(path, columns=["q_id"])["q_id"].to_list())
                                  for path in paths)))
    db = sqlite3.connect(f"file:{a.index_dir/'index.sqlite'}?mode=ro", uri=True)
    rows = []
    for start in range(0, len(wanted), 900):
        mids = wanted[start:start + 900]
        query = "SELECT id,name,address,country FROM records WHERE id IN (" + ",".join("?" * len(mids)) + ")"
        rows.extend(db.execute(query, mids))
    db.close()
    if len(rows) != len(wanted) or len({r[0] for r in rows}) != len(wanted):
        raise ValueError("Missing or duplicate retrieved target records")
    frame = pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(a.output, compression="zstd")
    print(json.dumps({"records": frame.height, "candidate_shards": len(paths),
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
