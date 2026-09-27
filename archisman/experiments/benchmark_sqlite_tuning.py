"""Benchmark SQLite read settings without changing retrieval SQL or candidates."""

import argparse
import hashlib
import itertools
import json
import pathlib
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
import infer  # noqa: E402


def once(index_dir, queries, setting):
    print(f"Starting {setting} on {len(queries)} S1 rows", flush=True)
    infer.init_worker(index_dir)
    connections = [infer.ENGINE_BASE.db, infer.ENGINE_EXTRA.source,
                   infer.ENGINE_EXTRA.db]
    applied = []
    for connection in connections:
        if setting == "mmap_cache":
            connection.execute("PRAGMA mmap_size=1073741824")
            connection.execute("PRAGMA cache_size=-262144")
        applied.append({"mmap_size": connection.execute("PRAGMA mmap_size").fetchone()[0],
                        "cache_size": connection.execute("PRAGMA cache_size").fetchone()[0]})
    timings = {"baseline": 0., "rescue": 0., "selection": 0.}
    rows = []
    for query in queries:
        start = time.perf_counter()
        base = infer.ENGINE_BASE.retrieve(query)
        timings["baseline"] += time.perf_counter() - start
        start = time.perf_counter()
        improved = infer.ENGINE_EXTRA.retrieve(query, base)
        timings["rescue"] += time.perf_counter() - start
        start = time.perf_counter()
        selected = infer.select(improved, .5, 16)
        timings["selection"] += time.perf_counter() - start
        rows.append(query["entity_id"] + "\t" + ",".join(selected) + "\n")
        if len(rows) % (1 if len(queries) <= 10 else 10) == 0:
            print(f"{setting}: {len(rows)}/{len(queries)} rows, "
                  f"{sum(timings.values()):.1f}s", flush=True)
    for connection in connections:
        connection.close()
    payload = "".join(rows).encode()
    return {"setting": setting, "rows": len(queries), "seconds": timings,
            "total_seconds": sum(timings.values()),
            "candidate_pairs": sum(row.count(",") + bool(row.split("\t", 1)[1].strip()) for row in rows),
            "candidate_sha256": hashlib.sha256(payload).hexdigest(),
            "effective_pragmas": applied,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--rows", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--report", type=pathlib.Path, required=True)
    a = p.parse_args()
    queries = list(itertools.islice(infer.rows(a.source1), a.start, a.start + a.rows))
    assert len(queries) == a.rows
    runs = [once(a.index_dir, queries, setting)
            for setting in ("default", "mmap_cache", "mmap_cache", "default")]
    assert len({run["candidate_sha256"] for run in runs}) == 1, "Candidate lists changed"
    result = {"rows": a.rows, "runs": runs,
              "stable_candidate_sha256": runs[0]["candidate_sha256"]}
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
