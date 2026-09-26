"""Read-only, bounded profile of the frozen inference path.

Uses a selected S1 file and an existing target index. It never reads labels to
choose candidates and never changes the production model or index. Optional
truth is used only to count each blocking route's contribution afterward.
"""

import argparse
import collections
import cProfile
import csv
import io
import json
import pathlib
import pstats
import resource
import sqlite3
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
import infer  # noqa: E402

SQL_TIMES = collections.Counter()
SQL_CALLS = collections.Counter()


class TimedConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if "FROM vocab" in sql:
            category = "vocabulary"
        elif "MATCH ?" in sql:
            category = "fts_search"
        elif "FROM records" in sql:
            category = "record_fetch"
        else:
            category = "other"
        started = time.perf_counter()
        try:
            return super().execute(sql, parameters)
        finally:
            SQL_TIMES[category] += time.perf_counter() - started
            SQL_CALLS[category] += 1


def load_truth(path):
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["source1_entity_id"]: set(r["matched_entity_ids"].split(","))
                if r["matched_entity_ids"] else set()
                for r in csv.DictReader(handle, delimiter="\t")}


def run(args):
    assert args.rows > 0 and args.batch_size > 0
    queries = list(__import__("itertools").islice(infer.rows(args.source1), args.rows))
    assert len(queries) == args.rows
    truth = load_truth(args.truth)
    profiler = cProfile.Profile() if args.cprofile else None
    if args.sql_profile:
        original_connect = sqlite3.connect
        def timed_connect(*a, **kw):
            return original_connect(*a, factory=TimedConnection, **kw)
        sqlite3.connect = timed_connect
    infer.init_worker(args.index_dir)
    models = infer.load_models(args.model_dir)
    db = __import__("sqlite3").connect(f"file:{args.index_dir/'index.sqlite'}?mode=ro", uri=True)
    routes = collections.defaultdict(lambda: [0, 0, 0])
    elapsed = collections.Counter()
    count = collections.Counter()
    candidates = []
    predictions = []
    marginal_route = collections.Counter()
    unique_candidates = set()
    t0 = time.perf_counter()
    if profiler:
        profiler.enable()
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start:start + args.batch_size]
        selected = []
        for q in batch:
            t = time.perf_counter()
            base = infer.ENGINE_BASE.retrieve(q)
            elapsed["baseline_retrieval"] += time.perf_counter() - t
            t = time.perf_counter()
            improved = infer.ENGINE_EXTRA.retrieve(q, base)
            elapsed["rescue_retrieval"] += time.perf_counter() - t
            t = time.perf_counter()
            mids = infer.select(improved, .5, 16)
            elapsed["candidate_select"] += time.perf_counter() - t
            selected.append(mids)
            known = truth.get(q["entity_id"], set())
            for route, ids in base["routes"].items():
                route_set = set(ids)
                stats = routes[route]
                stats[0] += len(ids)
                stats[1] += len(route_set)
                stats[2] += len(route_set & known)
                others = set().union(*(set(v) for k, v in base["routes"].items() if k != route))
                marginal_route[route] += len((route_set - others) & known)
            count["raw_stage1_pairs"] += base["raw_count"]
            count["raw_stage2_pairs"] += improved["raw_extra"]
            count["selected_pairs"] += len(mids)
            count["selected_true"] += len(set(mids) & known)
            count["true_links"] += len(known)
            candidates.append((q["entity_id"], mids))
            unique_candidates.update(mids)
        t = time.perf_counter()
        matches = infer.match_batch(batch, selected, db, models)
        elapsed["feature_and_model"] += time.perf_counter() - t
        predictions.extend((q["entity_id"], mids) for q, mids in zip(batch, matches))
    if profiler:
        profiler.disable()
    elapsed["total_loop"] = time.perf_counter() - t0
    t = time.perf_counter()
    with io.StringIO() as buffer:
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        writer.writerows((sid, ",".join(mids)) for sid, mids in candidates)
        candidate_bytes = len(buffer.getvalue().encode())
        buffer.seek(0); buffer.truncate(0)
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        writer.writerows((sid, ",".join(mids)) for sid, mids in predictions)
        match_bytes = len(buffer.getvalue().encode())
    elapsed["serialize_in_memory"] = time.perf_counter() - t
    db.close()
    if profiler:
        report = io.StringIO()
        pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(30)
        args.profile_text.write_text(report.getvalue())
    result = {
        "scope": "bounded read-only frozen inference profile; reduced target pool",
        "rows": len(queries),
        "target_count": json.loads((args.index_dir / "index_meta.json").read_text())["target_count"],
        "cprofile_enabled": bool(profiler),
        "stage_seconds": dict(elapsed),
        "rows_per_second": len(queries) / elapsed["total_loop"],
        "counts": dict(count),
        "route_counts": dict(routes),
        "route_exclusive_true_links": dict(marginal_route),
        "unique_selected_target_ids": len(unique_candidates),
        "sql_seconds": dict(SQL_TIMES),
        "sql_calls": dict(SQL_CALLS),
        "serialized_bytes": {"candidate": candidate_bytes, "matching": match_bytes},
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--model-dir", type=pathlib.Path,
                   default=ROOT / "code/business_entity_resolution/models")
    p.add_argument("--truth", type=pathlib.Path)
    p.add_argument("--rows", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--cprofile", action="store_true")
    p.add_argument("--sql-profile", action="store_true")
    p.add_argument("--report", type=pathlib.Path, required=True)
    p.add_argument("--profile-text", type=pathlib.Path)
    a = p.parse_args()
    if a.cprofile and a.profile_text is None:
        p.error("--profile-text required with --cprofile")
    run(a)
