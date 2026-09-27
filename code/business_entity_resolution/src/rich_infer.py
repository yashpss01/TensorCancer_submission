"""Rescore a bounded candidate shard with the independently validated rich matcher.

This command reads no labels. Generate candidates with `infer.py run` first,
then supply the same Source-1 shard and complete, unlabeled target TSVs here.
Use one bounded Source-1 shard per invocation to keep memory predictable.
"""

import argparse
import collections
import csv
import itertools
import json
import pathlib
import resource
import shutil
import sqlite3
import time

import numpy as np
from rapidfuzz import fuzz
from unidecode import unidecode
from xgboost import XGBClassifier

from blocking import rows, tokens
from features import FEATURE_NAMES, LEGAL, features, represent
from inference_features import EXTRA_NAMES, GROUP_NAMES, extra_vector, group_matrix
from infer import digest, load_models, target_records

MODELS = pathlib.Path(__file__).resolve().parents[1] / "models"
CORE_NAMES = (
    "frozen_probability", "core_equal", "target_address_empty", "india",
    "log_target_core_frequency", "log_source_core_frequency",
)
RAPID_NAMES = (
    "unicode_core_ratio", "unicode_core_token_set", "unicode_core_wratio",
    "unicode_name_ratio", "unicode_address_ratio",
)
FEATURE_LAYOUT = FEATURE_NAMES + EXTRA_NAMES + GROUP_NAMES[1:] + list(CORE_NAMES) + list(RAPID_NAMES)


def core(name):
    return " ".join(token for token in tokens(name) if token not in LEGAL)


def rapid_values(q, t):
    qcore = unidecode(q["core"])
    tcore = unidecode(t["core"])
    qname = unidecode(q["name"])
    tname = unidecode(t["name"])
    return (
        fuzz.ratio(qcore, tcore) / 100,
        fuzz.token_set_ratio(qcore, tcore) / 100,
        fuzz.WRatio(qcore, tcore) / 100,
        fuzz.ratio(qname, tname) / 100,
        fuzz.ratio(unidecode(q["addr"]), unidecode(t["addr"])) / 100
        if q["addr"] and t["addr"] else 0,
    )


def candidate_rows(source, candidate_tsv):
    lists = []
    with candidate_tsv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Unexpected candidate TSV header")
        for query, record in zip(source, reader):
            if query["entity_id"] != record["source1_entity_id"]:
                raise ValueError("Candidate/source IDs or order differ")
            mids = record["candidate_entity_ids"].split(",") if record["candidate_entity_ids"] else []
            if len(mids) != len(set(mids)):
                raise ValueError("Duplicate candidate ID")
            lists.append(mids)
        if next(reader, None) is not None or len(lists) != len(source):
            raise ValueError("Candidate/source row counts differ")
    return lists


def reference_rows(path, source):
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected baseline matching TSV header")
        records = list(reader)
    if len(records) != len(source) or [r["source1_entity_id"] for r in records] != [q["entity_id"] for q in source]:
        raise ValueError("Baseline/source rows differ")
    return [r["matched_entity_ids"].split(",") if r["matched_entity_ids"] else [] for r in records]


def frequencies(target_files, wanted):
    counter = collections.Counter()
    count = 0
    for path in target_files:
        with path.open(newline="", encoding="utf-8") as stream:
            for record in csv.DictReader(stream, delimiter="\t"):
                value = core(record["business_name"])
                if value in wanted:
                    counter[value] += 1
                count += 1
                if count % 500_000 == 0:
                    print("counted", count, "target names", flush=True)
    return counter, count


def batch_probabilities(queries, lists, targets, counts, frozen_models, rich_model):
    first, group, augmented, (nw, aw, default), (cn, ca, cdefault) = frozen_models
    x, e, reps, offsets, cores = [], [], [], [0], []
    for query, mids in zip(queries, lists):
        qr = represent(query["business_name"], query["business_address"], query["country"])
        current = []
        qcore = core(query["business_name"])
        for mid in mids:
            name, addr, country = targets[mid]
            tr = represent(name, addr, country)
            x.append(features(qr, tr, nw, aw, default))
            e.append(extra_vector(qr, tr, cn, ca, cdefault))
            current.append(tr)
            tcore = core(name)
            cores.append((float(bool(qcore) and qcore == tcore),
                          float(not addr.strip()), float(query["country"] == "India"),
                          np.log1p(counts[tcore]), np.log1p(counts[qcore]),
                          rapid_values(qr, tr)))
        reps.append(current)
        offsets.append(len(x))
    if not x:
        return ([np.empty(0, dtype="float32") for _ in queries],
                [np.empty(0, dtype="float32") for _ in queries])
    X = np.asarray(x, dtype="float32")
    E = np.asarray(e, dtype="float32")
    pair = np.concatenate((X, E), axis=1)
    first_p = first.predict_proba(X)[:, 1].astype("float32")
    A = np.concatenate([
        group_matrix(q, group_reps, first_p[offsets[i]:offsets[i+1]], aw, default)
        for i, (q, group_reps) in enumerate(zip(queries, reps))
    ], axis=0)
    group_input = np.concatenate((pair, A[:, 1:]), axis=1)
    frozen = (.4*group.predict_proba(group_input)[:, 1]
              + .6*augmented.predict_proba(pair)[:, 1]).astype("float32")
    context = np.asarray([
        (frozen[i], same, empty, india, tf, sf, *rapid)
        for i, (same, empty, india, tf, sf, rapid) in enumerate(cores)
    ], dtype="float32")
    rich = np.column_stack((X, E, A[:, 1:], context)).astype("float32")
    if rich.shape[1] != len(FEATURE_LAYOUT):
        raise ValueError("Rich feature width differs from frozen layout")
    changed = rich_model.predict_proba(rich)[:, 1].astype("float32")
    return ([frozen[offsets[i]:offsets[i+1]] for i in range(len(queries))],
            [changed[offsets[i]:offsets[i+1]] for i in range(len(queries))])


def run(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    started = time.monotonic()
    if args.start_row < 0 or args.stop_row is not None and args.stop_row <= args.start_row:
        raise ValueError("Invalid Source-1 row range")
    if args.stop_row is not None and args.stop_row-args.start_row > args.max_source_rows:
        raise ValueError("Requested shard exceeds --max-source-rows")
    last = args.stop_row if args.stop_row is not None else args.start_row+args.max_source_rows+1
    source = list(itertools.islice(rows(args.source1), args.start_row, last))
    if not source or len(source) > args.max_source_rows:
        raise ValueError("Source shard is empty or too large; split into bounded shards")
    if args.stop_row is not None and len(source) != args.stop_row-args.start_row:
        raise ValueError("Source-1 ended before the requested shard range")
    if len({q["entity_id"] for q in source}) != len(source):
        raise ValueError("Duplicate Source-1 ID")
    candidate_sha = digest(args.candidate_tsv)
    old_meta = args.candidate_tsv.parent / "inference_meta.json"
    if old_meta.exists():
        prior = json.loads(old_meta.read_text())
        if prior.get("candidate_sha256") != candidate_sha:
            raise ValueError("Candidate TSV differs from its inference metadata")
        if "start_row" in prior and prior["start_row"] != args.start_row:
            raise ValueError("Candidate shard starts at another Source-1 row")
        if "end_row_exclusive" in prior and prior["end_row_exclusive"] != args.start_row+len(source):
            raise ValueError("Candidate shard ends at another Source-1 row")
    lists = candidate_rows(source, args.candidate_tsv)
    baseline = reference_rows(args.reference_baseline, source)
    db = sqlite3.connect(f"file:{args.index_dir/'index.sqlite'}?mode=ro", uri=True)
    unique = {mid for mids in lists for mid in mids}
    targets = target_records(db, unique)
    db.close()
    wanted = {core(q["business_name"]) for q in source}
    wanted.update(core(targets[mid][0]) for mid in unique)
    counts, scanned = frequencies(args.target_tsv, wanted)
    frozen_models = load_models(args.model_dir)
    rich_path = args.model_dir / "rich_pair_v1v2/rich_pair_model.json"
    rich_manifest_path = args.model_dir / "rich_pair_v1v2/manifest.json"
    rich_manifest = json.loads(rich_manifest_path.read_text())
    if rich_manifest["feature_names"] != FEATURE_LAYOUT or rich_manifest["threshold"] != .775:
        raise ValueError("Unexpected rich model feature or threshold contract")
    if rich_manifest["model_sha256"] != digest(rich_path):
        raise ValueError("Rich model digest mismatch")
    rich_model = XGBClassifier()
    rich_model.load_model(rich_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    match = args.out_dir / "matching_results.tsv"
    temp = args.out_dir / "matching_results.tsv.partial"
    accepted = 0
    with temp.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for start in range(0, len(source), args.batch_size):
            end = min(start+args.batch_size, len(source))
            frozen, changed = batch_probabilities(source[start:end], lists[start:end],
                                                   targets, counts, frozen_models, rich_model)
            for i, (query, mids, p0, p1) in enumerate(zip(source[start:end], lists[start:end], frozen, changed), start):
                original = [mid for mid, probability in zip(mids, p0) if probability >= .74]
                if baseline is not None and original != baseline[i]:
                    raise ValueError(f"Frozen baseline mismatch at S1 row {i}")
                chosen = [mid for mid, probability in zip(mids, p1) if probability >= .775]
                accepted += len(chosen)
                writer.writerow([query["entity_id"], ",".join(chosen)])
            if end % 1000 < args.batch_size or end == len(source):
                print("rich scored", end, "S1 in", round(time.monotonic()-started, 1), "seconds", flush=True)
    temp.rename(match)
    shutil.copy2(args.candidate_tsv, args.out_dir / "candidate_pairs.tsv")
    if digest(args.out_dir / "candidate_pairs.tsv") != candidate_sha:
        raise ValueError("Candidate copy hash mismatch")
    meta = {
        "scope": "bounded label-blind rich matcher rescore; not a Portal result",
        "source1_rows": len(source), "candidate_pairs": sum(map(len, lists)),
        "start_row": args.start_row,
        "end_row_exclusive": args.start_row+len(source),
        "partial_run": args.start_row != 0 or args.stop_row is not None,
        "predicted_pairs": accepted,
        "candidate_sha256": candidate_sha, "matching_sha256": digest(match),
        "source1_sha256": digest(args.source1),
        "target_frequency_rows": scanned,
        "target_frequency_files": [{"path": str(p), "sha256": digest(p)} for p in args.target_tsv],
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "frozen_model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "rich_model_manifest_sha256": digest(rich_manifest_path),
        "seconds": time.monotonic()-started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.out_dir / "inference_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--target-tsv", type=pathlib.Path, nargs=2, required=True)
    p.add_argument("--model-dir", type=pathlib.Path, default=MODELS)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--reference-baseline", type=pathlib.Path)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-source-rows", type=int, default=50_000)
    p.add_argument("--start-row", type=int, default=0, help="Zero-based Source-1 shard start")
    p.add_argument("--stop-row", type=int, help="Exclusive Source-1 shard end")
    args = p.parse_args()
    if args.batch_size < 1 or args.max_source_rows < 1:
        raise ValueError("Batch size and max shard rows must be positive")
    run(args)


if __name__ == "__main__":
    main()
