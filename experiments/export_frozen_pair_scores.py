"""Cache frozen pair probabilities without rerunning candidate retrieval.

This is a development artifact, not a submission command. It never reads
labels. The optional reference match TSV verifies exact frozen decisions.
"""

import argparse
import hashlib
import itertools
import json
import pathlib
import sqlite3
import sys
import time

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from blocking import rows  # noqa: E402
from infer import digest, load_models, predict_batch_probabilities  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--model-dir", type=pathlib.Path,
                   default=ROOT / "code/business_entity_resolution/models")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--reference-matches", type=pathlib.Path)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    if args.batch_size < 1 or args.output.exists():
        raise ValueError("Positive batch size and a new output path are required")
    started = time.monotonic()
    models = load_models(args.model_dir)
    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    candidate_meta = args.candidate_tsv.parent / "inference_meta.json"
    if candidate_meta.exists():
        meta = json.loads(candidate_meta.read_text())
        if meta.get("candidate_sha256") != digest(args.candidate_tsv):
            raise ValueError("Candidate TSV digest mismatch")
    sids, mids, probs = [], [], []
    matching_digest = hashlib.sha256()
    matching_digest.update(b"source1_entity_id\tmatched_entity_ids\n")
    count = 0
    with args.candidate_tsv.open(encoding="utf-8", newline="") as stream:
        if stream.readline() != "source1_entity_id\tcandidate_entity_ids\n":
            raise ValueError("Unexpected candidate header")
        source_iter = rows(args.source1)
        while queries := list(itertools.islice(source_iter, args.batch_size)):
            candidate_lists = []
            for query in queries:
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    raise ValueError("Candidate TSV ended early")
                cells = line[:-1].split("\t")
                if len(cells) != 2 or cells[0] != query["entity_id"]:
                    raise ValueError("Candidate/source order mismatch")
                candidate_ids = cells[1].split(",") if cells[1] else []
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise ValueError("Duplicate candidate ID")
                candidate_lists.append(candidate_ids)
            group_probs = predict_batch_probabilities(queries, candidate_lists, db, models)
            for query, candidate_ids, scores in zip(queries, candidate_lists, group_probs):
                if len(candidate_ids) != len(scores):
                    raise ValueError("Probability/candidate length mismatch")
                chosen = []
                for mid, score in zip(candidate_ids, scores):
                    sids.append(query["entity_id"])
                    mids.append(mid)
                    probs.append(float(score))
                    if score >= .74:
                        chosen.append(mid)
                matching_digest.update((query["entity_id"] + "\t" + ",".join(chosen) + "\n").encode())
                count += 1
            if count % 1000 < args.batch_size:
                print(f"scored {count} S1 rows", flush=True)
        if stream.readline():
            raise ValueError("Candidate TSV has extra rows")
    db.close()
    matching_sha = matching_digest.hexdigest()
    if args.reference_matches and matching_sha != digest(args.reference_matches):
        raise ValueError("Frozen predictions differ from reference match TSV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"s1_id": sids, "target_id": mids, "frozen_probability": probs}).write_parquet(
        args.output, compression="zstd")
    result = {
        "source1_rows": count, "candidate_pairs": len(mids),
        "candidate_sha256": digest(args.candidate_tsv),
        "matching_sha256": matching_sha,
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "seconds": time.monotonic() - started,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
