"""Generate complete challenge TSVs with the frozen 98.7707% rich matcher.

This is an orchestration entry point. It does not change retrieval, features,
weights, or the 0.775 decision threshold. Finished shards can be resumed; a
partial shard is never silently reused or mistaken for a complete submission.
"""

import argparse
import json
import pathlib
import subprocess
import sys

from blocking import rows
from infer import digest, merge


HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE.parent / "models"


def complete_shard(directory, start, stop, source_digest, index_digest,
                   base_manifest_digest, rich_manifest_digest=None,
                   target_digests=None):
    meta_path = directory / "inference_meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    if (meta.get("start_row") != start or
            meta.get("end_row_exclusive") != stop or
            meta.get("source1_rows") != stop - start or
            meta.get("index_meta_sha256") != index_digest):
        return False
    if rich_manifest_digest is None:
        if meta.get("model_manifest_sha256") != base_manifest_digest:
            return False
    elif (meta.get("source1_sha256") != source_digest or
          meta.get("frozen_model_manifest_sha256") != base_manifest_digest or
          meta.get("rich_model_manifest_sha256") != rich_manifest_digest or
          [item.get("sha256") for item in meta.get("target_frequency_files", [])]
          != target_digests):
        return False
    for name, key in (("matching_results.tsv", "matching_sha256"),
                      ("candidate_pairs.tsv", "candidate_sha256")):
        path = directory / name
        if not path.is_file() or meta.get(key) != digest(path):
            return False
    return True


def invoke(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=pathlib.Path, required=True)
    parser.add_argument("--index-dir", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, default=MODELS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.shard_size) < 1 or args.shard_size > 50_000:
        parser.error("workers and batch size must be positive; shard size must be 1–50,000")
    source = args.test_dir / "test_source1.tsv"
    targets = [args.test_dir / f"test_source{i}.tsv" for i in (2, 3)]
    index_meta = args.index_dir / "index_meta.json"
    base_manifest = args.model_dir / "manifest.json"
    rich_manifest = args.model_dir / "rich_pair_v1v2/manifest.json"
    for required in (source, *targets, index_meta, base_manifest, rich_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.work_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Work and final output directories must differ")
    source_digest = digest(source)
    index_digest = digest(index_meta)
    base_digest = digest(base_manifest)
    rich_digest = digest(rich_manifest)
    target_digests = [digest(path) for path in targets]
    total = sum(1 for _ in rows(source))
    if total == 0:
        raise ValueError("Empty test Source-1 file")
    manifest = {
        "source1_sha256": source_digest,
        "target_sha256": target_digests,
        "index_meta_sha256": index_digest,
        "frozen_model_manifest_sha256": base_digest,
        "rich_model_manifest_sha256": rich_digest,
        "source_code_sha256": {path.name: digest(path)
                               for path in sorted(HERE.glob("*.py"))},
        "source1_rows": total,
        "shard_size": args.shard_size,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = args.work_dir / "rich_submission_run.json"
    if run_manifest.exists():
        if json.loads(run_manifest.read_text()) != manifest:
            raise ValueError("Existing work directory belongs to another input or shard plan")
    else:
        if any(args.work_dir.iterdir()):
            raise ValueError("Use an empty work directory for a new run")
        run_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    parts = []
    for start in range(0, total, args.shard_size):
        stop = min(start + args.shard_size, total)
        base = args.work_dir / f"baseline_{start:08d}_{stop:08d}"
        rich = args.work_dir / f"rich_{start:08d}_{stop:08d}"
        if complete_shard(rich, start, stop, source_digest, index_digest,
                          base_digest, rich_digest, target_digests):
            print("Verified completed rich shard", start, stop, flush=True)
            parts.append(rich)
            continue
        if rich.exists():
            raise RuntimeError(f"Incomplete rich shard at {rich}; inspect it before a new run")
        if not complete_shard(base, start, stop, source_digest, index_digest,
                              base_digest):
            if base.exists():
                raise RuntimeError(f"Incomplete baseline shard at {base}; inspect it before a new run")
            invoke(HERE / "infer.py", "run", "--source1", source,
                   "--index-dir", args.index_dir, "--output-dir", base,
                   "--model-dir", args.model_dir, "--workers", args.workers,
                   "--batch-size", args.batch_size,
                   "--start-row", start, "--stop-row", stop)
        invoke(HERE / "rich_infer.py", "--source1", source,
               "--candidate-tsv", base / "candidate_pairs.tsv",
               "--reference-baseline", base / "matching_results.tsv",
               "--index-dir", args.index_dir, "--target-tsv", *targets,
               "--model-dir", args.model_dir, "--out-dir", rich,
               "--batch-size", args.batch_size, "--start-row", start,
               "--stop-row", stop)
        if not complete_shard(rich, start, stop, source_digest, index_digest,
                              base_digest, rich_digest, target_digests):
            raise RuntimeError(f"Rich shard failed verification: {rich}")
        parts.append(rich)
    merge(source, parts, args.output_dir)
    print("Complete rich outputs written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
