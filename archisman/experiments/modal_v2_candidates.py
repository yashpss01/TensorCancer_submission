"""Bounded Modal CPU run for label-blind full-competitor candidate validation.

Prerequisites (after approving the cloud budget):
  modal volume create tensorcancer-v2-validation
  modal volume put tensorcancer-v2-validation train-data.tar.gz /input/train-data.tar.gz
  modal volume put tensorcancer-v2-validation selected-s1.tsv /input/selected-s1.tsv
  modal run experiments/modal_v2_candidates.py

The tar must contain only train_source1/2/3.tsv at its root. Do not upload
ground-truth data. Each retrieval worker searches the full country S1 index.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import modal

APP_NAME = "tensorcancer-v2-candidate-validation"
VOLUME_NAME = "tensorcancer-v2-validation"
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "code" / "business_entity_resolution" / "v2"

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6", "scipy==1.17.0", "scikit-learn==1.8.0",
                      "polars==1.44.2", "sparse-dot-topn==1.2.0", "Unidecode==1.4.0")
         .add_local_dir(str(V2), remote_path="/app/v2"))
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)
MOUNT = "/data"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


@app.function(image=image, volumes={MOUNT: volume}, cpu=2, memory=8192, timeout=1800)
def unpack_inputs() -> dict:
    archive = Path(MOUNT) / "input" / "train-data.tar.gz"
    raw = Path(MOUNT) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    expected = {f"train_source{i}.tsv" for i in (1, 2, 3)}
    with tarfile.open(archive, "r:gz") as tar:
        found = {member.name for member in tar.getmembers() if member.name in expected}
        if found != expected:
            raise ValueError(f"Input tar is missing source files: {expected - found}")
        for member in tar.getmembers():
            if member.name not in expected:
                continue
            target = raw / Path(member.name).name
            if target.exists():
                continue
            with tar.extractfile(member) as stream, target.open("wb") as output:
                while block := stream.read(4 * 1024 * 1024):
                    output.write(block)
    volume.commit()
    return {"raw_files": {p.name: p.stat().st_size for p in raw.glob("*.tsv")}}


@app.function(image=image, volumes={MOUNT: volume}, cpu=2, memory=8192, timeout=5400)
def normalize_source(source_number: int) -> dict:
    volume.reload()
    source = f"train_source{source_number}"
    out = Path(MOUNT) / "norm" / source
    run("python", "/app/v2/prepare.py", "--source",
        f"{MOUNT}/raw/{source}.tsv", "--out-dir", str(out),
        "--rows-per-batch", "100000")
    volume.commit()
    return json.loads((out / "manifest.json").read_text())


@app.function(image=image, volumes={MOUNT: volume}, cpu=4, memory=24576, timeout=7200)
def retrieve_shard(country: str, shard_index: int, shard_count: int) -> dict:
    volume.reload()
    queries = sorted((Path(MOUNT) / "norm" / "train_source2").glob("part-*.parquet"))
    queries += sorted((Path(MOUNT) / "norm" / "train_source3").glob("part-*.parquet"))
    queries = [file for number, file in enumerate(queries) if number % shard_count == shard_index]
    out = Path(MOUNT) / "candidates" / f"{country}-{shard_index}-of-{shard_count}"
    query_list = Path("/tmp") / f"query-files-{country}-{shard_index}.json"
    query_list.write_text(json.dumps([str(path) for path in queries]))
    # Keep Modal's heartbeat thread in the parent interpreter while the CPU
    # intensive sparse matrix work runs in a subprocess.
    run("python", "/app/v2/retrieve.py", "--index-dir",
        f"{MOUNT}/norm/train_source1", "--query-files-json", str(query_list),
        "--out-dir", str(out), "--country", country,
        "--selected-ids", f"{MOUNT}/input/selected-s1.tsv",
        "--rows-per-chunk", "25000", "--threads", "4",
        "--keep", "4", "--max-rank", "40")
    result = json.loads((out / "manifest.json").read_text())
    volume.commit()
    return result


@app.function(image=image, volumes={MOUNT: volume}, cpu=2, memory=8192, timeout=1800)
def export_selected_targets() -> dict:
    """Export raw records for retrieved targets; labels remain on the local host."""
    volume.reload()
    import polars as pl

    output = Path(MOUNT) / "selected_target_records.parquet"
    manifest_path = Path(MOUNT) / "selected_target_records.json"
    if output.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text())
    candidate_paths = sorted((Path(MOUNT) / "candidates").glob("*/candidates-*.parquet"))
    selected_targets = set(pl.concat([
        pl.read_parquet(path, columns=["q_id"]) for path in candidate_paths
    ])["q_id"].unique().to_list())
    selected_series = pl.Series("selected", sorted(selected_targets))
    records = []
    for source in (2, 3):
        for path in sorted((Path(MOUNT) / "norm" / f"train_source{source}").glob("part-*.parquet")):
            matched = pl.read_parquet(
                path, columns=["entity_id", "business_name", "business_address", "country"]
            ).filter(pl.col("entity_id").is_in(selected_series))
            if not matched.is_empty():
                records.append(matched)
    combined = pl.concat(records)
    if combined["entity_id"].n_unique() != len(selected_targets):
        raise ValueError("Missing or duplicate selected target records")
    temporary = output.with_suffix(".parquet.partial")
    combined.write_parquet(temporary, compression="zstd")
    temporary.replace(output)
    manifest = {"target_records": combined.height, "candidate_shards": len(candidate_paths),
                "label_blind": True}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print("Extracting only unlabeled source records")
    print(unpack_inputs.remote())
    normalizers = [normalize_source.spawn(i) for i in (1, 2, 3)]
    for i, job in zip((1, 2, 3), normalizers):
        result = job.get()
        print(f"Normalized source {i}: {result['rows']:,} rows", flush=True)
    workers = [(country, shard, retrieve_shard.spawn(country, shard, 2))
               for country in ("India", "US") for shard in (0, 1)]
    for country, shard, job in workers:
        result = job.get()
        print(f"{country} shard {shard}: {result['queries']:,} queries; "
              f"{result['candidates']:,} selected candidates; "
              f"{result['seconds']:.1f}s", flush=True)
    print("Selected target records:", export_selected_targets.remote())
    print("All candidate shards complete. Ground truth remains local and unopened.")
