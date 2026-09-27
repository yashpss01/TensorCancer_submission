"""Full fresh-10k cloud CPU check on a reduced target pool.

Upload only the unlabeled tar containing source1.tsv, bounded_index/, and the
pretrained experimental residual model. No ground truth is mounted. Every one
of the 10,000 fresh S1 rows is evaluated on the same 442,904-target pool.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import tarfile
import time
from pathlib import Path

import modal


APP_NAME = "tensorcancer-bounded-matcher-fresh10k"
VOLUME_NAME = "tensorcancer-v2-validation"
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code/business_entity_resolution/src"
MODELS = ROOT / "code/business_entity_resolution/models"
EXPERIMENTS = ROOT / "experiments"
MOUNT = "/data"

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6", "scipy==1.17.0", "scikit-learn==1.8.0",
                      "xgboost==3.2.0", "joblib==1.5.3", "Unidecode==1.4.0")
         .add_local_dir(str(SRC), remote_path="/workspace/code/business_entity_resolution/src")
         .add_local_dir(str(MODELS), remote_path="/workspace/code/business_entity_resolution/models")
         .add_local_dir(str(EXPERIMENTS), remote_path="/workspace/experiments"))
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App(APP_NAME)


@app.function(image=image, volumes={MOUNT: volume}, cpu=4, memory=12288, timeout=3600)
def run_fresh_10k() -> dict:
    volume.reload()
    archive = Path(MOUNT) / "input" / "bounded-fresh-v1.tar.gz"
    local = Path("/tmp/bounded-fresh10k")
    local.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    allowed = {"source1.tsv", "bounded_index/index.sqlite",
               "bounded_index/extra.sqlite", "bounded_index/index_meta.json",
               "residual_rerank_dev.json"}
    with tarfile.open(archive, "r:gz") as tar:
        actual = {m.name for m in tar.getmembers() if m.isfile()}
        if actual != allowed:
            raise ValueError(f"Unexpected or missing archive files: {actual ^ allowed}")
        for member in tar.getmembers():
            if not member.isfile():
                continue
            target = local / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as sink:
                while block := source.read(4 * 1024 * 1024):
                    sink.write(block)
    setup_seconds = time.monotonic() - start
    out = Path(MOUNT) / "bounded_fresh_v1" / "baseline_10k"
    changed = Path(MOUNT) / "bounded_fresh_v1" / "rerank_10k"
    if (out.exists() and any(out.iterdir())) or (changed.exists() and any(changed.iterdir())):
        raise FileExistsError("Fresh 10k output already exists")
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "python", "/workspace/code/business_entity_resolution/src/infer.py", "run",
        "--source1", str(local / "source1.tsv"),
        "--index-dir", str(local / "bounded_index"),
        "--output-dir", str(out),
        "--model-dir", "/workspace/code/business_entity_resolution/models",
        "--workers", "4", "--batch-size", "200",
    ]
    started_inference = time.monotonic()
    subprocess.run(command, check=True)
    inference_seconds = time.monotonic() - started_inference
    meta = json.loads((out / "inference_meta.json").read_text())
    if meta["source1_rows"] != 10_000:
        raise ValueError("Cloud baseline did not cover all 10,000 S1 records")
    started_rerank = time.monotonic()
    subprocess.run([
        "python", "/workspace/experiments/rescore_bounded_candidates.py",
        "--source1", str(local / "source1.tsv"),
        "--candidate-tsv", str(out / "candidate_pairs.tsv"),
        "--baseline-tsv", str(out / "matching_results.tsv"),
        "--index-dir", str(local / "bounded_index"),
        "--model-dir", "/workspace/code/business_entity_resolution/models",
        "--residual-model", str(local / "residual_rerank_dev.json"),
        "--out-dir", str(changed), "--batch-size", "200",
    ], check=True)
    rerank_seconds = time.monotonic() - started_rerank
    rerank_meta = json.loads((changed / "manifest.json").read_text())
    if rerank_meta["rows"] != 10_000:
        raise ValueError("Cloud rerank did not cover all 10,000 S1 records")
    volume.commit()
    return {
        "setup_seconds": setup_seconds,
        "inference_seconds": inference_seconds,
        "rerank_seconds": rerank_seconds,
        "rows": meta["source1_rows"],
        "candidate_pairs": meta["candidate_pairs"],
        "baseline_matching_sha256": meta["matching_sha256"],
        "rerank_matching_sha256": hashlib.sha256((changed / "matching_results.tsv").read_bytes()).hexdigest(),
        "candidate_sha256": meta["candidate_sha256"],
        "scope": "all 10000 fresh S1 rows, same 442904-target reduced pool, frozen baseline and experimental blend",
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_fresh_10k.remote(), indent=2), flush=True)
