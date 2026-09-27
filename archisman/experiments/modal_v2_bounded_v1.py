"""Bounded, label-blind v2 retrieval comparison on fresh v1's fixed target pool.

Reuses the already normalized full training S1 universe in the Modal Volume.
Only the 442,904 targets of fresh v1 are queried. The selected 10k S1 IDs are
filtered *after* global S1 ranking. Ground truth is not uploaded or mounted.
"""

import csv
import json
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "code/business_entity_resolution/v2"
MOUNT = "/data"
volume = modal.Volume.from_name("tensorcancer-v2-validation")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6", "scipy==1.17.0", "scikit-learn==1.8.0",
                      "polars==1.44.2", "sparse-dot-topn==1.2.0", "Unidecode==1.4.0")
         .add_local_dir(str(V2), remote_path="/app/v2"))
app = modal.App("tensorcancer-v2-bounded-fresh-v1")


@app.function(image=image, volumes={MOUNT: volume}, cpu=2, memory=8192, timeout=1800)
def prepare_bounded_targets() -> dict:
    volume.reload()
    root = Path(MOUNT) / "bounded_v1"
    norm = root / "norm_targets"
    manifest = norm / "manifest.json"
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result["rows"] != 442_904:
            raise ValueError("Existing bounded target normalization has wrong row count")
        return result
    archive = Path(MOUNT) / "input/bounded-fresh-v1.tar.gz"
    db_path = Path("/tmp/bounded_v1_index.sqlite")
    with tarfile.open(archive, "r:gz") as tar:
        item = tar.getmember("bounded_index/index.sqlite")
        with tar.extractfile(item) as source, db_path.open("wb") as sink:
            while block := source.read(4 * 1024 * 1024):
                sink.write(block)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "target_raw.tsv"
    count = 0
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    with raw.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        for eid, name, address, country in db.execute(
                "SELECT id,name,address,country FROM records ORDER BY rowid"):
            writer.writerow([eid, name, address, country])
            count += 1
    db.close()
    if count != 442_904:
        raise ValueError(f"Expected 442904 target records, found {count}")
    subprocess.run(["python", "/app/v2/prepare.py", "--source", str(raw),
                    "--out-dir", str(norm), "--rows-per-batch", "100000"], check=True)
    result = json.loads(manifest.read_text())
    if result["rows"] != 442_904:
        raise ValueError("Normalized target count mismatch")
    volume.commit()
    return result


@app.function(image=image, volumes={MOUNT: volume}, cpu=4, memory=24576, timeout=3600)
def retrieve_country(country: str, variant: str, keep: int, max_rank: int) -> dict:
    if country not in {"India", "US"}:
        raise ValueError(country)
    if (variant, keep, max_rank) not in {("default", 4, 40), ("wide", 16, 80)}:
        raise ValueError("Only the predeclared default/wide configurations are supported")
    volume.reload()
    root = Path(MOUNT) / "bounded_v1"
    norm = root / "norm_targets"
    if json.loads((norm / "manifest.json").read_text())["rows"] != 442_904:
        raise ValueError("Incomplete target normalization")
    index_manifest = json.loads((Path(MOUNT) / "norm/train_source1/manifest.json").read_text())
    if index_manifest["rows"] != 2_206_821:
        raise ValueError("Full S1 index is incomplete")
    selected = Path(MOUNT) / "input/fresh-v1-selected-ids.tsv"
    out = root / ("candidates" if variant == "default" else "candidates_wide") / country
    command = ["python", "/app/v2/retrieve.py", "--index-dir",
               str(Path(MOUNT) / "norm/train_source1"),
               "--query-dir", str(norm), "--out-dir", str(out),
               "--country", country, "--selected-ids", str(selected),
               "--rows-per-chunk", "25000", "--threads", "4",
               "--keep", str(keep), "--max-rank", str(max_rank)]
    subprocess.run(command, check=True)
    result = json.loads((out / "manifest.json").read_text())
    volume.commit()
    return result


@app.local_entrypoint()
def main(variant: str = "default", keep: int = 4, max_rank: int = 40) -> None:
    if (variant, keep, max_rank) not in {("default", 4, 40), ("wide", 16, 80)}:
        raise ValueError("Only default/4/40 and wide/16/80 are supported")
    prepared = prepare_bounded_targets.remote()
    print(json.dumps({"prepared_targets": prepared["rows"]}), flush=True)
    jobs = {country: retrieve_country.spawn(country, variant, keep, max_rank)
            for country in ("India", "US")}
    for country, job in jobs.items():
        result = job.get()
        print(json.dumps({"country": country, **result}, indent=2), flush=True)
