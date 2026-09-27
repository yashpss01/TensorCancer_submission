"""Label-blind, bounded-memory normalization for batched candidate retrieval.

Each input TSV is read in batches. The output Parquet shards retain original
entity IDs and record order; this module never opens ground-truth files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from normalize import norm_address, norm_name


def normalize_batch(frame: pl.DataFrame) -> pl.DataFrame:
    names = [norm_name(value) for value in frame["business_name"].fill_null("").to_list()]
    addresses = [norm_address(value) for value in frame["business_address"].fill_null("").to_list()]
    name_columns = {key: [item[key] for item in names] for key in names[0]} if names else {}
    address_columns = {key: [item[key] for item in addresses] for key in addresses[0]} if addresses else {}
    return frame.with_columns(
        [pl.Series(key, value) for key, value in (name_columns | address_columns).items()]
    )


def prepare_one(source: Path, destination: Path, rows_per_batch: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    contract = {"source": str(source), "bytes": source.stat().st_size,
                "rows_per_batch": rows_per_batch, "normalizer": "v2-rule-only"}
    input_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("input_hash") != input_hash:
            raise ValueError(f"Completed normalization has different inputs: {destination}")
        return manifest
    for partial in destination.glob("part-*.parquet"):
        partial.unlink()
    reader = pl.scan_csv(
        source,
        separator="\t",
        quote_char=None,
        empty_string_is_null=False,
        schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                          "business_address": pl.Utf8, "country": pl.Utf8},
    )
    total = 0
    parts = []
    for batch in reader.collect_batches(chunk_size=rows_per_batch):
        if batch.is_empty():
            continue
        clean = normalize_batch(batch)
        path = destination / f"part-{len(parts):05d}.parquet"
        temporary = path.with_suffix(".parquet.partial")
        clean.write_parquet(temporary, compression="zstd")
        temporary.replace(path)
        parts.append({"file": path.name, "rows": clean.height,
                      "countries": clean.group_by("country").len().to_dicts()})
        total += clean.height
        print(f"{source.name}: {total:,} rows", flush=True)
    manifest = {"source": str(source), "rows": total, "parts": parts,
                "input_hash": input_hash,
                "label_blind": True, "normalizer": "v2-rule-only"}
    temporary_manifest = destination / "manifest.json.partial"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rows-per-batch", type=int, default=100_000)
    args = parser.parse_args()
    if args.rows_per_batch < 1:
        parser.error("--rows-per-batch must be positive")
    prepare_one(args.source, args.out_dir, args.rows_per_batch)


if __name__ == "__main__":
    main()
