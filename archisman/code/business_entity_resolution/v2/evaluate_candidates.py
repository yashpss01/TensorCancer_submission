"""Score frozen candidate shards against a sealed labeled S1 batch.

Run this only after retrieval parameters and every candidate shard are fixed.
The result is a blocking ceiling, not a measured final matcher F0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def read_tsv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None,
                       empty_string_is_null=False,
                       schema_overrides={"entity_id": pl.Utf8,
                                         "source1_entity_id": pl.Utf8,
                                         "matched_entity_ids": pl.Utf8})


def target_counts(root: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for source in ("train_source2", "train_source3"):
        manifest = json.loads((root / source / "manifest.json").read_text())
        for part in manifest["parts"]:
            for country in part["countries"]:
                name = country["country"]
                totals[name] = totals.get(name, 0) + country["len"]
    return totals


def score(selected: Path, truth: Path, candidate_root: Path,
          normalized_root: Path, expected_shards: int = 4) -> dict:
    entities = read_tsv(selected).select("entity_id", "country")
    if entities["entity_id"].n_unique() != entities.height:
        raise ValueError("Duplicate selected Source-1 ID")
    labels = read_tsv(truth).rename({"source1_entity_id": "entity_id"})
    labeled = entities.join(labels, on="entity_id", how="left", validate="1:1")
    if labeled["matched_entity_ids"].null_count():
        raise ValueError("Missing labels for selected Source-1 IDs")
    truth_pairs = (labeled.select("entity_id", "matched_entity_ids")
                   .filter(pl.col("matched_entity_ids") != "")
                   .with_columns(pl.col("matched_entity_ids").str.split(","))
                   .explode("matched_entity_ids", empty_as_null=True)
                   .rename({"matched_entity_ids": "q_id"}))
    shard_directories = sorted(path for path in candidate_root.iterdir() if path.is_dir())
    if len(shard_directories) != expected_shards:
        raise ValueError(f"Expected {expected_shards} shards, found {len(shard_directories)}")
    manifests = []
    for directory in shard_directories:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Incomplete retrieval shard: {directory}")
        manifests.append(json.loads(manifest_path.read_text()))
    counts_by_country = target_counts(normalized_root)
    for country, target_count in counts_by_country.items():
        queried = sum(manifest["queries"] for manifest in manifests if manifest["country"] == country)
        if queried != target_count:
            raise ValueError(f"{country}: retrieved {queried:,} of {target_count:,} target queries")
    paths = sorted(candidate_root.glob("*/candidates-*.parquet"))
    if not paths:
        raise FileNotFoundError("No candidate Parquet files found")
    candidates = pl.concat([pl.read_parquet(path, columns=["s1_id", "q_id"])
                            for path in paths])
    duplicate_count = candidates.height - candidates.unique(["s1_id", "q_id"]).height
    if duplicate_count:
        raise ValueError(f"{duplicate_count} duplicate candidate pairs across shards")
    if candidates.join(entities, left_on="s1_id", right_on="entity_id", how="anti").height:
        raise ValueError("Candidate shard contains a nonselected Source-1 ID")
    true_counts = truth_pairs.group_by("entity_id").len().rename({"len": "truth_count"})
    candidate_counts = candidates.group_by("s1_id").len().rename(
        {"s1_id": "entity_id", "len": "candidate_count"})
    hits = candidates.join(truth_pairs, left_on=["s1_id", "q_id"],
                           right_on=["entity_id", "q_id"], how="inner")
    hit_counts = hits.group_by("s1_id").len().rename(
        {"s1_id": "entity_id", "len": "hit_count"})
    per_entity = (entities.join(true_counts, on="entity_id", how="left")
                  .join(candidate_counts, on="entity_id", how="left")
                  .join(hit_counts, on="entity_id", how="left")
                  .with_columns(pl.col("truth_count").fill_null(0),
                                pl.col("candidate_count").fill_null(0),
                                pl.col("hit_count").fill_null(0)))

    def summarize(frame: pl.DataFrame, country: str | None) -> dict:
        truth_count = int(frame["truth_count"].sum())
        true_positive = int(frame["hit_count"].sum())
        false_negative = truth_count - true_positive
        candidate_count = int(frame["candidate_count"].sum())
        false_positive = candidate_count - true_positive
        countries = [country] if country else frame["country"].unique().to_list()
        space = sum(frame.filter(pl.col("country") == c).height
                    * counts_by_country[c] for c in countries)
        true_negative = space - true_positive - false_negative - false_positive
        if true_negative < 0:
            raise ValueError("Candidate count exceeds eligible comparison space")
        tc = frame["truth_count"].to_numpy()
        hc = frame["hit_count"].to_numpy()
        # Oracle predicts exactly the retrieved true links and nothing else.
        # Empty truth plus empty prediction scores 1 per challenge convention.
        oracle = np.where(tc == 0, 1.0, 1.25 * hc / np.maximum(1.25 * hc + 0.25 * (tc - hc), 1e-12))
        cc = frame["candidate_count"].to_numpy()
        return {
            "s1_count": frame.height,
            "target_count": sum(counts_by_country[c] for c in countries),
            "eligible_comparison_space": space,
            "tp": true_positive, "fn": false_negative,
            "fp": false_positive, "tn": true_negative,
            "blocking_recall": true_positive / truth_count if truth_count else None,
            "candidate_precision": true_positive / candidate_count if candidate_count else None,
            "reduction_ratio": 1 - candidate_count / space,
            "specificity": true_negative / (true_negative + false_positive),
            "candidate_f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
            "oracle_macro_f0_5": float(oracle.mean()),
            "candidates_per_s1_mean": float(cc.mean()),
            "candidates_per_s1_p95": float(np.percentile(cc, 95)),
            "candidates_per_s1_max": int(cc.max()),
        }

    result = {"scope": "sealed labeled 10k S1; full train S1 and S2/S3 competitors; country-partitioned search",
              "selected_s1": entities.height,
              "candidate_pairs": candidates.height,
              "target_counts": counts_by_country,
              "overall": summarize(per_entity, None),
              "countries": {country: summarize(per_entity.filter(pl.col("country") == country), country)
                            for country in sorted(per_entity["country"].unique().to_list())}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.selected, args.truth, args.candidate_root,
                   args.normalized_root, args.expected_shards)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
