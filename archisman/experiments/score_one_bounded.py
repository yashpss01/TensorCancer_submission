"""Score one frozen prediction and its exact candidate TSV on a sealed 10k pool."""

import argparse
import csv
import hashlib
import json
import pathlib

from score_bounded_fresh import ids, summarize


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-manifest", type=pathlib.Path,
                        help="Origin manifest; defaults to frozen sparse export")
    parser.add_argument("--matching-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.cohort / "seal.json").read_text())
    pool = json.loads((args.cohort / "bounded_index/index_meta.json").read_text())
    if sha(args.cohort / "source1.tsv") != seal["source1_sha256"]:
        raise ValueError("Source-1 changed after sealing")
    if sha(args.cohort / "truth.tsv") != seal["truth_sha256"]:
        raise ValueError("Truth changed after sealing")
    origin = args.candidate_manifest or args.cohort / "forward_sparse_rank64.json"
    export = json.loads(origin.read_text())
    if origin.name == "forward_sparse_rank64.json" and export.get("export_rank") != 64:
        raise ValueError("Sparse export is not the frozen top-64 rule")
    if export.get("candidate_sha256") != sha(args.candidate_tsv):
        raise ValueError("Candidate TSV differs from its origin manifest")
    inference = json.loads((args.matching_tsv.parent / "inference_meta.json").read_text())
    if inference.get("candidate_sha256") != export["candidate_sha256"]:
        raise ValueError("Predictions were scored from other candidates")
    if inference.get("matching_sha256") != sha(args.matching_tsv):
        raise ValueError("Matching TSV differs from inference metadata")
    if inference.get("rich_model_manifest_sha256") != seal["rich_manifest_sha256"]:
        raise ValueError("Rich model differs from sealed manifest")
    by_country = {}
    observations = []
    n = 0
    for source, truth, candidate, prediction in zip(
            rows(args.cohort / "source1.tsv"), rows(args.cohort / "truth.tsv"),
            rows(args.candidate_tsv), rows(args.matching_tsv), strict=True):
        sid = source["entity_id"]
        if sid != truth["source1_entity_id"] or sid != candidate["source1_entity_id"] or sid != prediction["source1_entity_id"]:
            raise ValueError("Source-1 IDs/order differ")
        actual = set(ids(truth["matched_entity_ids"]))
        mids = ids(candidate["candidate_entity_ids"])
        selected = ids(prediction["matched_entity_ids"])
        candidates = set(mids)
        proposed = set(selected)
        if len(candidates) != len(mids) or len(proposed) != len(selected) or not proposed <= candidates:
            raise ValueError("Duplicate or out-of-candidate prediction")
        tp = len(proposed & actual)
        fp = len(proposed - actual)
        fn = len(actual - proposed)
        candidate_tp = len(candidates & actual)
        item = {
            "country": source["country"], "tp": tp, "fp": fp, "fn": fn,
            "f05": ((1.0 if not proposed else 0.0) if not actual else
                    1.25 * tp / (1.25 * tp + 0.25 * fn + fp)),
            "candidate_tp": candidate_tp,
            "candidate_fn": len(actual - candidates),
            "candidate_oracle_f05": (1.0 if not actual else
                                     1.25 * candidate_tp /
                                     (candidate_tp + 0.25 * len(actual))),
            "candidate_count": len(candidates),
            "singleton": int(not actual),
        }
        observations.append(item)
        by_country.setdefault(item["country"], []).append(item)
        n += 1
    if n != seal["source1_rows"]:
        raise ValueError("Incomplete S1 prediction coverage")
    result = {
        "scope": "sealed reduced target pool; not full test, France, or Portal",
        "candidate_origin_manifest": str(origin),
        "cohort_seal_sha256": sha(args.cohort / "seal.json"),
        "candidate_sha256": sha(args.candidate_tsv),
        "matching_sha256": sha(args.matching_tsv),
        "target_count": pool["target_count"],
        "overall": summarize(observations, pool["target_count"]),
        "countries": {country: summarize(items, pool["target_count"])
                      for country, items in by_country.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"overall": result["overall"], "countries": result["countries"]}, indent=2))


if __name__ == "__main__":
    main()
