"""Label-blind candidate-to-anchor graph features for exposed-cohort screens.

Anchors are *frozen baseline* matches (probability >= .74). The candidate's
relationship to anchors from the other target source can provide evidence
when its own address is missing. This export does not read ground truth.
"""

import argparse
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
from rapidfuzz import fuzz
from unidecode import unidecode

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from infer import digest, target_records  # noqa: E402
from normalize import address  # noqa: E402
from core_frequency_features import load  # noqa: E402

NAMES = (
    "cross_anchor_count", "cross_anchor_max_probability",
    "cross_anchor_core_ratio", "cross_anchor_core_token_set",
    "cross_anchor_core_wratio", "cross_anchor_exact_core",
    "cross_anchor_address_ratio", "cross_anchor_exact_address",
    "cross_anchor_number_overlap", "cross_anchor_name_probability_product",
    "same_source_anchor_core_ratio", "candidate_is_frozen_anchor",
)


def build_metadata(ids, db):
    targets = target_records(db, ids)
    result = {}
    for mid, (name, addr, _) in targets.items():
        normalized_addr = " ".join(address(addr))
        numbers = {v for v in normalized_addr.split() if v.isdigit()}
        result[mid] = (unidecode(name.casefold()), unidecode(normalized_addr), numbers)
    return result


def ratio(a, b):
    return fuzz.ratio(a, b) / 100 if a and b else 0.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError("Use a new output file")
    started = time.monotonic()
    data = load(args.source1, args.pair_scores, args.core_cache)
    frame = data["frame"]
    group = data["group"]
    mids = frame["target_id"].to_list()
    cores = [unidecode(value.casefold()) for value in frame["target_core"].to_list()]
    probs = data["frozen_probability"]
    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    targets = build_metadata(set(mids), db)
    db.close()
    if len(targets) != len(set(mids)):
        raise ValueError("Missing target record")
    output = np.lib.format.open_memmap(args.output, mode="w+", dtype="float32",
                                       shape=(len(mids), len(NAMES)))
    counts = np.bincount(group, minlength=len(data["source_ids"]))
    offsets = np.r_[0, np.cumsum(counts)]
    for i in range(len(data["source_ids"])):
        lo, hi = offsets[i:i+2]
        anchors = [j for j in range(lo, hi) if probs[j] >= .74]
        for j in range(lo, hi):
            mid = mids[j]
            cross = [k for k in anchors if mids[k][:2] != mid[:2]]
            same = [k for k in anchors if mids[k][:2] == mid[:2] and k != j]
            name, addr, nums = targets[mid]
            core = cores[j]
            values = []
            for k in cross:
                other_name, other_addr, other_nums = targets[mids[k]]
                other_core = cores[k]
                cr = ratio(core, other_core)
                values.append((
                    cr,
                    fuzz.token_set_ratio(core, other_core) / 100 if core and other_core else 0.,
                    fuzz.WRatio(core, other_core) / 100 if core and other_core else 0.,
                    float(bool(core) and core == other_core),
                    ratio(addr, other_addr),
                    float(bool(addr) and addr == other_addr),
                    float(bool(nums and other_nums) and bool(nums & other_nums)),
                    cr * float(probs[k]),
                ))
            maxima = [max(row[c] for row in values) if values else 0. for c in range(8)]
            same_max = max((ratio(core, cores[k]) for k in same), default=0.)
            output[j] = [len(cross), max((float(probs[k]) for k in cross), default=0.),
                         *maxima, same_max, float(j in anchors)]
        if (i + 1) % 1000 == 0:
            output.flush()
            print(f"anchor features {i+1}/{len(data['source_ids'])} in {time.monotonic()-started:.1f}s", flush=True)
    output.flush()
    meta = {
        "scope": "label-blind frozen-baseline cross-source anchor features; no truth read",
        "source1_rows": len(data["source_ids"]), "candidate_pairs": len(mids),
        "feature_names": NAMES,
        "source1_sha256": digest(args.source1),
        "pair_scores_sha256": digest(args.pair_scores),
        "core_cache_manifest_sha256": digest(args.core_cache / "manifest.json"),
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "seconds": time.monotonic()-started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_names"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
