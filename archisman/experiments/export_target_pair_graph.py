"""Export label-blind opposite-source candidate/anchor text features.

Anchors are selected only from frozen baseline probabilities. This exporter
does not read truth labels. It writes one target-target row per candidate and
up to three accepted anchors from the other target source.
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
from features import represent  # noqa: E402
from infer import digest, target_records  # noqa: E402
from core_frequency_features import load  # noqa: E402

FEATURE_NAMES = (
    "core_ratio", "core_token_set", "core_wratio", "name_ratio",
    "name_token_set", "address_ratio", "address_token_set",
    "exact_core", "exact_name", "exact_nonempty_address",
    "address_number_shared", "address_number_conflict",
    "candidate_address_empty", "anchor_address_empty",
    "candidate_is_source3", "candidate_frozen_probability",
    "anchor_frozen_probability",
)


def text_features(candidate, anchor, candidate_source3, candidate_p, anchor_p):
    c, a = candidate, anchor
    cc, ac = unidecode(c["core"]), unidecode(a["core"])
    cn, an = unidecode(c["name"]), unidecode(a["name"])
    ca, aa = unidecode(c["addr"]), unidecode(a["addr"])
    shared = len(c["nums"] & a["nums"])
    return (
        fuzz.ratio(cc, ac) / 100, fuzz.token_set_ratio(cc, ac) / 100,
        fuzz.WRatio(cc, ac) / 100, fuzz.ratio(cn, an) / 100,
        fuzz.token_set_ratio(cn, an) / 100,
        fuzz.ratio(ca, aa) / 100 if ca and aa else 0,
        fuzz.token_set_ratio(ca, aa) / 100 if ca and aa else 0,
        float(bool(cc) and cc == ac), float(bool(cn) and cn == an),
        float(bool(ca) and ca == aa), float(shared),
        float(bool(c["nums"] and a["nums"]) and shared == 0),
        float(not ca), float(not aa), float(candidate_source3),
        float(candidate_p), float(anchor_p),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    started = time.monotonic()
    data = load(args.source1, args.pair_scores, args.core_cache)
    mids = data["frame"]["target_id"].to_list()
    group = data["group"]
    probability = data["frozen_probability"]
    source3 = [mid.startswith("S3-") for mid in mids]
    if not all(is_s3 or mid.startswith("S2-") for mid, is_s3 in zip(mids, source3)):
        raise ValueError("Unknown target source prefix")
    db = sqlite3.connect(f"file:{args.index_dir/'index.sqlite'}?mode=ro", uri=True)
    records = target_records(db, set(mids))
    db.close()
    reps = {mid: represent(*record) for mid, record in records.items()}
    counts = np.bincount(group, minlength=len(data["source_ids"]))
    offsets = np.r_[0, np.cumsum(counts)]
    candidate_rows, anchor_rows, feature_rows = [], [], []
    for i in range(len(data["source_ids"])):
        lo, hi = offsets[i:i+2]
        accepted = [j for j in range(lo, hi) if probability[j] >= .74]
        accepted.sort(key=lambda j: -probability[j])
        by_source = {
            False: [j for j in accepted if not source3[j]][:3],
            True: [j for j in accepted if source3[j]][:3],
        }
        for j in range(lo, hi):
            for k in by_source[not source3[j]]:
                candidate_rows.append(j)
                anchor_rows.append(k)
                feature_rows.append(text_features(
                    reps[mids[j]], reps[mids[k]], source3[j],
                    probability[j], probability[k]))
        if (i+1) % 1000 == 0:
            print("target-pair features", i+1, "S1 in",
                  round(time.monotonic()-started, 1), "seconds", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "candidate_row.npy", np.asarray(candidate_rows, dtype=np.int32))
    np.save(args.out_dir / "anchor_row.npy", np.asarray(anchor_rows, dtype=np.int32))
    np.save(args.out_dir / "features.npy", np.asarray(feature_rows, dtype=np.float32))
    result = {
        "scope": "label-blind target-target graph features from frozen accepted anchors",
        "source1_rows": len(data["source_ids"]),
        "candidate_pairs": len(mids), "anchor_candidate_pairs": len(candidate_rows),
        "feature_names": FEATURE_NAMES,
        "source1_sha256": digest(args.source1),
        "pair_scores_sha256": digest(args.pair_scores),
        "core_cache_manifest_sha256": digest(args.core_cache / "manifest.json"),
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "seconds": time.monotonic()-started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "feature_names"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
