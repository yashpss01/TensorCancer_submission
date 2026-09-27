"""Export label-blind rich pair evidence for exposed-cohort model screens.

The frozen probability is independently reproduced for every pair. This
research export neither reads truth nor changes the submission inference path.
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
from features import FEATURE_NAMES, features, represent  # noqa: E402
from inference_features import EXTRA_NAMES, GROUP_NAMES, extra_vector, group_matrix  # noqa: E402
from infer import digest, load_models, target_records  # noqa: E402
from core_frequency_features import FEATURE_NAMES as CORE_NAMES, load  # noqa: E402

RAPID_NAMES = (
    "unicode_core_ratio", "unicode_core_token_set", "unicode_core_wratio",
    "unicode_name_ratio", "unicode_address_ratio",
)
NAMES = tuple(FEATURE_NAMES + EXTRA_NAMES + GROUP_NAMES[1:]) + CORE_NAMES + RAPID_NAMES


def rapid_values(q, t):
    qcore = unidecode(q["core"])
    tcore = unidecode(t["core"])
    qname = unidecode(q["name"])
    tname = unidecode(t["name"])
    return (
        fuzz.ratio(qcore, tcore) / 100,
        fuzz.token_set_ratio(qcore, tcore) / 100,
        fuzz.WRatio(qcore, tcore) / 100,
        fuzz.ratio(qname, tname) / 100,
        fuzz.ratio(unidecode(q["addr"]), unidecode(t["addr"])) / 100
        if q["addr"] and t["addr"] else 0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--batch-groups", type=int, default=64)
    args = p.parse_args()
    if args.output.exists() or args.batch_groups < 1:
        raise ValueError("Use a new output file and positive batch size")
    started = time.monotonic()
    data = load(args.source1, args.pair_scores, args.core_cache)
    source = data["sources"]
    frame = data["frame"]
    group = data["group"]
    core = data["feature_matrices"]["context_plus_core_frequency"]
    cached = data["frozen_probability"]
    mids = frame["target_id"].to_list()
    widths = np.bincount(group, minlength=len(source))
    offsets = np.r_[0, np.cumsum(widths)]
    models = load_models(ROOT / "code/business_entity_resolution/models")
    first, group_model, augmented, (nw, aw, default), (cn, ca, cdefault) = models
    output = np.lib.format.open_memmap(args.output, mode="w+", dtype="float32",
                                       shape=(len(frame), len(NAMES)))
    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    for start in range(0, len(source), args.batch_groups):
        stop = min(start + args.batch_groups, len(source))
        lo, hi = offsets[start], offsets[stop]
        records = target_records(db, mids[lo:hi])
        for i in range(start, stop):
            a, b = offsets[i:i+2]
            if a == b:
                continue
            qrow = source[i]
            qr = represent(qrow["business_name"], qrow["business_address"], qrow["country"])
            tr = [represent(*records[mid]) for mid in mids[a:b]]
            x = np.asarray([features(qr, t, nw, aw, default) for t in tr], dtype="float32")
            e = np.asarray([extra_vector(qr, t, cn, ca, cdefault) for t in tr], dtype="float32")
            first_prob = first.predict_proba(x)[:, 1].astype("float32")
            g = group_matrix(qrow, tr, first_prob, aw, default)
            pair = np.concatenate((x, e), axis=1)
            recalculated = (.4 * group_model.predict_proba(np.concatenate((pair, g[:, 1:]), axis=1))[:, 1]
                            + .6 * augmented.predict_proba(pair)[:, 1]).astype("float32")
            if not np.allclose(recalculated, cached[a:b], rtol=0, atol=2e-6):
                raise ValueError(f"Frozen probability mismatch at group {i}")
            rapid = np.asarray([rapid_values(qr, t) for t in tr], dtype="float32")
            output[a:b] = np.column_stack((x, e, g[:, 1:], core[a:b], rapid))
        if stop % 1000 < args.batch_groups or stop == len(source):
            output.flush()
            print(f"exported {stop}/{len(source)} groups in {time.monotonic()-started:.1f}s", flush=True)
    db.close()
    output.flush()
    meta = {
        "scope": "label-blind research feature export; no truth read",
        "source1_rows": len(source), "candidate_pairs": len(frame),
        "feature_names": NAMES, "source1_sha256": digest(args.source1),
        "pair_scores_sha256": digest(args.pair_scores),
        "core_cache_manifest_sha256": digest(args.core_cache / "manifest.json"),
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "model_manifest_sha256": digest(ROOT / "code/business_entity_resolution/models/manifest.json"),
        "seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_names"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
