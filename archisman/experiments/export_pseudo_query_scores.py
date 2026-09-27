"""Score uncertain candidates against frozen accepted cross-source targets.

This is an experimental, label-blind target-to-target self-consistency signal.
Only frozen baseline matches can act as anchors. The augmented pair model is
unchanged; no truth or test labels are read.
"""

import argparse
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import features, represent  # noqa: E402
from inference_features import extra_vector  # noqa: E402
from infer import digest, load_models, target_records  # noqa: E402
from core_frequency_features import load  # noqa: E402

NAMES = (
    "cross_anchor_augmented_probability",
    "cross_anchor_first_probability",
    "cross_anchor_high_augmented_count",
    "cross_anchor_probability_product",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--pair-scores", type=pathlib.Path, required=True)
    p.add_argument("--core-cache", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--min-frozen-probability", type=float, default=.01)
    args = p.parse_args()
    if args.output.exists() or not 0 <= args.min_frozen_probability < 1:
        raise ValueError("Use a new output and valid cutoff")
    started = time.monotonic()
    data = load(args.source1, args.pair_scores, args.core_cache)
    mids = data["frame"]["target_id"].to_list()
    group = data["group"]
    p0 = data["frozen_probability"]
    widths = np.bincount(group, minlength=len(data["source_ids"]))
    offsets = np.r_[0, np.cumsum(widths)]
    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    used = {mids[j] for j in range(len(mids)) if p0[j] >= args.min_frozen_probability}
    used.update(mids[j] for j in range(len(mids)) if p0[j] >= .74)
    records = target_records(db, used)
    db.close()
    models = load_models(ROOT / "code/business_entity_resolution/models")
    first, _, augmented, (nw, aw, default), (cn, ca, cdefault) = models
    output = np.zeros((len(mids), len(NAMES)), dtype=np.float32)
    x, e, mapping = [], [], []
    pair_count = 0

    def flush():
        nonlocal pair_count
        if not x:
            return
        X = np.asarray(x, dtype="float32")
        E = np.asarray(e, dtype="float32")
        first_prob = first.predict_proba(X)[:, 1]
        aug_prob = augmented.predict_proba(np.column_stack((X, E)))[:, 1]
        for (j, anchor_prob), pfirst, paug in zip(mapping, first_prob, aug_prob):
            output[j, 0] = max(output[j, 0], paug)
            output[j, 1] = max(output[j, 1], pfirst)
            output[j, 2] += float(paug >= .74)
            output[j, 3] = max(output[j, 3], paug * anchor_prob)
        pair_count += len(mapping)
        x.clear(); e.clear(); mapping.clear()

    for i in range(len(data["source_ids"])):
        lo, hi = offsets[i:i+2]
        anchors = sorted((j for j in range(lo, hi) if p0[j] >= .74),
                         key=lambda j: -p0[j])
        for j in range(lo, hi):
            if p0[j] < args.min_frozen_probability:
                continue
            candidate = mids[j]
            tr = represent(*records[candidate])
            opposite = [k for k in anchors if mids[k][:2] != candidate[:2]][:3]
            for k in opposite:
                qr = represent(*records[mids[k]])
                x.append(features(qr, tr, nw, aw, default))
                e.append(extra_vector(qr, tr, cn, ca, cdefault))
                mapping.append((j, float(p0[k])))
        if len(mapping) >= 20_000:
            flush()
        if (i+1) % 1000 == 0:
            print(f"pseudo query {i+1}/{len(data['source_ids'])} in {time.monotonic()-started:.1f}s", flush=True)
    flush()
    np.save(args.output, output)
    meta = {
        "scope": "label-blind frozen-model pseudo-query scores; no truth read",
        "source1_rows": len(data["source_ids"]), "candidate_pairs": len(mids),
        "scored_anchor_candidate_pairs": pair_count,
        "min_frozen_probability": args.min_frozen_probability,
        "feature_names": NAMES,
        "source1_sha256": digest(args.source1),
        "pair_scores_sha256": digest(args.pair_scores),
        "core_cache_manifest_sha256": digest(args.core_cache / "manifest.json"),
        "index_meta_sha256": digest(args.index_dir / "index_meta.json"),
        "model_manifest_sha256": digest(ROOT / "code/business_entity_resolution/models/manifest.json"),
        "seconds": time.monotonic()-started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_names"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
