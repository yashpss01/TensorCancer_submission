"""Development-only transliteration feature ablation for Unicode name pairs."""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import re
import sqlite3
import sys
import time

import numpy as np
from rapidfuzz import fuzz
from unidecode import unidecode
from xgboost import XGBClassifier


@functools.lru_cache(maxsize=500_000)
def normalized(name):
    value = re.sub(r"[^a-z0-9]+", " ", unidecode(name).casefold()).strip()
    core = " ".join(token for token in value.split()
                    if token not in {"private", "limited", "pvt", "ltd", "inc", "llc", "company", "co"})
    return value, core


def string_features(query, target):
    full_q, core_q = normalized(query)
    full_t, core_t = normalized(target)
    if not full_q or not full_t:
        return (0.0, 0.0, 0.0)
    return (fuzz.ratio(full_q, full_t) / 100.0,
            fuzz.token_set_ratio(core_q or full_q, core_t or full_t) / 100.0,
            fuzz.WRatio(core_q or full_q, core_t or full_t) / 100.0)


def extract(root, split, mask, offsets, select, records):
    if split == "train":
        base = root / "artifacts/blocking_round3/fresh"
        queries = json.loads((base / "queries.json").read_text())[:30_000]
    else:
        base = root / "artifacts/blocking_round4/fresh"
        queries = json.loads((base / "queries.json").read_text())
    out = np.empty((int(mask.sum()), 3), dtype=np.float32)
    position = 0
    with (base / "improved.jsonl").open() as handle:
        for i, (query, line) in enumerate(zip(queries, handle)):
            mids = select(json.loads(line), 0.5, 16)
            lo, hi = offsets[i:i+2]
            if hi - lo != len(mids):
                raise ValueError(f"Pair alignment mismatch at {split} group {i}")
            for j, mid in enumerate(mids):
                if mask[lo+j]:
                    out[position] = string_features(query["business_name"], records[mid])
                    position += 1
            if (i+1) % 5000 == 0:
                print(split, i+1, "groups", position, "unicode pairs", flush=True)
    if position != len(out):
        raise ValueError(f"Incomplete {split} feature extraction")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    root = args.source_root
    sys.path[:0] = [str(root / "code/business_entity_resolution/round5"),
                    str(root / "code/business_entity_resolution/round2")]
    import train
    from improve import select

    art = root / "artifacts/matching_round5"
    X = np.load(art / "train/X.npy", mmap_mode="r")
    E = np.load(art / "train/extra_pair.npy", mmap_mode="r")
    G = np.load(art / "train/group_features.npy", mmap_mode="r")
    y = np.load(art / "train/y.npy", mmap_mode="r").astype(bool)
    train_offsets = np.load(art / "train/offsets.npy")
    VX = np.load(art / "validation/X.npy", mmap_mode="r")
    VE = np.load(art / "validation/extra_pair.npy", mmap_mode="r")
    VG = np.load(art / "validation/group_features.npy", mmap_mode="r")
    vy = np.load(art / "validation/y.npy", mmap_mode="r").astype(bool)
    offsets = np.load(art / "validation/offsets.npy")
    train_mask = X[:, 20] > 0.5
    val_mask = VX[:, 20] > 0.5
    db_path = root / "artifacts/blocking_round3/fresh/index.sqlite"
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    records = {mid: name for mid, name in db.execute("SELECT id,name FROM records")}
    db.close()
    train_new = extract(root, "train", train_mask, train_offsets, select, records)
    val_new = extract(root, "validation", val_mask, offsets, select, records)
    train_features = np.column_stack((X[train_mask], E[train_mask], G[train_mask, 1:], train_new)).astype(np.float32)
    val_features = np.column_stack((VX[val_mask], VE[val_mask], VG[val_mask, 1:], val_new)).astype(np.float32)
    model = XGBClassifier(n_estimators=450, max_depth=5, learning_rate=0.05,
                          min_child_weight=6, subsample=0.9, colsample_bytree=0.9,
                          reg_lambda=4, tree_method="hist", n_jobs=4,
                          objective="binary:logistic", eval_metric="logloss",
                          random_state=20260927)
    model.fit(train_features, y[train_mask], verbose=False)
    expert = model.predict_proba(val_features)[:, 1].astype(np.float32)
    base = (0.4 * np.load(art / "validation_group_extra_prob.npy")
            + 0.6 * np.load(art / "validation_augmented_0p15_prob.npy"))
    truth, countries = train.group_truth("validation")
    baseline = train.score_prob(base, vy, offsets, truth, countries, 0.74)
    grid = []
    for threshold in np.arange(0.5, 0.951, 0.025):
        chosen = base >= 0.74
        chosen[val_mask] = expert >= threshold
        metrics = train.score_prob(chosen.astype(np.float32), vy, offsets, truth, countries, 0.5)
        grid.append({"expert_threshold": float(threshold), **metrics})
    best = max(grid, key=lambda item: (item["overall"]["macro_f05"], -item["overall"]["pair_fp"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out_dir / "script_aware_expert.json")
    report = {"scope": "old training30k to exposed development10k, 409141-target reduced pool",
              "new_features": ["unidecode_full_ratio", "unidecode_core_token_set_ratio", "unidecode_core_WRatio"],
              "train_unicode_pairs": int(train_mask.sum()),
              "validation_unicode_pairs": int(val_mask.sum()),
              "baseline": baseline, "best_development_choice": best,
              "grid": grid, "seconds": time.monotonic() - started}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"baseline": baseline["overall"],
                      "best": best, "seconds": report["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
