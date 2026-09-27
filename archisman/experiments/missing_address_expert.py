"""Train an XGBoost expert for pairs whose address evidence is missing.

Fit on the old 30k training groups, tune a gate on exposed 10k development
groups, then freeze before any fresh-batch check. Never opens fresh labels.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from xgboost import XGBClassifier


def score(chosen, y, offsets, truth, countries):
    pred = np.diff(np.r_[0, np.cumsum(chosen, dtype=np.int64)][offsets])
    hit = np.diff(np.r_[0, np.cumsum(chosen & y, dtype=np.int64)][offsets])
    fp = pred - hit
    value = np.where(truth == 0, (pred == 0).astype(float),
                     1.25 * hit / np.maximum(hit + 0.25 * truth + fp, 1e-12))
    def summary(mask):
        return {
            "groups": int(mask.sum()), "macro_f0_5": float(value[mask].mean()),
            "tp": int(hit[mask].sum()), "fp": int(fp[mask].sum()),
            "fn": int((truth[mask] - hit[mask]).sum()),
        }
    return {"overall": summary(np.ones(len(truth), dtype=bool)),
            "countries": {country: summary(mask) for country, mask in countries.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    root = args.source_root
    sys.path.insert(0, str(root / "code/business_entity_resolution/round5"))
    import train
    art = root / "artifacts/matching_round5"
    tx = np.load(art / "train/X.npy", mmap_mode="r")
    te = np.load(art / "train/extra_pair.npy", mmap_mode="r")
    tg = np.load(art / "train/group_features.npy", mmap_mode="r")
    ty = np.load(art / "train/y.npy", mmap_mode="r").astype(bool)
    vx = np.load(art / "validation/X.npy", mmap_mode="r")
    ve = np.load(art / "validation/extra_pair.npy", mmap_mode="r")
    vg = np.load(art / "validation/group_features.npy", mmap_mode="r")
    vy = np.load(art / "validation/y.npy", mmap_mode="r").astype(bool)
    offsets = np.load(art / "validation/offsets.npy")
    truth, countries = train.group_truth("validation")
    base = (0.4 * np.load(art / "validation_group_extra_prob.npy")
            + 0.6 * np.load(art / "validation_augmented_0p15_prob.npy"))
    train_missing = (tx[:, 34] > 0.5) | (te[:, 11] > 0.5)
    val_missing = (vx[:, 34] > 0.5) | (ve[:, 11] > 0.5)
    # The group feature's first column is the earlier stage probability.
    # It is out of fold for the old training groups.
    train_features = np.column_stack((tx[train_missing], te[train_missing], tg[train_missing])).astype(np.float32)
    val_features = np.column_stack((vx[val_missing], ve[val_missing], vg[val_missing])).astype(np.float32)
    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        min_child_weight=6, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=4, tree_method="hist", n_jobs=4,
        objective="binary:logistic", eval_metric="logloss",
        random_state=20260927,
    )
    model.fit(train_features, ty[train_missing], verbose=False)
    expert = model.predict_proba(val_features)[:, 1].astype(np.float32)
    baseline = score(base >= 0.74, vy, offsets, truth, countries)
    grid = []
    for threshold in np.arange(0.50, 0.951, 0.025):
        chosen = base >= 0.74
        chosen[val_missing] = expert >= threshold
        result = score(chosen, vy, offsets, truth, countries)
        grid.append({"expert_threshold": float(threshold), **result})
    best = max(grid, key=lambda item: (item["overall"]["macro_f0_5"], -item["overall"]["fp"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "missing_address_expert.json"
    model.save_model(model_path)
    report = {
        "scope": "old 30k training groups, exposed 10k development groups, 409141-target reduced pool",
        "train_missing_pairs": int(train_missing.sum()),
        "train_missing_positives": int(ty[train_missing].sum()),
        "validation_missing_pairs": int(val_missing.sum()),
        "validation_missing_positives": int(vy[val_missing].sum()),
        "baseline": baseline,
        "best_development_choice": best,
        "grid": grid,
        "seconds": time.monotonic() - started,
        "model_path": str(model_path),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("baseline", "best_development_choice", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
