"""Development-only model-family screen on the frozen candidate features.

This reuses the existing entity-disjoint 30k training and 10k development
groups. Its scores select a model for a later, disjoint validation batch; they
are not a fresh quality claim or a full-corpus result.
"""

import argparse
import hashlib
import json
import pathlib
import resource
import sys
import time

import numpy as np


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_arrays(artifact_root, split):
    directory = artifact_root / split
    x = np.load(directory / "X.npy", mmap_mode="r")
    e = np.load(directory / "extra_pair.npy", mmap_mode="r")
    group = np.load(directory / "group_features.npy", mmap_mode="r")
    y = np.load(directory / "y.npy", mmap_mode="r")
    offsets = np.load(directory / "offsets.npy")
    assert len(x) == len(e) == len(group) == len(y) == int(offsets[-1])
    assert np.isfinite(x).all() and np.isfinite(e).all() and np.isfinite(group).all()
    # group[:, 0] duplicates the first-stage pair score already represented
    # elsewhere in the existing group-extra baseline.
    features = np.concatenate((x, e, group[:, 1:]), axis=1).astype("float32", copy=False)
    return features, y, offsets, directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("lightgbm", "catboost"), required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root / "code/business_entity_resolution/round5"))
    import train as existing

    artifact_root = args.source_root / "artifacts/matching_round5"
    output = args.output_dir / args.model
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    train_x, train_y, train_offsets, train_dir = load_arrays(artifact_root, "train")
    val_x, val_y, val_offsets, val_dir = load_arrays(artifact_root, "validation")
    assert train_x.shape[1] == val_x.shape[1] == 85
    assert len(train_offsets) == 30001 and len(val_offsets) == 10001
    counts, masks = existing.group_truth("validation")

    if args.model == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        model = LGBMClassifier(
            n_estimators=1200, learning_rate=0.05, num_leaves=63,
            max_depth=8, min_child_samples=10, colsample_bytree=0.9,
            subsample=0.9, subsample_freq=1, reg_lambda=2,
            objective="binary", n_jobs=8, random_state=20260927,
            verbosity=-1,
        )
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)],
                  eval_metric="binary_logloss",
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        best_iteration = int(model.best_iteration_)
        model_path = output / "model.txt"
        model.booster_.save_model(str(model_path))
    else:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=900, depth=7, learning_rate=0.05,
            l2_leaf_reg=2, loss_function="Logloss", eval_metric="Logloss",
            random_seed=20260927, thread_count=8, od_type="Iter", od_wait=50,
            verbose=False, allow_writing_files=False,
        )
        model.fit(train_x, train_y, eval_set=(val_x, val_y),
                  use_best_model=True, verbose=False)
        best_iteration = int(model.best_iteration_)
        model_path = output / "model.cbm"
        model.save_model(str(model_path))

    fit_seconds = time.monotonic() - start
    predict_start = time.monotonic()
    probability = model.predict_proba(val_x)[:, 1].astype("float32")
    predict_seconds = time.monotonic() - predict_start
    np.save(output / "development_prob.npy", probability)
    grid, best = existing.optimize(probability, val_y, val_offsets, counts, masks)
    aggregate = best["overall"]
    tp, fp, fn = (aggregate[k] for k in ("pair_tp", "pair_fp", "pair_fn"))
    result = {
        "status": "development_only",
        "model": args.model,
        "training_groups": len(train_offsets) - 1,
        "development_groups": len(val_offsets) - 1,
        "train_pairs": len(train_y),
        "development_pairs": len(val_y),
        "features": train_x.shape[1],
        "best_iteration": best_iteration,
        "development_threshold": best["threshold"],
        "overall": aggregate,
        "countries": best["countries"],
        "pair_precision": tp / (tp + fp),
        "pair_recall": tp / (tp + fn),
        "fit_seconds_including_array_load": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "train_meta_sha256": digest(train_dir / "meta.json"),
        "development_meta_sha256": digest(val_dir / "meta.json"),
        "model_sha256": digest(model_path),
        "parameters": model.get_params(),
    }
    (output / "development_metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (output / "development_threshold_grid.json").write_text(json.dumps(grid, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("model", "best_iteration", "overall", "countries", "pair_precision", "pair_recall", "fit_seconds_including_array_load", "predict_seconds")}, indent=2))


if __name__ == "__main__":
    main()
