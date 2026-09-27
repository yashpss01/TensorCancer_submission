"""Step 3 (full-density variant): train and validate on the complete training set.

Same fold logic as train.py (entity folds 0-3 stage 1, 4-6 stage 2, 7-9 validation)
but every country is processed separately so that only one country's records and
candidates are in memory at a time.  Per-country row indices are offset so that the
light frames of all countries can be concatenated for the context model.

Usage: python train_full.py --work-dir <work> --split train --models-dir ../models
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from features import FEATURE_COLS
from pipeline_core import (NORM_COLS, STAGE2_COLS, add_context, add_labels, assign_one_to_one, decide, featurize,
                           load_cands, macro_f05, score_stage1_chunked, stage1_matrix, stage2_matrix, truth_pairs)
from train import LGB_PARAMS, fold_of, train_lgb


def load_country(work_dir: str, split: str, country: str):
    nd = os.path.join(work_dir, "norm")
    s1 = (pl.scan_parquet(os.path.join(nd, f"{split}_s1.parquet")).filter(pl.col("country") == country)
          .select(NORM_COLS + ["country"]).collect())
    qs = pl.concat([pl.scan_parquet(os.path.join(nd, f"{split}_s{i}.parquet")).filter(pl.col("country") == country)
                    .select(NORM_COLS).collect() for i in (2, 3)])
    return s1, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--s1-pos-per-country", type=int, default=900_000)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.models_dir, exist_ok=True)
    tmp = os.path.join(a.work_dir, f"trainfull_{a.split}")
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    countries = sorted(pl.scan_parquet(os.path.join(a.work_dir, "norm", f"{a.split}_s1.parquet"))
                       .select("country").unique().collect()["country"].to_list())
    print("countries", countries, flush=True)

    # ---------------- pass 1: stage-1 training samples per country ----------------
    # Two disjoint samples (entity folds {0,1} and {2,3}) so that stage-1 scores used as
    # *context* for the stage-2 model can be cross-fitted (never in-sample).
    XA, yA, XB, yB, Xes, yes = [], [], [], [], [], []
    half = a.s1_pos_per_country // 2
    for ctry in countries:
        s1, qs = load_country(a.work_dir, a.split, ctry)
        c = load_cands(a.work_dir, a.split, s1, qs, country=ctry)
        c = add_labels(c, a.work_dir, s1, qs)
        c = c.with_columns(pl.Series("fold", fold_of(c["a"].to_numpy()).astype(np.int8)))
        for grp, folds, Xl, yl in (("A", [0, 1], XA, yA), ("B", [2, 3], XB, yB)):
            tr = c.filter(pl.col("fold").is_in(folds))
            pos = tr.filter(pl.col("label") == 1)
            neg = tr.filter(pl.col("label") == 0)
            n_pos = min(pos.shape[0], half)
            pos = pos.sample(n=n_pos, seed=1)
            neg = neg.sample(n=min(neg.shape[0], int(n_pos * a.neg_ratio)), seed=2)
            samp = pl.concat([pos, neg])
            df = featurize(samp, s1, qs, a.workers)
            Xl.append(stage1_matrix(df)); yl.append(df["label"].to_numpy())
            print(f"[{ctry}] stage-1 sample {grp}: {samp.shape[0]} (pos {n_pos}), {time.time()-t0:.0f}s", flush=True)
            del tr, pos, neg, samp, df
        f4 = c.filter(pl.col("fold") == 4)
        es = f4.sample(n=min(300_000, f4.shape[0]), seed=4)
        dfe = featurize(es, s1, qs, a.workers)
        Xes.append(stage1_matrix(dfe)); yes.append(dfe["label"].to_numpy())
        del c, dfe, f4, es, s1, qs
        gc.collect()
    XA = np.concatenate(XA); yA = np.concatenate(yA); XB = np.concatenate(XB); yB = np.concatenate(yB)
    Xe = np.concatenate(Xes); ye = np.concatenate(yes)
    del Xes, yes
    mA = train_lgb(XA, yA, Xe, ye)          # scores folds 2-3 (out of sample)
    mB = train_lgb(XB, yB, Xe, ye)          # scores folds 0-1 (out of sample)
    X = np.concatenate([XA, XB]); y = np.concatenate([yA, yB])
    del XA, XB, yA, yB
    print("stage-1 matrix", X.shape, flush=True)
    m1 = train_lgb(X, y, Xe, ye)            # final model: scores folds 4-9 here and everything at test time
    m1.save_model(os.path.join(a.models_dir, "stage1.txt"))
    imp = sorted(zip(FEATURE_COLS, m1.feature_importance("gain")), key=lambda x: -x[1])
    print("stage-1 best iter", m1.best_iteration, "(A", mA.best_iteration, "B", mB.best_iteration, ") top features:",
          [(k, int(v)) for k, v in imp[:25]], flush=True)
    del X, y, Xe, ye
    gc.collect()

    # ---------------- pass 2: score folds >= 4 per country ----------------
    off_a = off_b = 0
    lights, truths, all_as, bys = [], [], [], []
    for ctry in countries:
        s1, qs = load_country(a.work_dir, a.split, ctry)
        c = load_cands(a.work_dir, a.split, s1, qs, country=ctry)
        c = add_labels(c, a.work_dir, s1, qs)
        c = c.with_columns(pl.Series("fold", fold_of(c["a"].to_numpy()).astype(np.int8)))
        parts = []
        for folds, model in (([0, 1], mB), ([2, 3], mA), ([4, 5, 6, 7, 8, 9], m1)):
            sub = c.filter(pl.col("fold").is_in(folds))
            if sub.shape[0]:
                parts.append(score_stage1_chunked(sub, s1, qs, model, chunk=1_000_000, workers=a.workers,
                                                  extra_cols=("fold",)))
            del sub
        del c
        light = pl.concat(parts); del parts
        # competition features over the COMPLETE candidate groups (as at test time), then keep
        # only the rows of entities used for stage 2 / validation
        light = add_context(light).filter(pl.col("fold") >= 4)
        light = light.with_columns((pl.col("a") + off_a).alias("a"), (pl.col("b") + off_b).alias("b"))
        light.write_parquet(os.path.join(tmp, f"light_{ctry}.parquet"))
        lights.append(light)
        tr = truth_pairs(a.work_dir, s1, qs)
        truths.append(tr.with_columns((pl.col("a") + off_a).alias("a"), (pl.col("b") + off_b).alias("b")))
        n_a = s1.shape[0]
        all_as.append(np.arange(n_a, dtype=np.int32) + off_a)
        bys.append(pl.DataFrame({"a": (np.arange(n_a, dtype=np.int32) + off_a), "country": [ctry] * n_a}))
        print(f"[{ctry}] scored light frame {light.shape}, {time.time()-t0:.0f}s", flush=True)
        off_a += n_a
        off_b += qs.shape[0]
        del s1, qs, light
        gc.collect()
    light = pl.concat(lights); del lights
    truth = pl.concat(truths); del truths
    all_a = np.concatenate(all_as)
    by = pl.concat(bys).with_columns(pl.col("a").cast(pl.Int32))
    print("context features ready", light.shape, f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- stage 2 ----------------
    tr2 = light.filter(pl.col("fold").is_in([4, 5]))
    es2 = light.filter(pl.col("fold") == 6)
    if tr2.shape[0] > 8_000_000:
        tr2 = tr2.sample(n=8_000_000, seed=5)
    if es2.shape[0] > 1_000_000:
        es2 = es2.sample(n=1_000_000, seed=6)
    m2 = train_lgb(stage2_matrix(tr2), tr2["label"].to_numpy(), stage2_matrix(es2), es2["label"].to_numpy())
    m2.save_model(os.path.join(a.models_dir, "stage2.txt"))
    imp2 = sorted(zip(STAGE2_COLS, m2.feature_importance("gain")), key=lambda x: -x[1])
    print("stage-2 best iter", m2.best_iteration, "top features:", [(k, int(v)) for k, v in imp2[:20]], flush=True)
    del tr2, es2
    gc.collect()

    # ---------------- validation (folds 7-9, full density) ----------------
    va = light.filter(pl.col("fold") >= 7)
    va = va.with_columns(pl.Series("p2", m2.predict(stage2_matrix(va), num_threads=a.workers).astype(np.float32)))
    # fold of an offset index: recover per-country local index for fold_of
    loc = np.concatenate([np.arange(len(x), dtype=np.int64) for x in all_as])
    fold_all = fold_of(loc)
    va_a = all_a[fold_all >= 7]
    # map offset index -> local index (folds are defined on per-country local indices)
    lookup = np.empty(len(all_a), dtype=np.int64); lookup[all_a] = loc
    truth_va = truth.filter(pl.Series(fold_of(lookup[truth["a"].to_numpy()]) >= 7))
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("valid pairs", va.shape, "AUC p1", round(roc_auc_score(va["label"], va["p1"]), 5),
          "AUC p2", round(roc_auc_score(va["label"], va["p2"]), 5),
          "AP p2", round(average_precision_score(va["label"], va["p2"]), 5), flush=True)
    assigned = assign_one_to_one(va, "p2")
    results = {}
    for thr in [0.5, 0.6, 0.7, 0.8]:
        pred = decide(assigned, "p2", mode="threshold", threshold=thr)
        r, _ = macro_f05(pred, truth_va, va_a, by)
        results[f"thr_{thr}"] = r
        print(f"threshold {thr}: {r}", flush=True)
    pred = decide(assigned, "p2", mode="expected")
    r, m = macro_f05(pred, truth_va, va_a, by)
    results["expected"] = r
    print(f"expected-F decision: {r}", flush=True)
    best = max(results, key=lambda k: results[k]["macro_f05"])
    cfg = {"decision": "expected" if best == "expected" else "threshold",
           "threshold": float(best.split("_")[1]) if best.startswith("thr_") else 0.5,
           "results": results, "stage1_iter": m1.best_iteration, "stage2_iter": m2.best_iteration,
           "trained_on": f"{a.split} (full density)"}
    with open(os.path.join(a.models_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("best:", best, cfg["results"][best], f"total {time.time()-t0:.0f}s", flush=True)
    va.write_parquet(os.path.join(a.work_dir, f"valid_scored_{a.split}.parquet"))
    m.write_parquet(os.path.join(a.work_dir, f"valid_entities_{a.split}.parquet"))


if __name__ == "__main__":
    main()
