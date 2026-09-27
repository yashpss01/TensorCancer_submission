"""Step 5 (optional): stage-3 decoy-aware correction on top of the stage-2 probabilities.

Decoy records are near-copies of a Source-1 entity whose house number is shifted UP by a
small k in {1,2,3,4,5,7,9,11,13,21}; genuine noise on a true record is symmetric (digit drops,
padding, +-1).  The stage-1/2 number features are all symmetric, so they cannot see the sign.
Stage 3 adds, for the uncertain band 0.003 < p2 < 0.997:
  * signed shift of the first house number, its edit type, decoy / anti-decoy flag;
  * the same on the multiset of all address numbers (Indian addresses are re-ordered);
  * whether the legal form changed;
  * name tokens ADDED to / DROPPED from the Source-1 name, target-encoded per (country, token)
    out of fold (decoys add "group", "holdings", "enterprises" ...; noise adds "services", "center" ...).
Trained on the held-out validation pairs (entity folds 7-9) of train_full.py, which were scored
out of sample by the stage-1/2 models.  Applied to India/US; countries without training labels
(France) keep their stage-2 probability.  No stage-1/2 retrain, no re-blocking.

usage (from src/, PYTHONPATH=.):
  python stage3.py check  <model_dir>                   # train on folds 7-8, logloss on fold 9 (sanity)
  python stage3.py train  <model_dir>                   # final fit on folds 7-9
  python stage3.py apply  <model_dir> <in_dir> <out_dir> # rescore work/scored_test_*.parquet, decide, write tsv
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from collections import Counter

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline_core import STAGE2_COLS, assign_one_to_one, decide

ROOT = os.environ.get("ER_ROOT", "/Users/akanksha/aws/TensorCancer_submission")
W = ROOT + "/work/"
N = W + "norm/"
LO, HI = 0.003, 0.997
DECOY_K = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
TRAIN_COUNTRIES = ("India", "US")
OFF_A = {"India": 0, "US": 883188}      # offsets used by train_full.py in valid_scored_train.parquet
OFF_B = {"India": 0, "US": 4133346}

BASE = ["p2_logit", "is_us"] + [c for c in STAGE2_COLS if c != "p1_logit"]
N1 = ["fn_sdiff", "fn_etype", "fn_decoyk", "ms_onlya", "ms_onlyb", "ms_sdiff", "ms_decoyk"]
TE_NAMES = ["add_te_min", "add_te_max", "add_te_sum", "drop_te_min", "n_add", "n_drop"]
FEATS = BASE + N1 + ["legal_changed"]           # + TE_NAMES appended as a block
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "num_threads": 6, "seed": 7}
ROUNDS = 800
TE_M = 20.0


# --------------------------------------------------------------------------- number features
def _edit_type(x: str, y: str) -> int:
    lx, ly = len(x), len(y)
    if ly == lx - 1:
        for i in range(lx):
            if x[:i] + x[i + 1:] == y:
                return 1 if i == lx - 1 else 2
        return 7
    if ly == lx + 1:
        for i in range(ly):
            if y[:i] + y[i + 1:] == x:
                return 3 if i == ly - 1 else 4
        return 7
    if lx == ly:
        diff = [i for i in range(lx) if x[i] != y[i]]
        if len(diff) == 1:
            return 5 if diff[0] == lx - 1 else 6
        if len(diff) == 2 and diff[1] == diff[0] + 1 and x[diff[0]] == y[diff[1]] and x[diff[1]] == y[diff[0]]:
            return 8
        return 9
    if y.startswith(x) or y.endswith(x):
        return 10
    if x.startswith(y) or x.endswith(y):
        return 11
    return 12


def num_feats(fa, fb, na, nb) -> dict:
    n = len(fa)
    sd = np.full(n, np.nan, np.float32); et = np.zeros(n, np.int8); dk = np.zeros(n, np.int8)
    oa = np.zeros(n, np.int8); ob = np.zeros(n, np.int8); msd = np.full(n, np.nan, np.float32)
    mdk = np.zeros(n, np.int8)
    for i in range(n):
        x, y = fa[i] or "", fb[i] or ""
        if x and y and x != y:
            et[i] = _edit_type(x, y)
            try:
                d = int(y[:9]) - int(x[:9])
                sd[i] = max(-100000, min(100000, d))
                dk[i] = 1 if d in DECOY_K else (-1 if -d in DECOY_K else 0)
            except ValueError:
                pass
        A = Counter(na[i].split()) if na[i] else Counter()
        B = Counter(nb[i].split()) if nb[i] else Counter()
        OA, OB = A - B, B - A
        oa[i] = min(sum(OA.values()), 9); ob[i] = min(sum(OB.values()), 9)
        if OA and OB:
            best = None
            for u in OA:
                for v in OB:
                    try:
                        d = int(v[:9]) - int(u[:9])
                    except ValueError:
                        continue
                    if best is None or abs(d) < abs(best):
                        best = d
            if best is not None:
                msd[i] = max(-100000, min(100000, best))
                mdk[i] = 1 if best in DECOY_K else (-1 if -best in DECOY_K else 0)
    return dict(fn_sdiff=sd, fn_etype=et, fn_decoyk=dk, ms_onlya=oa, ms_onlyb=ob, ms_sdiff=msd, ms_decoyk=mdk)


# --------------------------------------------------------------------------- loading / features
def load_side(split: str, country: str, a_idx, b_idx):
    """norm columns for the needed LOCAL indices of one country (a: S1 rows, b: S2 rows then S3 rows)."""
    cols = ["addr_first_num", "addr_nums", "name_full", "name_legal"]
    s1 = (pl.scan_parquet(N + f"{split}_s1.parquet").filter(pl.col("country") == country).select(cols)
          .with_row_index("a").with_columns(pl.col("a").cast(pl.Int32))
          .filter(pl.col("a").is_in(pl.Series(a_idx).implode())).collect())
    n2 = pl.scan_parquet(N + f"{split}_s2.parquet").filter(pl.col("country") == country).select(pl.len()).collect().item()
    qs = []
    for s, off in ((f"{split}_s2", 0), (f"{split}_s3", n2)):
        qs.append(pl.scan_parquet(N + s + ".parquet").filter(pl.col("country") == country).select(cols)
                  .with_row_index("b").with_columns((pl.col("b").cast(pl.Int64) + off).cast(pl.Int32).alias("b"))
                  .filter(pl.col("b").is_in(pl.Series(b_idx).implode())).collect())
    return s1, pl.concat(qs)


def build(band: pl.DataFrame, split: str, country: str) -> pl.DataFrame:
    """band: rows with LO < p2 < HI carrying the STAGE2 columns (local a/b).  Adds stage-3 features."""
    s1, q = load_side(split, country, band["a"].unique(), band["b"].unique())
    s1 = s1.rename({"addr_first_num": "fa", "addr_nums": "na", "name_full": "name_full_a", "name_legal": "name_legal_a"})
    q = q.rename({"addr_first_num": "fb", "addr_nums": "nb", "name_full": "name_full_b", "name_legal": "name_legal_b"})
    band = band.join(s1, on="a", how="left").join(q, on="b", how="left")
    for c in ("fa", "fb", "na", "nb", "name_full_a", "name_full_b", "name_legal_a", "name_legal_b"):
        band = band.with_columns(pl.col(c).fill_null(""))
    f = num_feats(band["fa"].to_list(), band["fb"].to_list(), band["na"].to_list(), band["nb"].to_list())
    band = band.with_columns([pl.Series(k, v) for k, v in f.items()])
    return band.with_columns(
        pl.lit(country).alias("country"),
        pl.lit(1 if country == "US" else 0, dtype=pl.Int8).alias("is_us"),
        (pl.col("name_legal_a") != pl.col("name_legal_b")).cast(pl.Int8).alias("legal_changed"),
        ((pl.col("p2").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p2").clip(1e-6, 1 - 1e-6))).log()).alias("p2_logit"),
        pl.col("name_full_b").str.split(" ").list.set_difference(pl.col("name_full_a").str.split(" ")).alias("tok_add"),
        pl.col("name_full_a").str.split(" ").list.set_difference(pl.col("name_full_b").str.split(" ")).alias("tok_drop"),
    ).drop("fa", "fb", "na", "nb", "name_full_a", "name_full_b", "name_legal_a", "name_legal_b")


def _long(band: pl.DataFrame, col: str) -> pl.DataFrame:
    return (band.select("rid", "country", *[c for c in ("label", "fold") if c in band.columns], pl.col(col).alias("tok"))
            .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))


def te_tables(band: pl.DataFrame, folds) -> dict:
    sub = band.filter(pl.col("fold").is_in(folds))
    prior = float(sub["label"].mean()); lp = float(np.log(prior / (1 - prior)))
    out = {}
    for kind in ("tok_add", "tok_drop"):
        st = (_long(sub, kind).group_by("country", "tok")
              .agg(pl.len().alias("n"), pl.col("label").cast(pl.Int64).sum().alias("p")))
        out[kind] = st.with_columns(((pl.col("p") + TE_M * prior) / (pl.col("n") + TE_M)).alias("r")).with_columns(
            ((pl.col("r") / (1 - pl.col("r"))).log() - lp).alias("te")).select("country", "tok", "te")
    return out


def te_apply(band: pl.DataFrame, tabs: dict) -> np.ndarray:
    res = band.select("rid")
    for kind, pre in (("tok_add", "a"), ("tok_drop", "d")):
        L = _long(band, kind).join(tabs[kind], on=["country", "tok"], how="left").with_columns(pl.col("te").fill_null(0.0))
        agg = L.group_by("rid").agg(pl.col("te").min().alias(pre + "mn"), pl.col("te").max().alias(pre + "mx"),
                                    pl.col("te").sum().alias(pre + "sm"), pl.len().alias(pre + "n"))
        res = res.join(agg, on="rid", how="left")
    res = res.fill_null(0)
    return res.select("amn", "amx", "asm", "dmn", "an", "dn").to_numpy().astype(np.float32)


def matrix(band: pl.DataFrame, TE: np.ndarray) -> np.ndarray:
    return np.hstack([band.select(FEATS).to_numpy().astype(np.float32), TE])


# --------------------------------------------------------------------------- train
def train_band() -> pl.DataFrame:
    cols = list(dict.fromkeys(["a", "b", "label", "fold", "p2"] + [c for c in STAGE2_COLS if c != "p1_logit"]))
    bands = []
    for ctry in TRAIN_COUNTRIES:
        bd = (pl.scan_parquet(W + "valid_scored_train.parquet").filter(pl.col("country") == ctry)
              .filter((pl.col("p2") > LO) & (pl.col("p2") < HI))
              .with_columns((pl.col("a") - OFF_A[ctry]).cast(pl.Int32), (pl.col("b") - OFF_B[ctry]).cast(pl.Int32))
              .select(cols).collect())
        bands.append(build(bd, "train", ctry))
        del bd; gc.collect()
    return pl.concat(bands, how="diagonal_relaxed").with_row_index("rid")


def oof_te(band: pl.DataFrame, folds=(7, 8, 9)) -> np.ndarray:
    fold = band["fold"].to_numpy()
    TE = np.zeros((band.shape[0], len(TE_NAMES)), np.float32)
    for f in folds:
        m = fold == f
        if m.any():
            TE[m] = te_apply(band.filter(pl.Series(m)), te_tables(band, [g for g in folds if g != f]))
    return TE


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cmd_check(model_dir: str):
    """Sanity check of the recipe: fit on folds 7-8, compare band logloss on fold 9 with stage 2."""
    t0 = time.time()
    band = train_band()
    lab = band["label"].to_numpy().astype(np.int64); fold = band["fold"].to_numpy()
    TE = oof_te(band.filter(pl.Series(fold != 9)), folds=(7, 8))
    tr = band.filter(pl.Series(fold != 9)); te = band.filter(pl.Series(fold == 9))
    TE9 = te_apply(te, te_tables(band, [7, 8]))
    m = lgb.train(PARAMS, lgb.Dataset(matrix(tr, TE), lab[fold != 9]), ROUNDS)
    p3 = m.predict(matrix(te, TE9), num_threads=6)
    y9 = lab[fold == 9]; p2 = te["p2"].to_numpy()
    print(f"fold-9 band rows {len(y9)}: logloss stage2 {logloss(y9, p2):.4f} -> stage3 {logloss(y9, p3):.4f}; "
          f"mean p2 {p2.mean():.4f} p3 {p3.mean():.4f} label {y9.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(model_dir, exist_ok=True)
    te.select("a", "b").with_columns(pl.Series("p3", p3.astype(np.float32)), pl.Series("fold", fold[fold == 9])).write_parquet(
        os.path.join(model_dir, "check_fold9_p3.parquet"))


def cmd_train(model_dir: str):
    t0 = time.time()
    os.makedirs(model_dir, exist_ok=True)
    band = train_band()
    lab = band["label"].to_numpy().astype(np.int64)
    print("train band", band.shape, f"{time.time()-t0:.0f}s", flush=True)
    TE = oof_te(band)                                     # nested out-of-fold encoding for training rows
    tabs = te_tables(band, [7, 8, 9])                     # used at test time
    for kind in ("tok_add", "tok_drop"):
        tabs[kind].write_parquet(os.path.join(model_dir, f"stage3_te_{kind}.parquet"))
    m = lgb.train(PARAMS, lgb.Dataset(matrix(band, TE), lab), ROUNDS)
    m.save_model(os.path.join(model_dir, "stage3.txt"))
    json.dump({"FEATS": FEATS + TE_NAMES, "LO": LO, "HI": HI, "countries": list(TRAIN_COUNTRIES), "rounds": ROUNDS},
              open(os.path.join(model_dir, "stage3.json"), "w"), indent=1)
    print("stage-3 trained", f"{time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- apply to test
def cmd_apply(model_dir: str, in_dir: str, out_dir: str):
    """in_dir: the stage-2 submission folder (for candidate_pairs.tsv, which does not change)."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    m = lgb.Booster(model_file=os.path.join(model_dir, "stage3.txt"))
    tabs = {k: pl.read_parquet(os.path.join(model_dir, f"stage3_te_{k}.parquet")) for k in ("tok_add", "tok_drop")}
    all_s1 = pl.read_parquet(N + "test_s1.parquet", columns=["entity_id", "country"])
    keep = list(dict.fromkeys(["a", "b", "p2"] + [c for c in STAGE2_COLS if c != "p1_logit"]))
    lists = []
    for ctry in sorted(all_s1["country"].unique().to_list()):
        lf = pl.scan_parquet(W + f"scored_test_{ctry}.parquet")
        sc = lf.select("a", "b", "p2").collect()
        if ctry in TRAIN_COUNTRIES:
            bd = build(lf.filter((pl.col("p2") > LO) & (pl.col("p2") < HI)).select(keep).collect(), "test", ctry).with_row_index("rid")
            p3 = m.predict(matrix(bd, te_apply(bd, tabs)), num_threads=6)
            upd = bd.select("a", "b").with_columns(pl.Series("p3", p3.astype(np.float32)))
            del bd; gc.collect()
            sc = sc.join(upd, on=["a", "b"], how="left").with_columns(pl.coalesce("p3", "p2").alias("p3"))
        else:
            sc = sc.with_columns(pl.col("p2").alias("p3"))          # no training labels for this country
        pred = decide(assign_one_to_one(sc, "p3"), "p3", mode="expected")
        s1_ids = all_s1.filter(pl.col("country") == ctry)["entity_id"].to_numpy()
        ids2 = pl.scan_parquet(N + "test_s2.parquet").filter(pl.col("country") == ctry).select("entity_id").collect()["entity_id"]
        ids3 = pl.scan_parquet(N + "test_s3.parquet").filter(pl.col("country") == ctry).select("entity_id").collect()["entity_id"]
        q_ids = pl.concat([ids2, ids3]).to_numpy()
        p = pred.with_columns(pl.Series("source1_entity_id", s1_ids[pred["a"].to_numpy()]),
                              pl.Series("q_id", q_ids[pred["b"].to_numpy()]))
        lists.append(p.group_by("source1_entity_id").agg(pl.col("q_id").sort().str.join(",").alias("matched_entity_ids")))
        print(f"[{ctry}] accepted {pred.shape[0]} ({pred.shape[0]/len(s1_ids):.3f}/S1), {time.time()-t0:.0f}s", flush=True)
        del sc, pred, p; gc.collect()
    out = all_s1.select(pl.col("entity_id").alias("source1_entity_id")).join(
        pl.concat(lists), on="source1_entity_id", how="left", maintain_order="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, ids in out.iter_rows():
            f.write(f"{sid}\t{ids}\n")
    shutil.copyfile(os.path.join(in_dir, "candidate_pairs.tsv"), os.path.join(out_dir, "candidate_pairs.tsv"))
    print("written", out_dir, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        cmd_check(sys.argv[2])
    elif cmd == "train":
        cmd_train(sys.argv[2])
    elif cmd == "apply":
        cmd_apply(sys.argv[2], sys.argv[3], sys.argv[4])
