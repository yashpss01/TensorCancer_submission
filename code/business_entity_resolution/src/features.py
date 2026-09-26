"""Pairwise feature engineering for (Source-1 record, Source-2/3 record) candidates.

Input: a polars DataFrame with the normalised columns of both records, suffixed
``_a`` (Source 1) and ``_b`` (Source 2/3), plus the blocking similarity columns
``cos_name``, ``cos_addr``, ``cos_addrw`` (0 when the pair did not come from that
blocker).  Output: the same frame with numeric feature columns appended.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, Indel
from rapidfuzz.process import cpdist

_vow = re.compile(r"[aeiouyhw]")


def skeleton(s: str) -> str:
    """Consonant skeleton with common transliteration merges; robust to vowels."""
    s = s.replace("ph", "f").replace("v", "w").replace("z", "j").replace("q", "k").replace("c", "k")
    s = s.replace("x", "ks").replace("th", "t").replace("dh", "d").replace("sh", "s")
    s = _vow.sub("", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _tok_feats(df: pl.DataFrame, col: str, prefix: str) -> pl.DataFrame:
    a = pl.col(f"{col}_a").str.split(" ").list.drop_nulls().list.eval(pl.element().filter(pl.element() != ""))
    b = pl.col(f"{col}_b").str.split(" ").list.drop_nulls().list.eval(pl.element().filter(pl.element() != ""))
    return df.with_columns(
        a.list.len().alias(f"{prefix}_na"),
        b.list.len().alias(f"{prefix}_nb"),
        a.list.set_intersection(b).list.len().alias(f"{prefix}_ninter"),
        a.list.set_union(b).list.len().alias(f"{prefix}_nunion"),
        (a.list.first() == b.list.first()).cast(pl.Int8).fill_null(0).alias(f"{prefix}_first_eq"),
    ).with_columns(
        (pl.col(f"{prefix}_ninter") / pl.max_horizontal(pl.col(f"{prefix}_nunion"), 1)).alias(f"{prefix}_jac"),
        (pl.col(f"{prefix}_ninter") / pl.max_horizontal(pl.min_horizontal(f"{prefix}_na", f"{prefix}_nb"), 1)).alias(f"{prefix}_overlap"),
        (pl.col(f"{prefix}_na") - pl.col(f"{prefix}_nb")).abs().alias(f"{prefix}_ndiff"),
    )


def _str_sim(df: pl.DataFrame, col: str, scorer, name: str, workers: int = 8, **kw) -> np.ndarray:
    a = df[f"{col}_a"].to_list()
    b = df[f"{col}_b"].to_list()
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32, **kw)


def add_features(df: pl.DataFrame, workers: int = 8) -> pl.DataFrame:
    # ---- token features ----
    df = _tok_feats(df, "name_core", "nc")
    df = _tok_feats(df, "name_sig", "ns")
    df = _tok_feats(df, "addr_alpha", "aa")
    df = _tok_feats(df, "addr_nums", "an")

    # ---- string similarities (rapidfuzz, multithreaded C++) ----
    sims = {}
    sims["r_core"] = _str_sim(df, "name_core", fuzz.ratio, "r_core", workers)
    sims["r_core_tsort"] = _str_sim(df, "name_core", fuzz.token_sort_ratio, "r_core_tsort", workers)
    sims["r_core_tset"] = _str_sim(df, "name_core", fuzz.token_set_ratio, "r_core_tset", workers)
    sims["r_core_partial"] = _str_sim(df, "name_core", fuzz.partial_ratio, "r_core_partial", workers)
    sims["r_full"] = _str_sim(df, "name_full", fuzz.ratio, "r_full", workers)
    sims["r_nospace"] = _str_sim(df, "name_nospace", fuzz.ratio, "r_nospace", workers)
    sims["r_sig_tset"] = _str_sim(df, "name_sig", fuzz.token_set_ratio, "r_sig_tset", workers)
    sims["r_sig_partial"] = _str_sim(df, "name_sig", fuzz.partial_ratio, "r_sig_partial", workers)
    sims["lev_core"] = _str_sim(df, "name_core", Levenshtein.distance, "lev_core", workers)
    sims["r_addr_full"] = _str_sim(df, "addr_full", fuzz.ratio, "r_addr_full", workers)
    sims["r_addr_tsort"] = _str_sim(df, "addr_alpha", fuzz.token_sort_ratio, "r_addr_tsort", workers)
    sims["r_addr_tset"] = _str_sim(df, "addr_alpha", fuzz.token_set_ratio, "r_addr_tset", workers)
    sims["r_addr_partial"] = _str_sim(df, "addr_alpha", fuzz.partial_ratio, "r_addr_partial", workers)
    sims["r_nums"] = _str_sim(df, "addr_nums", fuzz.token_set_ratio, "r_nums", workers)
    # phonetic skeleton similarity of the core names
    ska = [skeleton(s) for s in df["name_core_a"].to_list()]
    skb = [skeleton(s) for s in df["name_core_b"].to_list()]
    sims["r_skel"] = cpdist(ska, skb, scorer=fuzz.ratio, workers=workers, dtype=np.float32)
    sims["r_skel_tset"] = cpdist(ska, skb, scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32)
    # house-number digit similarity
    sims["fn_lev"] = _str_sim(df, "addr_first_num", Levenshtein.distance, "fn_lev", workers)
    sims["r_addr_full_partial"] = _str_sim(df, "addr_full", fuzz.partial_ratio, "r_addr_full_partial", workers)
    df = df.with_columns([pl.Series(k, v) for k, v in sims.items()])
    fa, fb = pl.col("addr_first_num_a"), pl.col("addr_first_num_b")
    ia = fa.str.slice(0, 9).cast(pl.Int64, strict=False)
    ib = fb.str.slice(0, 9).cast(pl.Int64, strict=False)
    df = df.with_columns(
        pl.when(ia.is_null() | ib.is_null()).then(-1).otherwise((ia - ib).abs().clip(0, 100000)).alias("fn_absdiff"),
        pl.when(ia.is_null() | ib.is_null()).then(-1).otherwise(((ia - ib).abs() % 2)).alias("fn_diff_odd"),
        pl.when(ia.is_null() | ib.is_null() | (pl.max_horizontal(ia, ib) == 0)).then(-1.0)
          .otherwise(pl.min_horizontal(ia, ib) / pl.max_horizontal(ia, ib)).alias("fn_ratio"),
    )
    df = df.with_columns(
        pl.when((fa == "") | (fb == "")).then(0)
          .when(fa.str.starts_with(fb) | fb.str.starts_with(fa) | fa.str.ends_with(fb) | fb.str.ends_with(fa)).then(1)
          .otherwise(-1).cast(pl.Int8).alias("fn_affix"),
        (fa.str.len_chars().cast(pl.Int16) - fb.str.len_chars().cast(pl.Int16)).abs().alias("fn_lendiff"),
    )

    # ---- simple flags ----
    df = df.with_columns(
        (pl.col("name_sig_a") == pl.col("name_sig_b")).cast(pl.Int8).alias("sig_eq"),
        (pl.col("name_core_a") == pl.col("name_core_b")).cast(pl.Int8).alias("core_eq"),
        (pl.col("name_sorted_a") == pl.col("name_sorted_b")).cast(pl.Int8).alias("sorted_eq"),
        pl.when((pl.col("name_legal_a") == "") | (pl.col("name_legal_b") == "")).then(0)
          .when(pl.col("name_legal_a") == pl.col("name_legal_b")).then(1).otherwise(-1).cast(pl.Int8).alias("legal_agree"),
        pl.col("name_nonlatin_b").cast(pl.Int8).alias("nonlatin_b"),
        pl.col("addr_empty_b").cast(pl.Int8).alias("addr_empty_b"),
        pl.col("addr_empty_a").cast(pl.Int8).alias("addr_empty_a"),
        pl.when((pl.col("addr_state_a") == "") | (pl.col("addr_state_b") == "")).then(0)
          .when(pl.col("addr_state_a") == pl.col("addr_state_b")).then(1).otherwise(-1).cast(pl.Int8).alias("state_agree"),
        pl.when((pl.col("addr_first_num_a") == "") | (pl.col("addr_first_num_b") == "")).then(0)
          .when(pl.col("addr_first_num_a") == pl.col("addr_first_num_b")).then(1).otherwise(-1).cast(pl.Int8).alias("firstnum_agree"),
        pl.col("name_core_a").str.len_chars().alias("len_core_a"),
        pl.col("name_core_b").str.len_chars().alias("len_core_b"),
        pl.col("addr_alpha_a").str.len_chars().alias("len_addr_a"),
        pl.col("addr_alpha_b").str.len_chars().alias("len_addr_b"),
        pl.col("entity_id_b").str.slice(0, 2).eq("S2").cast(pl.Int8).alias("is_s2"),
    )
    df = df.with_columns(
        (pl.col("lev_core") / pl.max_horizontal(pl.max_horizontal("len_core_a", "len_core_b"), 1)).alias("lev_core_norm"),
        # first number of a appears anywhere in b's numbers or vice versa
        (pl.col("an_ninter") > 0).cast(pl.Int8).alias("any_num_common"),
    )
    return df


FEATURE_COLS = [
    "cos_joint", "cos_jskel", "cos_fullskel", "cos_name", "h", "h_rank", "h_gap", "h_ntie", "q_ncand_all", "src_extra",
    "s1_name_freq", "s1_sig_freq", "fn_lev", "fn_affix", "fn_lendiff", "r_addr_full_partial",
    "nc_na", "nc_nb", "nc_ninter", "nc_nunion", "nc_first_eq", "nc_jac", "nc_overlap", "nc_ndiff",
    "ns_na", "ns_nb", "ns_ninter", "ns_first_eq", "ns_jac", "ns_overlap",
    "aa_na", "aa_nb", "aa_ninter", "aa_nunion", "aa_first_eq", "aa_jac", "aa_overlap", "aa_ndiff",
    "an_na", "an_nb", "an_ninter", "an_jac", "an_first_eq",
    "r_core", "r_core_tsort", "r_core_tset", "r_core_partial", "r_full", "r_nospace", "r_sig_tset",
    "r_sig_partial", "lev_core", "lev_core_norm", "r_addr_full", "r_addr_tsort", "r_addr_tset",
    "r_addr_partial", "r_nums", "r_skel", "r_skel_tset",
    "sig_eq", "core_eq", "sorted_eq", "legal_agree", "nonlatin_b", "addr_empty_a", "addr_empty_b",
    "state_agree", "firstnum_agree", "len_core_a", "len_core_b", "len_addr_a", "len_addr_b", "is_s2",
    "any_num_common",
]
