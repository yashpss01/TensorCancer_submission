"""Learn a token-level transliteration dictionary from the TRAINING pairs.

Many Source-2/3 names are written in an Indic script (Devanagari, Bengali, Tamil,
Kannada, Telugu, Gujarati, Punjabi ...) while the matching Source-1 name is in
Latin script.  Because the pool of business-name words is small, the same word
is rendered the same way in a given script every time, so a simple positional
alignment of (Indic name, English name) pairs with the same token count yields a
high-coverage dictionary.  Only the provided training data is used.

Usage: python translit.py --data-dir <dataset> --out <src/translit_dict.json>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import polars as pl

from normalize import has_non_latin, LEGAL_MAP

_tok = re.compile(r"[^\s\.,;:\-/()\[\]{}'\"&+#!?|]+", re.UNICODE)


def tokens(s: str) -> list[str]:
    return [t.lower() for t in _tok.findall(s)]


def english_tokens(s: str) -> list[str]:
    s = s.lower().replace("&", " and ")
    toks = tokens(s)
    return toks


def build(data_dir: str, min_support: int = 2, min_dom: float = 0.5) -> dict:
    gt = pl.read_csv(f"{data_dir}/train/train_ground_truth.tsv", separator="\t", quote_char=None,
                     empty_string_is_null=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    pairs = (gt.filter(pl.col("matched_entity_ids") != "")
             .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .rename({"matched_entity_ids": "mid", "source1_entity_id": "s1"}))
    s1 = pl.read_csv(f"{data_dir}/train/train_source1.tsv", separator="\t", quote_char=None,
                     empty_string_is_null=False).select("entity_id", "business_name")
    s23 = pl.concat([pl.read_csv(f"{data_dir}/train/train_source{i}.tsv", separator="\t", quote_char=None,
                                 empty_string_is_null=False).select("entity_id", "business_name") for i in (2, 3)])
    s23 = s23.with_columns(pl.Series("nonlat", [has_non_latin(x or "") for x in s23["business_name"].to_list()]))
    s23 = s23.filter(pl.col("nonlat"))
    j = pairs.join(s23, left_on="mid", right_on="entity_id").join(s1, left_on="s1", right_on="entity_id", suffix="_en")
    print("non-latin training pairs:", j.shape[0])
    counts: dict[str, Counter] = defaultdict(Counter)
    n_aligned = 0
    for ind, en in zip(j["business_name"].to_list(), j["business_name_en"].to_list()):
        it = tokens(ind)
        et = english_tokens(en)
        # keep only the Indic tokens (mixed-script names exist)
        if len(it) != len(et):
            continue
        n_aligned += 1
        for a, b in zip(it, et):
            if has_non_latin(a):
                counts[a][b] += 1
    print("aligned pairs:", n_aligned, "distinct indic tokens:", len(counts))
    out = {}
    for a, c in counts.items():
        b, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_support and n / tot >= min_dom:
            out[a] = b
    print("dictionary size:", len(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = build(a.data_dir)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=0)
    print("saved", a.out)


if __name__ == "__main__":
    main()
