"""Screen reverse sparse retrieval on an exposed development cohort.

The full training Source-1 corpus competes for each known-positive target.
Only target queries from the exposed cohort are processed, so this measures
positive-link rank coverage, not final candidate counts or matcher quality.
It must not be used on a sealed confirmation cohort.
"""

import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from unidecode import unidecode

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from blocking import rows, tokens
from infer import target_records

RANKS = (16, 32, 64, 128, 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=pathlib.Path, required=True)
    parser.add_argument("--truth", type=pathlib.Path, required=True)
    parser.add_argument("--target-index", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    started = time.monotonic()
    source = collections.defaultdict(lambda: ([], [], []))
    source_country = {}
    with args.source1.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            country = row["country"]
            source[country][0].append(row["entity_id"])
            source[country][1].append(" ".join(tokens(row["business_name"])))
            source[country][2].append(" ".join(tokens(row["business_address"])))
            source_country[row["entity_id"]] = country
    print("loaded", len(source_country), "full Source-1 records in", round(time.monotonic()-started, 1), "s", flush=True)

    owner = {}
    for row in rows(args.truth):
        for target_id in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else ():
            if target_id in owner:
                raise ValueError("Positive target has multiple Source-1 owners")
            owner[target_id] = row["source1_entity_id"]
    db = sqlite3.connect(f"file:{args.target_index}?mode=ro", uri=True)
    targets = target_records(db, owner)
    db.close()
    queries = collections.defaultdict(lambda: ([], [], [], []))
    cross_country = 0
    for target_id, s1_id in owner.items():
        name, address, country = targets[target_id]
        if source_country.get(s1_id) != country:
            cross_country += 1
            continue
        queries[country][0].append(target_id)
        queries[country][1].append(" ".join(tokens(name)))
        queries[country][2].append(" ".join(tokens(address)))
        queries[country][3].append(s1_id)
    print("positive targets", len(owner), "cross-country", cross_country, flush=True)
    report = {"scope": "exposed positive-target rank diagnostic; no negative target queries",
              "full_s1_competition": len(source_country), "positive_links": len(owner),
              "cross_country": cross_country, "countries": {}}
    for country, (target_ids, target_names, target_addresses, true_s1) in queries.items():
        s1_ids, s1_names, s1_addresses = source[country]
        index_by_id = {entity_id: i for i, entity_id in enumerate(s1_ids)}
        country_report = {"source1_count": len(s1_ids),
                          "positive_links": len(target_ids), "views": {}}
        union_ranks = np.full(len(target_ids), RANKS[-1]+1, dtype=np.int32)
        view_inputs = (
            ("name_words", s1_names, target_names, "word"),
            ("address_words", s1_addresses, target_addresses, "word"),
            ("name_chars", None, None, "char_wb"),
        )
        for view_name, s1_docs, target_docs, analyzer in view_inputs:
            if view_name == "name_chars":
                s1_docs = [unidecode(x) for x in s1_names]
                target_docs = [unidecode(x) for x in target_names]
            print(country, view_name, "fit", len(s1_docs), "S1 documents", flush=True)
            options = dict(analyzer=analyzer, ngram_range=(1, 2) if analyzer == "word" else (3, 5),
                           lowercase=False, dtype=np.float32, sublinear_tf=True)
            if analyzer == "word":
                options["token_pattern"] = r"(?u)\b\w+\b"
            else:
                options["min_df"] = 2
            vectorizer = TfidfVectorizer(**options)
            matrix = vectorizer.fit_transform(s1_docs).tocsr()
            right = matrix.T.tocsr()
            query_matrix = vectorizer.transform(target_docs).tocsr()
            positive_ranks = np.full(len(target_ids), RANKS[-1]+1, dtype=np.int32)
            for start in range(0, len(target_ids), 1000):
                stop = min(start+1000, len(target_ids))
                scores = sp_matmul_topn(query_matrix[start:stop], right,
                                        top_n=RANKS[-1], threshold=0.0, sort=True,
                                        n_threads=args.workers).tocsr()
                for j in range(stop-start):
                    begin, end = scores.indptr[j:j+2]
                    hits = scores.indices[begin:end]
                    where = np.flatnonzero(hits == index_by_id[true_s1[start+j]])
                    if where.size:
                        positive_ranks[start+j] = int(where[0])+1
                print(country, view_name, "ranked", stop, "positives in",
                      round(time.monotonic()-started, 1), "s", flush=True)
            union_ranks = np.minimum(union_ranks, positive_ranks)
            found = {str(k): int((positive_ranks <= k).sum()) for k in RANKS}
            country_report["views"][view_name] = {"topk_found": found,
                                                 "topk_recall": {k: found[str(k)]/len(target_ids) for k in RANKS}}
            del matrix, right, query_matrix, vectorizer
            if view_name == "name_chars":
                del s1_docs, target_docs
        union_found = {str(k): int((union_ranks <= k).sum()) for k in RANKS}
        country_report["union"] = {"topk_found": union_found,
                                   "topk_recall": {k: union_found[str(k)]/len(target_ids) for k in RANKS}}
        report["countries"][country] = country_report
        del index_by_id
    report["seconds"] = time.monotonic()-started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
