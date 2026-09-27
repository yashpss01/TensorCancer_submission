"""Measure own batched S1-to-target sparse retrieval on an exposed cohort.

This screen loads the complete positive-enriched reduced target pool for a
previously exposed 10k cohort. It computes candidate recall and volume only;
it does not select a production blocker or claim full-test performance.
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from unidecode import unidecode

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from blocking import rows, tokens
from phonetic import key as phonetic_key

RANKS = (8, 16, 32, 64, 128)
CAPS = (64, 100, 128, 160, 200)


def normalized(name, address):
    return " ".join(tokens(name)), " ".join(tokens(address))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=pathlib.Path, required=True)
    parser.add_argument("--truth", type=pathlib.Path, required=True)
    parser.add_argument("--target-index", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--candidate-out", type=pathlib.Path,
                        help="Write the exact label-blind per-S1 candidate union")
    parser.add_argument("--export-rank", type=int, choices=RANKS, default=64)
    args = parser.parse_args()
    started = time.monotonic()
    truth = {}
    for row in rows(args.truth):
        truth[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    source = collections.defaultdict(lambda: ([], [], [], []))
    for row in rows(args.source1):
        name, address = normalized(row["business_name"], row["business_address"])
        ids, names, addresses, raw_names = source[row["country"]]
        ids.append(row["entity_id"])
        names.append(name)
        addresses.append(address)
        raw_names.append(row["business_name"])
    db = sqlite3.connect(f"file:{args.target_index}?mode=ro", uri=True)
    target = collections.defaultdict(lambda: ([], [], [], []))
    for entity_id, name, address, country in db.execute("SELECT id,name,address,country FROM records"):
        norm_name, norm_address = normalized(name, address)
        ids, names, addresses, raw_names = target[country]
        ids.append(entity_id)
        names.append(norm_name)
        addresses.append(norm_address)
        raw_names.append(name)
    db.close()
    print("loaded", sum(len(x[0]) for x in source.values()), "S1 and",
          sum(len(x[0]) for x in target.values()), "targets in",
          round(time.monotonic()-started, 1), "s", flush=True)
    report = {"scope": "exposed reduced target pool; not full-corpus or France quality",
              "views": ["name_words", "address_words", "name_chars",
                        "phonetic_chars", "joint_words", "address_chars"],
              "countries": {}}
    exported = {} if args.candidate_out else None
    for country, (s1_ids, s1_names, s1_addresses, s1_raw_names) in source.items():
        target_ids, target_names, target_addresses, target_raw_names = target[country]
        target_by_id = {target_id: i for i, target_id in enumerate(target_ids)}
        hits_by_view = []
        specs = (("name_words", target_names, s1_names, "word"),
                 ("address_words", target_addresses, s1_addresses, "word"),
                 ("name_chars", None, None, "char_wb"),
                 ("phonetic_chars", None, None, "char"),
                 ("joint_words", None, None, "word"),
                 ("address_chars", None, None, "char_wb"))
        for view_name, target_docs, query_docs, analyzer in specs:
            if view_name == "name_chars":
                target_docs = [unidecode(value) for value in target_names]
                query_docs = [unidecode(value) for value in s1_names]
            elif view_name == "phonetic_chars":
                target_docs = [phonetic_key(value) for value in target_raw_names]
                query_docs = [phonetic_key(value) for value in s1_raw_names]
            elif view_name == "joint_words":
                target_docs = [name+" "+address for name, address in zip(target_names, target_addresses)]
                query_docs = [name+" "+address for name, address in zip(s1_names, s1_addresses)]
            elif view_name == "address_chars":
                target_docs = [unidecode(value) for value in target_addresses]
                query_docs = [unidecode(value) for value in s1_addresses]
            options = dict(analyzer=analyzer,
                           ngram_range=(1, 2) if analyzer == "word" else
                                       ((2, 4) if analyzer == "char" else (3, 5)),
                           lowercase=False, dtype=np.float32, sublinear_tf=True)
            if analyzer == "word":
                options["token_pattern"] = r"(?u)\b\w+\b"
            else:
                options["min_df"] = 2
            vectorizer = TfidfVectorizer(**options)
            matrix = vectorizer.fit_transform(target_docs).tocsr()
            right = matrix.T.tocsr()
            query_matrix = vectorizer.transform(query_docs).tocsr()
            hits = []
            for offset in range(0, len(s1_ids), 500):
                stop = min(offset+500, len(s1_ids))
                scored = sp_matmul_topn(query_matrix[offset:stop], right,
                                        top_n=RANKS[-1], threshold=0.0, sort=True,
                                        n_threads=args.workers).tocsr()
                hits.extend((scored.indices[scored.indptr[j]:scored.indptr[j+1]].copy(),
                             scored.data[scored.indptr[j]:scored.indptr[j+1]].copy())
                            for j in range(stop-offset))
            hits_by_view.append(hits)
            print(country, view_name, "complete", len(s1_ids), "S1 in",
                  round(time.monotonic()-started, 1), "s", flush=True)
            del vectorizer, matrix, right, query_matrix
            if view_name in ("name_chars", "phonetic_chars", "joint_words", "address_chars"):
                del target_docs, query_docs
        by_rank = {}
        missed_at_max = []
        for rank in RANKS:
            counts = []
            retrieved = 0
            positives = 0
            for row, s1_id in enumerate(s1_ids):
                picked = set()
                for view in hits_by_view:
                    picked.update(view[row][0][:rank])
                expected = truth[s1_id]
                positives += len(expected)
                retrieved += sum(target_by_id[target_id] in picked for target_id in expected)
                if rank == RANKS[-1]:
                    missed_at_max.extend({"source1_id": s1_id, "target_id": target_id,
                                          "source1_name": s1_names[row],
                                          "source1_address": s1_addresses[row],
                                          "target_name": target_names[target_by_id[target_id]],
                                          "target_address": target_addresses[target_by_id[target_id]]}
                                         for target_id in expected if target_by_id[target_id] not in picked)
                counts.append(len(picked))
            by_rank[str(rank)] = {"candidate_pairs": sum(counts),
                                  "candidate_mean": float(np.mean(counts)),
                                  "candidate_p95": float(np.percentile(counts, 95)),
                                  "candidate_max": max(counts),
                                  "true_links": positives,
                                  "retrieved_links": retrieved,
                                  "blocking_recall": retrieved/positives}
        if exported is not None:
            for row, s1_id in enumerate(s1_ids):
                picked = set()
                for view in hits_by_view:
                    picked.update(view[row][0][:args.export_rank])
                exported[s1_id] = sorted(target_ids[index] for index in picked)
        report["countries"][country] = {"source1_rows": len(s1_ids),
                                        "target_rows": len(target_ids),
                                        "by_rank": by_rank,
                                        "missed_at_max_rank": missed_at_max}
        selection = {method: {str(cap): 0 for cap in CAPS}
                     for method in ("max_cosine", "sum_cosine", "sum_reciprocal_rank")}
        for row, s1_id in enumerate(s1_ids):
            candidates = {}
            for view, hits in enumerate(hits_by_view):
                indices, scores = hits[row]
                for rank, (target_index, value) in enumerate(zip(indices, scores), 1):
                    item = candidates.setdefault(int(target_index), [0.0, 0.0, 0.0])
                    item[0] = max(item[0], float(value))
                    item[1] += float(value)
                    item[2] += 1.0/rank
            expected = {target_by_id[target_id] for target_id in truth[s1_id]}
            for column, method in enumerate(selection):
                ordered = sorted(candidates, key=lambda target_index:
                                 (-candidates[target_index][column], target_index))
                for cap in CAPS:
                    selection[method][str(cap)] += len(expected.intersection(ordered[:cap]))
        positives = by_rank[str(RANKS[-1])]["true_links"]
        report["countries"][country]["by_cap"] = {
            method: {cap: {"retrieved_links": count,
                           "blocking_recall": count/positives}
                     for cap, count in values.items()}
            for method, values in selection.items()}
        del hits_by_view, target_by_id
    report["seconds"] = time.monotonic()-started
    if exported is not None:
        args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
        with args.candidate_out.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("source1_entity_id", "candidate_entity_ids"))
            for row in rows(args.source1):
                writer.writerow((row["entity_id"], ",".join(exported[row["entity_id"]])))
        with args.candidate_out.open("rb") as stream:
            report["candidate_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        report["candidate_out"] = str(args.candidate_out)
        report["export_rank"] = args.export_rank
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
