"""Paired experimental rescore of an already retrieved fresh candidate file.

This does not retrieve new candidates, tune on fresh labels, or change the
submission's frozen inference code. It checks frozen-score parity on every
entity before writing the alternative predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
from xgboost import XGBClassifier


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def chunks(iterable, size):
    while batch := list(itertools.islice(iterable, size)):
        yield batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--baseline-tsv", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--model-dir", type=pathlib.Path, required=True)
    p.add_argument("--residual-model", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--limit", type=int, help="Only for parity smoke tests on a prefix")
    args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    code = pathlib.Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"
    sys.path[:0] = [str(code), str(pathlib.Path(__file__).resolve().parent)]
    import infer
    from features import FEATURE_NAMES, features, represent
    from inference_features import EXTRA_NAMES, GROUP_NAMES, extra_vector, group_matrix
    from residual_rerank_screen import make_features

    first, group, augmented, (nw, aw, default), (cn, ca, cdefault) = infer.load_models(args.model_dir)
    residual = XGBClassifier()
    residual.load_model(args.residual_model)
    db = sqlite3.connect(f"file:{args.index_dir / 'index.sqlite'}?mode=ro", uri=True)
    q_iter = read_tsv(args.source1)
    c_iter = read_tsv(args.candidate_tsv)
    b_iter = read_tsv(args.baseline_tsv)
    out = args.out_dir / "matching_results.tsv"
    started = time.monotonic()
    total = predicted = 0
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "matched_entity_ids"))
        source_rows = itertools.islice(q_iter, args.limit) if args.limit else q_iter
        for query_rows in chunks(source_rows, args.batch_size):
            candidate_rows = list(itertools.islice(c_iter, len(query_rows)))
            baseline_rows = list(itertools.islice(b_iter, len(query_rows)))
            if len(candidate_rows) != len(query_rows) or len(baseline_rows) != len(query_rows):
                raise ValueError("Source/candidate/baseline coverage mismatch")
            for q, c, b in zip(query_rows, candidate_rows, baseline_rows):
                assert q["entity_id"] == c["source1_entity_id"] == b["source1_entity_id"]
            mids_by_query = [r["candidate_entity_ids"].split(",") if r["candidate_entity_ids"] else []
                             for r in candidate_rows]
            all_ids = list(itertools.chain.from_iterable(mids_by_query))
            records = infer.target_records(db, all_ids)
            X, E, reps, offsets = [], [], [], [0]
            for q, mids in zip(query_rows, mids_by_query):
                qr = represent(q["business_name"], q["business_address"], q["country"])
                current = []
                for mid in mids:
                    tr = represent(*records[mid])
                    X.append(features(qr, tr, nw, aw, default))
                    E.append(extra_vector(qr, tr, cn, ca, cdefault))
                    current.append(tr)
                reps.append(current)
                offsets.append(len(X))
            if not X:
                for q, b in zip(query_rows, baseline_rows):
                    assert not b["matched_entity_ids"]
                    writer.writerow((q["entity_id"], ""))
                total += len(query_rows)
                continue
            X = np.asarray(X, dtype=np.float32)
            E = np.asarray(E, dtype=np.float32)
            assert X.shape[1] == len(FEATURE_NAMES) and E.shape[1] == len(EXTRA_NAMES)
            pair = np.concatenate((X, E), axis=1)
            first_prob = first.predict_proba(X)[:, 1].astype(np.float32)
            A = np.concatenate([
                group_matrix(q, rep, first_prob[offsets[i]:offsets[i+1]], aw, default)
                for i, (q, rep) in enumerate(zip(query_rows, reps))
            ], axis=0)
            assert A.shape == (len(X), len(GROUP_NAMES))
            group_prob = group.predict_proba(np.concatenate((pair, A[:, 1:]), axis=1))[:, 1]
            augmented_prob = augmented.predict_proba(pair)[:, 1]
            baseline_prob = (0.4 * group_prob + 0.6 * augmented_prob).astype(np.float32)
            residual_prob = residual.predict_proba(make_features(X, E, A, group_prob, augmented_prob))[:, 1]
            changed_prob = 0.8 * baseline_prob + 0.2 * residual_prob
            for i, (q, b, mids) in enumerate(zip(query_rows, baseline_rows, mids_by_query)):
                lo, hi = offsets[i:i+2]
                baseline = [mid for mid, prob in zip(mids, baseline_prob[lo:hi]) if prob >= 0.74]
                expected = b["matched_entity_ids"].split(",") if b["matched_entity_ids"] else []
                if baseline != expected:
                    raise AssertionError(f"Frozen scorer parity failed for {q['entity_id']}")
                changed = [mid for mid, prob in zip(mids, changed_prob[lo:hi]) if prob >= 0.74]
                writer.writerow((q["entity_id"], ",".join(changed)))
                total += 1
                predicted += len(changed)
            print(f"rescored {total:,} S1 rows in {time.monotonic()-started:.1f}s", flush=True)
    if next(c_iter, None) is not None or next(b_iter, None) is not None:
        raise ValueError("Candidate or baseline file has extra rows")
    db.close()
    manifest = {
        "rows": total,
        "predicted_pairs": predicted,
        "seconds": time.monotonic() - started,
        "candidate_sha256": digest(args.candidate_tsv),
        "baseline_sha256": digest(args.baseline_tsv),
        "residual_model_sha256": digest(args.residual_model),
        "frozen_score_parity": "passed every S1 row",
        "decision": "0.8 * frozen_probability + 0.2 * residual_probability >= 0.74",
        "scope": "experimental, fresh 10k, reduced target pool",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
