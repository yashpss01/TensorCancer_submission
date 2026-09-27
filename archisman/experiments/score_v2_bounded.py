"""Local, labeled evaluation of the bounded v2 candidate architecture.

Retrieval and frozen-model prediction finish before this script opens truth.
Scores the raw v2 candidate ceiling, the fixed top-16 cap, and final per-S1
macro F0.5. This is a reduced-pool architecture screen, not a Portal result.
"""

import argparse
import csv
import json
import pathlib
import sqlite3
import statistics

import polars as pl


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def ids(cell):
    return cell.split(",") if cell else []


def percentile(values):
    ordered = sorted(values)
    return ordered[(95 * len(ordered) + 99) // 100 - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--truth", type=pathlib.Path, required=True)
    p.add_argument("--candidate-root", type=pathlib.Path, required=True)
    p.add_argument("--prediction-tsv", type=pathlib.Path, required=True)
    p.add_argument("--capped-candidate-tsv", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    source = read_rows(a.source1)
    truth = read_rows(a.truth)
    predictions = read_rows(a.prediction_tsv)
    capped = read_rows(a.capped_candidate_tsv)
    if not all(len(x) == len(source) == 10_000 for x in (truth, predictions, capped)):
        raise ValueError("Expected complete, aligned fresh 10k outputs")
    paths = sorted(a.candidate_root.glob("*/candidates-*.parquet"))
    if not paths:
        raise FileNotFoundError("Missing v2 candidate shards")
    raw = pl.concat([pl.read_parquet(path, columns=["s1_id", "q_id"])
                     for path in paths])
    if raw.unique(["s1_id", "q_id"]).height != raw.height:
        raise ValueError("Duplicate raw v2 candidate pairs")
    selected = {r["entity_id"] for r in source}
    if set(raw["s1_id"].unique().to_list()) - selected:
        raise ValueError("Candidate shard contains unselected S1 ID")
    raw_by_s1 = {}
    for sid, mid in raw.iter_rows():
        raw_by_s1.setdefault(sid, set()).add(mid)
    db = sqlite3.connect(f"file:{a.index_dir/'index.sqlite'}?mode=ro", uri=True)
    target_countries = dict(db.execute("SELECT country,count(*) FROM records GROUP BY country"))
    db.close()
    per = []
    for s, t, m, c in zip(source, truth, predictions, capped):
        sid = s["entity_id"]
        if sid != t["source1_entity_id"] or sid != m["source1_entity_id"] or sid != c["source1_entity_id"]:
            raise ValueError("S1 ID/order mismatch")
        known = set(ids(t["matched_entity_ids"]))
        candidates = raw_by_s1.get(sid, set())
        cap = set(ids(c["candidate_entity_ids"]))
        chosen = set(ids(m["matched_entity_ids"]))
        if not chosen <= cap <= candidates:
            raise ValueError("Predictions or top-16 candidates outside raw v2 set")
        if len(cap) != len(ids(c["candidate_entity_ids"])) or len(chosen) != len(ids(m["matched_entity_ids"])):
            raise ValueError("Duplicate ID in candidate or prediction TSV")
        tp = len(chosen & known)
        fp = len(chosen - known)
        fn = len(known - chosen)
        f05 = (1.0 if not chosen else 0.0) if not known else 1.25 * tp / (1.25 * tp + .25 * fn + fp)
        raw_hit = len(candidates & known)
        cap_hit = len(cap & known)
        oracle = lambda hit: (1.0 if not known else 1.25 * hit / (hit + .25 * len(known)))
        per.append(dict(country=s["country"],truth=len(known),raw_count=len(candidates),
                        raw_hit=raw_hit,raw_oracle=oracle(raw_hit),cap_count=len(cap),
                        cap_hit=cap_hit,cap_oracle=oracle(cap_hit),tp=tp,fp=fp,fn=fn,f05=f05))

    def summary(items):
        total_true = sum(x["truth"] for x in items)
        space = sum(target_countries[x["country"]] for x in items)
        def candidate(kind):
            hits = sum(x[kind + "_hit"] for x in items)
            n = sum(x[kind + "_count"] for x in items)
            sizes = [x[kind + "_count"] for x in items]
            fp = n - hits
            fn = total_true - hits
            tn = space - hits - fp - fn
            return dict(comparison_space=space,tp=hits,fn=fn,fp=fp,tn=tn,
                        recall=hits/total_true,precision=hits/n if n else 0.,
                        reduction_ratio=1-n/space,specificity=tn/(tn+fp),
                        candidate_f1=2*hits/(2*hits+fp+fn),
                        oracle_macro_f05=statistics.mean(x[kind+"_oracle"] for x in items),
                        candidate_mean=statistics.mean(sizes),candidate_p95=percentile(sizes),
                        candidate_max=max(sizes))
        return dict(s1_count=len(items),true_links=total_true,
                    raw=candidate("raw"),top16=candidate("cap"),
                    final_macro_f05=statistics.mean(x["f05"] for x in items),
                    final_tp=sum(x["tp"] for x in items),
                    final_fp=sum(x["fp"] for x in items),
                    final_fn=sum(x["fn"] for x in items))

    report = dict(scope="fresh v1 exposed 10000 training S1; full training S1 rank competition; 442904 target reduced pool; country-partitioned v2 retrieval; frozen original matcher; not France or Portal",
                  raw_candidate_pairs=raw.height,target_counts=target_countries,
                  overall=summary(per),countries={country:summary([x for x in per if x["country"]==country])
                                                 for country in ("India", "US")})
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
