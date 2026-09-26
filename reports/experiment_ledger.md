# Entity-resolution experiment ledger

Last updated: 2026-09-27 05:15 IST. This is a working checkpoint, not a submission claim.

## Guardrails

- Goal: honestly measured per-S1 macro F0.5 at least 99.5%; do not claim the goal from a reduced target pool alone.
- Keep each selection, development, fresh validation, and final confirmation cohort distinct. `work/fresh_10k_v1` is exposed; `work/fresh_10k_v2` and `work/fresh_10k_final` remain sealed.
- The 442,904-target v1 pool is positive-enriched and omits most full-corpus distractors. No France or Portal score exists.
- Do not start the stopped broad full-corpus scan or complete test TSV generation before the quality gate and phase review.
- Total Modal spending ceiling: $5 until explicitly changed. The 2026-09-27 05:12 IST billing report showed approximately $1.164 posted across all stopped apps; current-hour billing can lag. No Modal app is running.
- Account-wide Codex weekly usage was 33% at the last check. Preserve more than 40% remaining; stop new model-driven work at 60% used. Check before major continuation; do not intentionally overshoot a lagging meter.

## Completed quality work

1. Frozen model on exposed old 10k development cohort: 98.3658% overall, 98.1085% India, 98.5386% US. Candidate oracle 99.8794%; 92.6% of entity-score loss is after retrieval. Exact raw mixed-label duplicates: zero.
2. Simple context-threshold tuning, missing-address expert, extra training, deeper or alternate tree families, and target-neighbor address borrowing did not yield a material development gain. A 20% residual XGBoost blend gave a small development gain to 98.3888% overall.
3. First fresh v1 10k batch (original training S1 rows 280001–290000): frozen baseline 98.4139% overall / 98.2916% India / 98.4970% US. Residual blend 98.4989% / 98.4338% / 98.5432%. The gain is 0.0851 percentage points overall, still about 1.001 points below goal. Same 503,552 candidates for both methods locally. Details: [pipeline_bottleneck_diagnosis.md](pipeline_bottleneck_diagnosis.md), `work/fresh_10k_v1/paired_score.json`.
4. Script-aware Unidecode/RapidFuzz name feature specialist on old development data scored 98.3847% overall at its best exposed-development threshold versus 98.3658% baseline, and 98.1801% India. This is a tiny, selection-optimistic development signal; no fresh validation or promotion.

## Performance profiling, before any production optimization

- Full test target indexing: 9,969,589 targets, 1,608.3 seconds, 3.21 GB main SQLite index + 1.18 GB extra SQLite index, 458 MB reported peak RSS. This is a one-time reusable stage.
- Earlier 1.49-million-pair base-feature generation: 126.3 seconds; XGBoost/logistic training: 35.4 seconds. Training is not the hours-long bottleneck.
- Read-only frozen inference profile on 500 fresh v1 S1 rows against 442,904 targets, single worker: 239.12 seconds total. Baseline retrieval 106.47s, rescue retrieval 120.49s, feature calculation plus saved-model scoring 12.14s, in-memory TSV serialization 0.006s. Retrieval is 94.9% of time. 25,486 selected pairs from 92,188 stage-1 unique hits and 38,816 stage-2 unique hits; exact counts in `work/fresh_10k_v1/inference_profile_500.json`.
- `cProfile` on 100 rows: 56.5 profiled seconds, 41.4 seconds inside `sqlite3.Connection.execute`. SQL categorization without profiler: 988 FTS searches took 37.55 seconds, 50,261 vocabulary lookups 3.56 seconds, 406 record fetches 0.05 seconds. 100-row end-to-end total 47.74 seconds. Query/result ranking is currently the dominant computational cost. Details in `work/fresh_10k_v1/inference_profile_100_sql.json` and `inference_cprofile_100.txt`.
- Route exclusive true-link counts on the 100-row sample: address14, name1, joint1, char1, phonetic1, phonetic_joint0. This small sample cannot justify dropping a route; blocking recall must be checked on a larger cohort before changing route rules.
- SQLite mmap/cache benchmark with unchanged FTS queries and exact candidate hash: on 100 reduced-pool rows, default 46.30s then 43.36s, tuned 42.70s then 42.36s. On five real test rows against the **full 9,969,589-target index**, default 58.93s cold and 48.33s warm versus tuned 56.14s and 52.66s. Candidate SHA matched in all passes; tuned memory peak reached 1.66 GB versus 469.5 MB before tuning. The warm default won. **Reject** this setting; no production patch. A 50-row × four-setting full-index test was stopped because it was excessively slow and memory-intensive, without a completed report.
- Added the backward-compatible `infer.py score-cached` path after measuring retrieval cost. A 100-S1 pilot reused the candidate TSV and reproduced both existing output TSVs byte-for-byte in 3.38s, versus 47.74s for the unmodified single-worker inference profile. The full 10k verification completed in **227.87s versus 1,451.94s** for retrieval plus scoring: **6.37× faster matcher-only iteration**, with identical full candidate and matching TSV SHA-256 hashes and unchanged quality. This does not speed the initial retrieval pass. It records input candidate and model/index hashes; model verification and candidate/source-order checks remain.
- Constructed FTS query expressions for 1,000 fresh S1 rows without executing FTS. Only 34 of 7,919 route expressions repeated (0.4%); per-query FTS result caching would save little and is rejected.

## Cloud v1 replication completed

- Modal app `ap-FyAP5yf4CBPxFZeLI8oeoo`, CPU4/RAM12GiB/timeout3600s, unlabeled `source1.tsv` and same bounded target index. No truth mounted.
- Cloud baseline completed all 10k in 2,019.3s, 503,552 candidate pairs. Its `matching_results.tsv` hash **matches local exactly** (`369e24f8...`). Experimental paired rescore completed in 310.1s and its match TSV also exactly matches local (`f76f2dbb...`). Modal app completed and stopped. Cloud setup/extraction was 2.7s; total function computation roughly 2,350s, excluding image build/upload/download overhead.
- Candidate TSVs are not byte-identical: 9,999 of 10,000 rows match exactly; one India row swaps one negative candidate for another, leaving candidate count and every known positive unchanged. This may be an SQLite FTS ranking tie across platforms, but the exact cause is not established. `work/fresh_10k_v1/cloud_10k/candidate_diff.json` records the count and ID. **Do not claim exact candidate parity.**
- Downloaded cloud predictions and candidates were independently scored locally against v1 truth using the unchanged scorer. Cloud baseline and blend scores are **exactly the same** as local: 98.4139% / 98.4989% overall, 98.2916% / 98.4338% India, 98.4970% / 98.5432% US. Candidate recall, p95, maximum, and country counts are also identical. See `work/fresh_10k_v1/cloud_10k/paired_score.json`.

## Bounded batched retrieval architecture screen

The first reverse TF-IDF retrieval screen reused the full 2,206,821-record training S1 index and queried only the same 442,904-target fresh-v1 reduced pool. Country-specific workers retained 49,930 pairs for the selected 10,000 S1 rows (mean 4.99; p95 10). India finished in 613.1 seconds and US in 673.9 seconds concurrently, after roughly 443 seconds of index construction per worker. Truth was never mounted in Modal. This is an architecture screen on an exposed validation batch, not a fresh validation result or a France/full-corpus measurement.

The default `keep=4,max_rank=40` setting retrieved 33,520 of 34,471 true links: **97.2412%** raw blocking recall overall, 95.6955% India, 98.2934% US. Its candidate oracle macro F0.5 was 99.1642% overall and only 98.6714% India. Scoring the top-16 capped pairs with the frozen matcher gave **97.7186% final macro F0.5** overall, 97.1766% India, 98.0869% US. The original frozen FTS pipeline scored 98.4139% overall on the same batch, with 99.7331% blocking recall. Therefore the fast default v2 cannot replace FTS. Full results: `work/v2_bounded_v1/score.json`.

Of v2's 951 missed true links, the original FTS candidates recovered 872. The misses were spread across low- and high-candidate-count S1 groups. A retrospective low-top-score fallback rule would have to rerun FTS on about **93.7% of S1 rows** to reach 99.5% union blocking recall on this exposed batch. A zero-candidate-only FTS fallback cannot fix this architecture. That diagnostic is selection-optimistic and does not validate a production hybrid.

The predeclared wider variant (`keep=16,max_rank=80`) completed on the same exposed batch. It retained 71,532 pairs, recovered 33,792 links, and reached **98.0302%** blocking recall overall / **96.7197%** India. Its perfect-decision macro F0.5 ceiling was only **99.3857%** overall / **98.9472%** India. It therefore also fails as a direct replacement; matcher scoring was skipped because even a perfect matcher could not reach the 99.5% overall goal. Both cloud apps stopped. Details: [batched_retrieval_architecture_screen.md](batched_retrieval_architecture_screen.md).

## Next evidence-based steps

1. If platform-exact candidate TSVs are required, inspect the one SQLite FTS disagreement using the saved S1 ID and both candidate IDs; in the current bounded validation it does not change candidate recall or final F0.5.
2. Retain the default SQLite connection settings. The completed full-index five-row benchmark found the larger-cache/mmap setting slower on the warm repeat and much heavier in memory; it is not a production optimization.
3. The first five full-index test rows required about 10–12s/S1 in one process, a severe throughput warning. A broader but bounded full-index benchmark should follow only a candidate architecture change with progress and time limits; no TSV submission has started.
4. Keep the validated FTS retrieval for now. A new batched design must preserve its true-link coverage before replacement, followed by fresh disjoint validation; neither screened configuration qualifies.
5. Pursue new high-specificity evidence for empty target addresses and script mismatches. Reject simple threshold loosening, which added too many false positives. Choose any new method on development, then freeze before fresh v2. Reserve fresh final batch for confirmation, not iterative tuning.

Do not label an experiment a win merely because it reduces wall time; report candidate counts, blocking recall, final matching F0.5, memory, and scope together.
