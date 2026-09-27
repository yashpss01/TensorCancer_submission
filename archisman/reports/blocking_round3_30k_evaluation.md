# Frozen blocking evaluation: next 30,000 S1 records

**Improved recall: 99.66938% overall and 99.46624% India.** The >=99.5% overall/India criterion was not met. The baseline was run on the identical target pool.

| Metric | Original baseline | Frozen improvement |
|---|---:|---:|
| S1 count | 30,000 | 30,000 |
| Target count | 409,141 | 409,141 |
| Comparison space | 12,274,230,000 | 12,274,230,000 |
| TP | 103,245 | 103,400 |
| FN | 498 | 343 |
| FP | 1,162,250 | 1,166,920 |
| TN | 12,272,964,007 | 12,272,959,337 |
| Recall | 99.51997% | 99.66938% |
| Precision | 8.15847% | 8.13968% |
| Reduction ratio | 99.98969% | 99.98965% |
| Specificity | 99.99053% | 99.99049% |
| Candidate F1 | 0.150807 | 0.150503 |
| Candidate pairs | 1,265,495 | 1,270,320 |
| Candidates/S1 mean | 42.183167 | 42.344000 |
| Candidates/S1 p95 | 73 | 73 |
| Candidates/S1 max | 93 | 96 |
| Oracle macro F0.5 ceiling | 99.84614% | 99.90022% |
| True singletons | 1,671 | 1,671 |
| S1 with at least one miss | 448 | 318 |

## India and US

### India

| Metric | Original baseline | Frozen improvement |
|---|---:|---:|
| S1 count | 12,066 | 12,066 |
| Target count | 409,141 | 409,141 |
| Comparison space | 4,936,695,306 | 4,936,695,306 |
| TP | 41,414 | 41,556 |
| FN | 365 | 223 |
| FP | 587,639 | 591,577 |
| TN | 4,936,065,888 | 4,936,061,950 |
| Recall | 99.12636% | 99.46624% |
| Precision | 6.58355% | 6.56355% |
| Reduction ratio | 99.98726% | 99.98717% |
| Specificity | 99.98810% | 99.98802% |
| Candidate F1 | 0.123471 | 0.123145 |
| Candidate pairs | 629,053 | 633,133 |
| Candidates/S1 mean | 52.134344 | 52.472485 |
| Candidates/S1 p95 | 76 | 76 |
| Candidates/S1 max | 93 | 96 |
| Oracle macro F0.5 ceiling | 99.71019% | 99.82909% |
| True singletons | 669 | 669 |
| S1 with at least one miss | 316 | 199 |

### US

| Metric | Original baseline | Frozen improvement |
|---|---:|---:|
| S1 count | 17,934 | 17,934 |
| Target count | 409,141 | 409,141 |
| Comparison space | 7,337,534,694 | 7,337,534,694 |
| TP | 61,831 | 61,844 |
| FN | 133 | 120 |
| FP | 574,611 | 575,343 |
| TN | 7,336,898,119 | 7,336,897,387 |
| Recall | 99.78536% | 99.80634% |
| Precision | 9.71510% | 9.70578% |
| Reduction ratio | 99.99133% | 99.99132% |
| Specificity | 99.99217% | 99.99216% |
| Candidate F1 | 0.177063 | 0.176912 |
| Candidate pairs | 636,442 | 637,187 |
| Candidates/S1 mean | 35.488012 | 35.529553 |
| Candidates/S1 p95 | 69 | 69 |
| Candidates/S1 max | 90 | 90 |
| Oracle macro F0.5 ceiling | 99.93760% | 99.94808% |
| True singletons | 1,002 | 1,002 |
| S1 with at least one miss | 132 | 119 |

## Paired change

The improved set recovered 155 additional true pairs and lost 0 baseline true pairs. It added 4,825 candidates (0.1608 per S1, maximum 4), including 4,670 false candidates. Its per-S1 set is checked as a superset of the original baseline.

## Population and audit

Following the user's stop instruction, the evaluated batch is original S1 file rows 20,001–50,000 inclusive: exactly 30,000 references with 103,743 known true links. The pool had already been built for an originally planned 60k run and contains all 207,489 labeled targets for rows 20,001–80,000 plus 201,652 deterministic 2% distractors: 409,141 total. Targets labeled to the unevaluated second half remain in the common pool as nontrue candidates for these 30k references. Both pipelines searched this same pool. The paired comparison space is 30,000 × 409,141 = 12,274,230,000 pairs. The complete S2/S3 corpus has 10,320,219 targets.

The full truth ownership audit found zero shared labeled targets with any other S1 group, including prior batches. S1 IDs do not overlap. The first 20,000 S1 rows and the originally planned 60k batch share 0 exact normalized name/address/country signatures. This exact check does not resolve all near-duplicate or labeling ambiguities.

The pool includes every known positive but omits most full-corpus negatives, so absolute performance may be optimistic relative to all 10.32 million targets. The 30k result is tied to this 409,141-target pool and should not be compared directly with the earlier 10k pools. Consecutive file rows may not represent unseen data. France accuracy remains unmeasured. Country values were never used to filter retrieval. These are blocking candidate metrics, not final matching precision, F1, F0.5, or leaderboard scores.

## Frozen method and validation

The round-2 source and settings were reused without modification: rescue score threshold 0.70, at most 4 additions per S1. Original code/configuration hashes were checked before preparation and each retrieval. The new query, truth, index, extra index, and manifest hashes were sealed at 2026-09-26T11:02:44Z before running either pipeline. No selection or tuning used this batch.

Labels were used only to build and audit the pool and to score the completed slice; IDs identify rows and seed the deterministic sample. Both exported TSVs were reopened and checked row by row against the exact frozen selected sets and target IDs. All known positives for these 30k references are in the pool, the two scored runs contain exactly 30,000 ordered references, and every original candidate is preserved. The raw retrieval JSONL files retain some unscored rows written during shutdown. Validation status: PASS.

## Metric definitions

For each group, comparison space = S1 count × entire target pool, including cross-country pairs. TP/FN are retained/missed true links; FP counts nontrue candidates; TN is comparison space minus TP, FN, and FP. Recall = TP/(TP+FN); precision = TP/(TP+FP); reduction ratio = 1−(TP+FP)/comparison space; specificity = TN/(TN+FP); candidate F1 = 2TP/(2TP+FP+FN). p95 uses the nearest-rank percentile.

The oracle macro F0.5 ceiling assumes a perfect downstream classifier discards every false candidate. It averages per-S1 F0.5 and gives true singletons a score of one when predicting an empty set. It is an upper bound, not an achieved classifier score.

## Runtime and memory

Pool construction and ownership audit: 56.76s, parent peak RSS 183.1 MB. Extra index construction: 32.45s, parent peak RSS 77.9 MB.

- baseline: 2012.6s to the 30,000th result, measured from the progress log; 3 workers. Retrieval RSS was not captured because the processes were interrupted at the requested stopping point.
- improved: 1819.9s to the 30,000th result, measured from the progress log; 6 workers. Retrieval RSS was not captured because the processes were interrupted at the requested stopping point.

The two retrieval processes overlapped in wall time, so their durations should not be added to estimate elapsed time. Input preparation, indexing, scoring, and report generation are separate.

## Artifacts

Reproduction commands: `code/business_entity_resolution/round3/README.md`. Frozen configuration: `artifacts/blocking_round2/frozen_config.json`; new input seal: `artifacts/blocking_round3/sealed.json`. Candidate TSVs and metrics: `artifacts/blocking_round3/fresh/baseline/` and `artifacts/blocking_round3/fresh/improved/`. Prior round artifacts remain unchanged.
