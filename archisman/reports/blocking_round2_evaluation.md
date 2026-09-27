# Round 2: paired blocking evaluation on the next 10,000 references

**Fresh overall recall: 99.79554%; India: 99.66265%.** The >=99.5% target was met. The original baseline was evaluated on the same 239,812-target pool. No fresh-result tuning followed.

| Metric | Original baseline | Improved |
|---|---:|---:|
| S1 count | 10,000 | 10,000 |
| Target count | 239,812 | 239,812 |
| Comparison space | 2,398,120,000 | 2,398,120,000 |
| TP | 34,621 | 34,655 |
| FN | 105 | 71 |
| FP | 347,106 | 348,091 |
| TN | 2,397,738,168 | 2,397,737,183 |
| Recall | 99.69763% | 99.79554% |
| FN rate | 0.30237% | 0.20446% |
| Precision | 9.06957% | 9.05431% |
| Reduction ratio | 99.98408% | 99.98404% |
| Specificity | 99.98553% | 99.98548% |
| Candidate F1 | 0.166266 | 0.166023 |
| Candidate pairs | 381,727 | 382,746 |
| Candidates/S1 mean | 38.172700 | 38.274600 |
| Candidates/S1 p95 | 72 | 72 |
| Candidates/S1 max | 92 | 93 |
| Oracle macro F0.5 ceiling | 99.91751% | 99.94605% |
| True singletons | 553 | 553 |
| S1 with ≥1 miss | 96 | 67 |

## Country-level paired results

### India

| Metric | Original baseline | Improved |
|---|---:|---:|
| S1 count | 4,012 | 4,012 |
| Target count | 239,812 | 239,812 |
| Comparison space | 962,125,744 | 962,125,744 |
| TP | 13,854 | 13,885 |
| FN | 78 | 47 |
| FP | 179,795 | 180,652 |
| TN | 961,932,017 | 961,931,160 |
| Recall | 99.44014% | 99.66265% |
| FN rate | 0.55986% | 0.33735% |
| Precision | 7.15418% | 7.13746% |
| Reduction ratio | 99.97987% | 99.97978% |
| Specificity | 99.98131% | 99.98122% |
| Candidate F1 | 0.133480 | 0.133209 |
| Candidate pairs | 193,649 | 194,537 |
| Candidates/S1 mean | 48.267448 | 48.488784 |
| Candidates/S1 p95 | 75 | 75 |
| Candidates/S1 max | 92 | 93 |
| Oracle macro F0.5 ceiling | 99.84183% | 99.90726% |
| True singletons | 216 | 216 |
| S1 with ≥1 miss | 69 | 43 |

### US

| Metric | Original baseline | Improved |
|---|---:|---:|
| S1 count | 5,988 | 5,988 |
| Target count | 239,812 | 239,812 |
| Comparison space | 1,435,994,256 | 1,435,994,256 |
| TP | 20,767 | 20,770 |
| FN | 27 | 24 |
| FP | 167,311 | 167,439 |
| TN | 1,435,806,151 | 1,435,806,023 |
| Recall | 99.87015% | 99.88458% |
| FN rate | 0.12985% | 0.11542% |
| Precision | 11.04170% | 11.03560% |
| Reduction ratio | 99.98690% | 99.98689% |
| Specificity | 99.98835% | 99.98834% |
| Candidate F1 | 0.198849 | 0.198753 |
| Candidate pairs | 188,078 | 188,209 |
| Candidates/S1 mean | 31.409152 | 31.431029 |
| Candidates/S1 p95 | 68 | 68 |
| Candidates/S1 max | 78 | 79 |
| Oracle macro F0.5 ceiling | 99.96821% | 99.97204% |
| True singletons | 337 | 337 |
| S1 with ≥1 miss | 27 | 24 |

## Paired change

The improvement recovered 34 true pairs and lost zero baseline pairs. It added 1,019 candidate pairs (0.102/S1 on average; maximum 4), including 985 additional false candidates. Superset preservation is checked for every reference; it guarantees paired recall cannot decline, while candidate cost can increase.

## Development and freeze

All first 10,000 records—including the former inspected holdout—were development data. The chosen rescue threshold is 0.70, with at most 4 added candidates per reference. Development selection examined caps 4/8/12/16/24 and cutoffs 0.50–0.80, seeking the smallest mean set with >=99.70% India and overall recall. The complete grid and chosen metrics are saved. Code, baseline configuration, fresh queries, both indices, and the manifest were hashed before fresh evaluation (2026-09-26T10:03:05Z).

The original baseline source and configuration remain unchanged. New routes use coarse phonetic bigrams with address context; conventional address abbreviations, ordinal and leading-zero normalization; and up to three strongly linked text-derived anchors. Normalization retains original strings in the records. Extra retrieval is capped at 40 normalized-address results, 40 phonetic/address results, and 20 per anchor. Existing filtered baseline retrieval records may also be reconsidered. The final rescue cap governs the actual extra matcher inputs; raw retrieval is not the reported candidate set.

## Evaluation population and leakage audit

Fresh queries are original training S1 rows 10,001–20,000. All 34,726 labeled targets are included alongside 205,086 deterministic distractors drawn with the same 2% hash rule as round 1. Both pipelines use the identical 239,812-target pool. Labels only construct/audit the benchmark and score outputs; IDs only identify records and reproducible samples. Neither is a retrieval feature.

A full truth audit found no overlapping S1 IDs or labeled target ownership between the two batches or another S1 group. No exact normalized name/address/country signature was shared between batches. Shared unlabelled distractors are deliberate. This audit does not independently establish the absence of every possible near-duplicate beyond supplied truth.

The pool is enriched for positives and omits most distractors, so results are optimistic relative to all 10,320,219 targets. Consecutive-row sampling can be biased. France remains untested; all country labels remain eligible, but compatibility is not measured France accuracy. No final classifier, final matching score, or leaderboard result is claimed.

## Metric definitions

For each group, N = S1 count × full fresh pool target count (including cross-country pairs). TP/FN count retained/missed true links; FP counts nontrue candidates; TN = N−TP−FN−FP. Recall=TP/(TP+FN); precision=TP/(TP+FP); reduction ratio=1−(TP+FP)/N; specificity=TN/(TN+FP); candidate F1=2TP/(2TP+FP+FN). p95 uses the nearest-rank percentile.

The oracle discards all false candidates. A non-singleton with h retained true links of t scores 1.25h/(h+0.25t), or zero if h=0. Singletons receive an empty set and score one. Averaging these per-S1 scores yields a ceiling, not achieved classifier performance.

## Runtime and memory

Fresh pool preparation (ownership audit plus streaming pool/index construction): 45.73s, peak process RSS 80.6 MB. Extra index: 19.42s, peak process RSS 74.7 MB.

- baseline: 394.44s retrieval wall time; parent peak RSS 42.4 MB; maximum child peak RSS 165.6 MB; three workers.
- improved: 465.01s retrieval wall time; parent peak RSS 440.2 MB; maximum child peak RSS 141.4 MB; three workers.

The improved pipeline reuses baseline retrieval and costs 859.44s combined retrieval wall time, not merely the incremental rescue time. Input JSON loading precedes the retrieval timer (baseline 0.01s; rescue 0.82s) and is excluded from that sum; index construction, scoring, validation, and report generation are separate. RSS is macOS resource usage in bytes; parent and maximum child peaks are not aggregate concurrent memory. These measurements do not establish full-corpus scalability.

## Artifacts and checks

Runnable commands: `code/business_entity_resolution/round2/README.md`. Frozen configuration and input hashes: `artifacts/blocking_round2/frozen_config.json`. Fresh candidate TSVs and complete metrics/missed-pair text: `artifacts/blocking_round2/fresh/baseline/` and `.../improved/`. The reports use exactly the final selected sets. Original first-round artifacts remain under `artifacts/blocking/`.

The organizer validator helper checked all 10,000 rows for both candidate sets against the exact pool IDs. Additional checks verify positive inclusion, identical query sets, original-candidate preservation, exact exported sets, and frozen code/data hashes. Unit tests cover metric arithmetic, singleton oracle behavior, score/cap/rescue selection, FTS ranking, Unicode/number normalization, and original-code preservation. These are local evaluation outputs, not competition test submissions.

## Remaining errors

There are 71 remaining missed true pairs: 36 were absent from the expanded retrieval and 35 were filtered by final selection. 29 have empty target addresses and 15 have Indic-script target names (overlapping categories). Original text for every miss is saved for diagnosis. These fresh outcomes were inspected only after freezing, and were not used for further tuning.

Observed examples include a common name with no address (Surgical Group → surgical group Enterprises, ranked sixth and outside the four-addition cap); a substantially changed name and no address (Global Institute → GLOBAL CENTER, absent from retrieval); and a random-looking trade alias with only partial city/number evidence (Vidhi Medical Centre Clinic → Vioaviavi, retrieved but below the score cutoff). Cross-script names with very short or changed addresses also remain. These are diagnoses only; thresholds and routes were not revised using them.
