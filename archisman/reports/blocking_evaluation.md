# Stage 1 blocking evaluation

**Reduced-pool experiment on the first 10,000 training S1 rows. No matching classifier has been built.**
Overall holdout recall is **99.65623%** (36 misses / 10,472 true links), meeting the requested overall >99.5% target for this reduced pool. India remains below that threshold at 99.32652%; US recall is 99.86075%.


Frozen main rule: take up to 60 ranked candidates with score ≥ 0.45, then union the top 12 address-route records and selective missing-address name rescues. The final set can exceed the main cap; actual mean/p95/max are shown below. Selection used development outcomes only; holdout was opened once after freezing.

| Metric | Development (tuned) | Holdout (untouched until freeze) |
|---|---:|---:|
| S1 references | 7,000 | 3,000 |
| Target records | 239,788 | 239,788 |
| Comparison space | 1,678,516,000 | 719,364,000 |
| TP | 24,208 | 10,436 |
| FN | 72 | 36 |
| FP | 242,412 | 104,466 |
| TN | 1,678,249,308 | 719,249,062 |
| Recall / pair completeness | 99.70346% | 99.65623% |
| FN rate | 0.29654% | 0.34377% |
| Candidate precision | 9.07959% | 9.08252% |
| Reduction ratio | 99.98412% | 99.98403% |
| Specificity | 99.98556% | 99.98548% |
| Candidate F1 | 0.166435 | 0.166478 |
| Final candidate pairs | 266,620 | 114,902 |
| Candidates/S1 mean | 38.088571 | 38.300667 |
| Candidates/S1 p95 | 72 | 72 |
| Candidates/S1 max | 87 | 88 |
| Oracle macro F0.5 ceiling | 99.90614% | 99.91752% |
| S1 with ≥1 missed true match | 67 | 29 |
| True singletons | 395 | 165 |

## Per-country holdout results

| Metric | India | US |
|---|---:|---:|
| S1 references | 1,152 | 1,848 |
| Target records | 239,788 | 239,788 |
| Comparison space | 276,235,776 | 443,128,224 |
| TP | 3,982 | 6,454 |
| FN | 27 | 9 |
| FP | 52,330 | 52,136 |
| TN | 276,179,437 | 443,069,625 |
| Recall / pair completeness | 99.32652% | 99.86075% |
| FN rate | 0.67348% | 0.13925% |
| Candidate precision | 7.07132% | 11.01553% |
| Reduction ratio | 99.97961% | 99.98678% |
| Specificity | 99.98106% | 99.98823% |
| Candidate F1 | 0.132027 | 0.198423 |
| Final candidate pairs | 56,312 | 58,590 |
| Candidates/S1 mean | 48.881944 | 31.704545 |
| Candidates/S1 p95 | 74 | 68 |
| Candidates/S1 max | 87 | 88 |
| Oracle macro F0.5 ceiling | 99.83371% | 99.96976% |
| S1 with ≥1 missed true match | 20 | 9 |
| True singletons | 66 | 99 |

## Per-country development results

| Metric | India | US |
|---|---:|---:|
| S1 references | 2,805 | 4,195 |
| Target records | 239,788 | 239,788 |
| Comparison space | 672,605,340 | 1,005,910,660 |
| TP | 9,759 | 14,449 |
| FN | 48 | 24 |
| FP | 124,100 | 118,312 |
| TN | 672,471,433 | 1,005,777,875 |
| Recall / pair completeness | 99.51055% | 99.83417% |
| FN rate | 0.48945% | 0.16583% |
| Candidate precision | 7.29051% | 10.88347% |
| Reduction ratio | 99.98010% | 99.98680% |
| Specificity | 99.98155% | 99.98824% |
| Candidate F1 | 0.135857 | 0.196273 |
| Final candidate pairs | 133,859 | 132,761 |
| Candidates/S1 mean | 47.721569 | 31.647437 |
| Candidates/S1 p95 | 74 | 68 |
| Candidates/S1 max | 87 | 76 |
| Oracle macro F0.5 ceiling | 99.82802% | 99.95838% |
| S1 with ≥1 missed true match | 43 | 24 |
| True singletons | 157 | 238 |

## Denominators and formulas

For every column, N = evaluated S1 count × 239,788 targets, including cross-country pairs. P = TP + FN is the number of ground-truth links; C = TP + FP is the final candidate count. TN = N − TP − FN − FP. Recall = TP/P; precision = TP/C; reduction ratio = 1 − C/N; specificity = TN/(TN+FP); candidate F1 = 2TP/(2TP+FP+FN).

The oracle removes every false candidate. For each non-singleton S1 with h retained true links and t total true links, oracle F0.5 = 1.25h/(h+0.25t), with 0 when h=0. True singletons score 1. The reported ceiling is the average across all S1 references; it is not an achieved matching score. p95 uses the nearest-rank empirical percentile.

## Scope and honesty

The first 10,000 S1 records were split deterministically into 7,000 development and 3,000 holdout entities. All 34,752 labeled targets were retained, plus 205,036 deterministic distractors from a 2% sample of other training targets. Every evaluation query searches the same 239,788-record pool. Entity groups were audited against the full ground truth; sampled targets have no second S1 owner. Labels were used to construct and evaluate this enriched pool, never as retrieval features.

The full training corpus has 10,320,219 targets. Dropping most distractors makes this experiment optimistic; first-row sampling can also introduce bias. Results are neither a full-corpus benchmark nor a France evaluation. France has no training labels. The country field is not used as a retrieval restriction, but that does not establish unseen-country accuracy. No leaderboard performance is claimed.

## Development tradeoff and protocol amendment

The valid initial lexical/address/trigram baseline recalled 94.12685% at 20 candidates/S1 and 98.93740% in its full raw union. Character/script errors and partial addresses motivated the phonetic and selective rescue routes. A score-only choice achieved 99.75288% development recall at 75.50 candidates/S1. Before any holdout results were opened, the development guardrail was amended from 99.75% to 99.70% to favor the smaller selective configuration: 99.70346% recall at 38.09 candidates/S1. The requested holdout target remained 99.5%. This was a development-stage budget decision; no settings were changed after holdout.

The first compact-index diagnostic returned zero BM25 ranks and was invalid. Its artifacts remain separately labeled; those numbers are not reported as a valid retrieval baseline. All tuning grids and valid baseline results are retained in the iteration log.

## Teammate comparison

The supplied teammate example reports comparison space 385,710,000, TP 16,553, FN 697, FP 159,247, TN 385,533,503, recall 95.96%, precision 9.42%, RR 99.9544%, specificity 99.9587%, and candidate F1 0.1715. Their code, subset, target pool, and split were unavailable. Our table uses the same kinds of metrics, but neither absolute counts nor percentages support an apples-to-apples improvement claim.

## Runtime and memory

- Reduced index: 29.12 seconds; peak process RSS 112.6 MB; disk size 86.4 MB. The target scan read all 10,320,219 rows once and retained 239,788.
- Development retrieval: 271.36 seconds for 7,000 references; peak process RSS 492.6 MB.
- Holdout retrieval: 119.91 seconds for 3,000 references; peak process RSS 227.6 MB.

- Three workers: maximum child peak RSS was 157.2 MB for development and 158.2 MB for holdout.

RSS uses macOS `resource.getrusage` bytes. Parent and maximum child peaks are separate; aggregate concurrent worker/system memory and filesystem cache were not measured. Query timing covers index opening and retrieval/ranking, excluding metric rendering. Pool construction time excludes split preparation. The initial index configuration was repaired from the saved pool; `index_meta.json` records the additional repair time. Aborted pilots and development tuning are recorded separately and are not included in final retrieval timing. These are measured local values, not full-corpus projections.

## Artifacts and reproduction

See `code/business_entity_resolution/README.md` for commands and `reports/evaluation_protocol.md` for the locked design. `artifacts/blocking/frozen_config.json` records the decision and hashes. Development sweeps, exact final `candidate_pairs.tsv`, runtime, metrics, and missed-pair text are saved under each split directory. The candidate files contain the exact stage-one set intended for the eventual matcher. They are local evaluation artifacts, not a test-set submission. No `matching_results.tsv` or artificial final classifier output was created.

The current ranking rule and the tuned score threshold are blocking heuristics. Candidate size is explicitly measured because the newer organizer update says it influences ranking, overriding the older local README statement.

## Remaining errors and checks

Of 36 held-out misses, 22 never entered the bounded raw retrieval union and 14 were removed by final candidate selection. 11 have empty target addresses and 20 have Indic-script target names; these categories overlap. No configuration changes followed this inspection.

Examples include City Tech → Gujarati-script name with an abbreviated and altered street number; Oncology Medicine → a misspelled name with no address; and Bright Pub → Bbrt-Pub with spelled-out street numbering and a full state name. These show remaining transliteration, abbreviation, numeric, and severe name-corruption failures. The complete 36 missed pairs, their original text, and raw retrieval ranks are saved in `artifacts/blocking/holdout/misses.json`.

Seven focused unit tests passed, covering confusion counts, singleton-aware oracle scoring, cutoff/rescue selection, Unicode derivation, feature isolation from IDs/country, and nonzero FTS ranking. The organizer validator helper verified all 7,000 development and 3,000 holdout candidate rows against the exact reference/target IDs. An additional check confirmed exported candidates exactly equal the frozen selection rule and that every labeled target is present in the pool. Twenty development retrieval reruns matched cached results exactly. This validates local evaluation files, not a full test-set submission.

## Measured construction detail

The initial pool scan/index build took 29.12s. The final in-pool index rebuild took an additional 12.68s with 124.4 MB peak RSS. Development timing predates cached selective-rescue enrichment; the holdout timing includes the final rescue logic. The unused experimental phonetic index and SQLite free pages are included in the reported on-disk size.
