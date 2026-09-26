# Round 4: fresh validation of the blocking selection change

**Updated recall: 99.72687% overall; 99.61473% India.** The >=99.5% overall/India criterion was met. This is a 10,000-reference fresh slice scored after the rule was frozen.

| Metric | Frozen round 2 | Updated selection |
|---|---:|---:|
| S1 count | 10,000 | 10,000 |
| Target count | 409,141 | 409,141 |
| Comparison space | 4,091,410,000 | 4,091,410,000 |
| TP | 34,296 | 34,322 |
| FN | 120 | 94 |
| FP | 386,936 | 459,251 |
| TN | 4,090,988,648 | 4,090,916,333 |
| Recall | 99.65132% | 99.72687% |
| Candidate precision | 8.14183% | 6.95378% |
| Reduction ratio | 99.98970% | 99.98794% |
| Specificity | 99.99054% | 99.98878% |
| Candidate F1 | 0.150537 | 0.130010 |
| Candidate pairs | 421,232 | 493,573 |
| Candidates/S1 mean | 42.123200 | 49.357300 |
| Candidates/S1 p95 | 73 | 87 |
| Candidates/S1 max | 92 | 107 |
| Oracle macro F0.5 ceiling | 99.84994% | 99.87942% |
| True singletons | 584 | 584 |
| S1 with at least one miss | 111 | 88 |

## Country-level paired results

### India

| Metric | Frozen round 2 | Updated selection |
|---|---:|---:|
| S1 count | 4,018 | 4,018 |
| Target count | 409,141 | 409,141 |
| Comparison space | 1,643,928,538 | 1,643,928,538 |
| TP | 13,950 | 13,962 |
| FN | 66 | 54 |
| FP | 195,104 | 234,714 |
| TN | 1,643,719,418 | 1,643,679,808 |
| Recall | 99.52911% | 99.61473% |
| Candidate precision | 6.67292% | 5.61453% |
| Reduction ratio | 99.98728% | 99.98487% |
| Specificity | 99.98813% | 99.98572% |
| Candidate F1 | 0.125073 | 0.106299 |
| Candidate pairs | 209,054 | 248,676 |
| Candidates/S1 mean | 52.029368 | 61.890493 |
| Candidates/S1 p95 | 77 | 91 |
| Candidates/S1 max | 92 | 107 |
| Oracle macro F0.5 ceiling | 99.77032% | 99.81775% |
| True singletons | 226 | 226 |
| S1 with at least one miss | 60 | 49 |

### US

| Metric | Frozen round 2 | Updated selection |
|---|---:|---:|
| S1 count | 5,982 | 5,982 |
| Target count | 409,141 | 409,141 |
| Comparison space | 2,447,481,462 | 2,447,481,462 |
| TP | 20,346 | 20,360 |
| FN | 54 | 40 |
| FP | 191,832 | 224,537 |
| TN | 2,447,269,230 | 2,447,236,525 |
| Recall | 99.73529% | 99.80392% |
| Candidate precision | 9.58912% | 8.31370% |
| Reduction ratio | 99.99133% | 99.98999% |
| Specificity | 99.99216% | 99.99083% |
| Candidate F1 | 0.174961 | 0.153488 |
| Candidate pairs | 212,178 | 244,897 |
| Candidates/S1 mean | 35.469408 | 40.938984 |
| Candidates/S1 p95 | 69 | 78 |
| Candidates/S1 max | 83 | 97 |
| Oracle macro F0.5 ceiling | 99.90341% | 99.92085% |
| True singletons | 358 | 358 |
| S1 with at least one miss | 51 | 39 |

## What changed

The original round-2 candidate rule used threshold 0.70 and at most 4 rescues. The updated rule uses threshold 0.50 and at most 16 rescues from the same frozen retrieval rankings. It retains every original candidate. On this fresh slice it recovered 26 true links and lost none, adding 72,341 candidates (7.234 per S1; maximum 16), including 72,315 nontrue candidates. A larger candidate set increases downstream matching work and may lower candidate precision.

## Development and independent check

Rows 20,001–50,000 were inspected development data; the choice reached 99.55959% India recall with 49.603 candidates/S1 on that slice. Rows 50,001–60,000 were reserved as the next evaluation slice. The exact selection rule, source code, query/truth inputs, and index hashes were sealed at 2026-09-26T16:24:33Z before either fresh retrieval pass. No fresh-result tuning followed.

The common 409,141-target pool was already constructed for original S1 rows 20,001–80,000. It includes every labeled target for the fresh 10k and deterministic 2% distractors. The same pool and raw retrieval rankings serve both candidate rules. The pool includes positives for other S1 rows and omits most of the 10.32 million target corpus, so these are reduced-pool results; full-corpus and unseen-France behavior remain unverified. Labels were used to construct/audit the pool and to score outputs, not for retrieval. IDs identify records, and country does not filter retrieval.

## Validation and metric meaning

Validation PASS: both exports have exactly 10,000 aligned S1 rows; every exported target ID belongs to the sealed pool; all known true targets are present; the updated set contains the original set per reference; and the exports were reopened and compared with the exact selected sets. Frozen code and input hashes matched before both retrieval runs and scoring.

Comparison space = S1 rows × all pool targets, including cross-country pairs. TP/FN are kept/missed true links; FP are nontrue candidates; TN is every other pair. Candidate precision and candidate F1 measure blocking output, not final match decisions. The oracle macro F0.5 ceiling assumes a perfect later classifier that removes every false candidate, and assigns true singletons a score of one for an empty prediction. No final matching classifier or final F0.5 was evaluated.

## Runtime and artifacts

- baseline retrieval: 338.97s wall time, 6 workers, parent peak RSS 39.1 MB, maximum child peak RSS 243.6 MB.
- improved retrieval: 403.23s wall time, 6 workers, parent peak RSS 103.5 MB, maximum child peak RSS 136.9 MB.

These child peaks do not measure aggregate concurrent memory. Pool construction happened in round 3 and is separate from these retrieval times.

Run details: `code/business_entity_resolution/round4/README.md`. Frozen configuration: `artifacts/blocking_round4/frozen_config.json`. Exact candidate TSVs and metrics: `artifacts/blocking_round4/fresh/frozen/` and `artifacts/blocking_round4/fresh/updated/`. Earlier code and artifacts are preserved.
