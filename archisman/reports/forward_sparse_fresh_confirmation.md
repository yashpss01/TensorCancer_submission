# Frozen batched retriever on a disjoint 10k cohort

The six-view, top-64-per-view forward sparse retriever was fixed after two
exposed development screens. The next 10,000 original training Source-1 rows
(zero-based 310,000–319,999) and their truth were sealed before retrieval.
The target pool was the prior reduced 443,023-target pool plus every positive
for this new cohort, yielding 476,856 targets. Source-1 IDs were checked for
uniqueness within the slice, and selected target ownership was audited against
all training truth rows. See ignored local seal at
`work/fresh_10k_sparse_confirm/seal.json`.
The selected S1 IDs have zero overlap with the three prior exposed 10k
cohorts.

This is a **positive-enriched reduced-pool** confirmation. It does not measure
full 10-million-target competition, France, test predictions, or a Portal
score. The labels selected required target records for the pool and scored
completed outputs; the retriever used text fields alone. The rich XGBoost
model and its 0.775 threshold remained frozen.

| Metric | Overall | India | US |
| --- | ---: | ---: | ---: |
| S1 records | 10,000 | 3,977 | 6,023 |
| Candidate TP | 34,426 | 13,663 | 20,763 |
| Candidate FN | 86 | 55 | 31 |
| Blocking recall | 99.7508% | 99.5991% | 99.8509% |
| Candidate pairs | 2,554,584 | 1,063,839 | 1,490,745 |
| Candidates/S1 mean | 255.46 | 267.50 | 247.51 |
| Final matching TP | 33,587 | 13,279 | 20,308 |
| Final matching FP | 187 | 84 | 103 |
| Final matching FN | 925 | 439 | 486 |
| **Per-S1 macro F0.5** | **98.6527%** | **98.3201%** | **98.8723%** |

The new method clears the 99.5% *blocking recall* guard here, but final
matching remains below the 99.5% F0.5 objective. Its 255 mean candidates/S1
also make rich feature scoring the throughput bottleneck. Retrieval, candidate
export, and the retrospective screen took 104.73 seconds; rich scoring of the
same 2.55 million pairs took 1,481.58 seconds on one local CPU process with
2.29 GB recorded peak RSS. This is a measured bounded-pool run, not a full-test
runtime forecast. Candidate SHA-256 was
`32e2812cb4672dffd2f624db7edafac3d082e9154ba7e020e7912c8fd1ca4fad`;
matching SHA-256 was
`9fdf5697ae2e7a8cd81a2170c2ab3446499e0394f23ee03faf4da781eca69000`.
The scored candidate copy matched the exported candidate hash, and the model
manifest matched the seal. The evaluator verified all 10,000 S1 rows, unique
candidates and matches, and match-subset inclusion.

## Paired original-retriever comparison

The original frozen FTS retriever was run on the **same 10,000 S1 rows and
476,856 targets**. Its 508,320 candidate pairs were then scored by the same
frozen rich model and 0.775 threshold. The evaluator checked candidate and
matching digests, the cohort seal and model manifest, S1 order and coverage,
unique IDs, and that every predicted match appeared among the candidates.

| Metric | Original FTS + rich matcher | Six-view batched + rich matcher |
| --- | ---: | ---: |
| Candidate pairs | 508,320 | 2,554,584 |
| Candidates/S1 mean | 50.83 | 255.46 |
| Candidate TP / FN | 34,430 / 82 | 34,426 / 86 |
| Blocking recall | 99.7624% | 99.7508% |
| Final matching TP / FP / FN | 33,670 / 208 / 842 | 33,587 / 187 / 925 |
| **Per-S1 macro F0.5 overall** | **98.6720%** | **98.6527%** |
| India macro F0.5 | 98.3799% | 98.3201% |
| US macro F0.5 | 98.8649% | 98.8723% |

The batched retriever is 0.0193 percentage points lower in overall macro
F0.5 and 0.0599 points lower in India, although it is 0.0075 points higher in
the US. It also gives the rich scorer 5.03 times as many candidate pairs. The
original retriever has four more true candidate links overall, including 11
more in India, and higher overall and India blocking recall. **This batched
retriever is not promoted as an output-preserving or quality-improving
replacement.** Neither pipeline reaches the 99.5% final F0.5 objective on
this reduced pool. The original FTS candidate oracle macro F0.5 was 99.9318%,
so matcher decisions are the larger quality gap in this paired run.

A retrospective link-overlap audit found 34,385 true links in both candidate
sets, 45 only in FTS, 41 only in the batched set, and 41 in neither. The two
candidate lists overlap by just 36.16 candidates per S1 on average. Sampled
FTS-only links include cross-script Indian names, severe name typos, and
shortened or empty target addresses. This diagnosis has now **exposed this
cohort**; any rule designed from these errors requires another disjoint
confirmation before a quality claim.

The original FTS run reported 1,404.73 seconds for retrieval plus its original
saved-model scoring, and the separate rich rescore took 224.59 seconds. A
different six-worker inference process ran concurrently with that FTS run,
so these wall times are **not** a controlled end-to-end speed comparison. The
batched path took 104.73 seconds for candidate export and 1,481.58 seconds
for rich scoring; its much larger candidate set shifted the bottleneck to
feature computation. Its rich scorer reported 2.29 GB peak RSS versus 1.93 GB
for the FTS candidate rescore, but these are separate process peaks, not
complete-pipeline memory totals. The prior 98.7707% rich-model confirmation
used another cohort and target pool and must not be compared directly.

Original FTS candidate SHA-256:
`ffd335e9fa256ac84ef1a620cafd06b39d5e68d653c5845cfbd6032c81bfdf4a`.
Its rich matching SHA-256:
`7f20477d4d7097fa866e86fb592909bb45ad8f250b9f1244afffefe750ea8ad1`.
The ignored local paired score is at
`work/fresh_10k_sparse_confirm/score_frozen_fts.json`.
