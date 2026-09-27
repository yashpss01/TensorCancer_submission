# Batched sparse retrieval: exposed-cohort screen

This is an independent retrieval experiment on **already exposed** original
training S1 rows 280,001–290,000 (10,000 S1; 34,471 known positive links).
The target pool has 442,904 positive-enriched S2/S3 records. It is not the
9.97-million-target test corpus, a new holdout, a France result, or a Portal
score. Labels were used to score retrieved IDs after retrieval, not as text
features or retrieval inputs.

The original frozen inference performs repeated per-S1 SQLite FTS searches.
This screen instead fits independent TF-IDF indexes on the bounded target pool
and queries S1 in sparse batches using our own `blocking.tokens` and
`phonetic.key` normalizers. Views are name words, address words, transliterated
name characters, phonetic-name characters, joint name/address words, and
transliterated address characters. Each view contributes its top *k* targets;
candidate lists are deduplicated per S1. The screen deliberately does not
run the saved matcher, so blocking recall is the only quality result here.

| Views / per-view k | India recall | US recall | India candidates/S1 | US candidates/S1 | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 views, 128 | 99.0403% | 99.8147% | 338.1 | 312.2 | 28.9 s |
| 4 views, 32 | 99.1477% | 99.5514% | 106.3 | 98.7 | 45.8 s |
| 4 views, 64 | 99.4700% | 99.7806% | 217.0 | 199.4 | 45.8 s |
| 4 views, 128 | 99.6276% | 99.8927% | 431.3 | 398.2 | 45.8 s |
| 6 views, 32 | 99.2909% | 99.7903% | 131.5 | 122.7 | 80.1 s |
| **6 views, 64** | **99.5846%** | **99.8976%** | **267.4** | **246.7** | **80.1 s** |

The wall times include loading the bounded target pool, fitting every view,
querying 10,000 S1 records, and the Python metric aggregation. They are not
full-test projections; index size, posting density, RAM, and France are
different at full scale. The six-view rank-64 union recalls 13,904/13,962
India and 20,488/20,509 US links (34,392/34,471 overall, 99.7708%). Its
candidate volume is about five times the frozen FTS candidate allowance on
this cohort.

Three cheap aggregate rankings were also tested on the six-view top-128
union. At a fixed 200-candidate cap, maximum cosine, sum cosine, and sum of
reciprocal ranks recalled 97.987%, 99.148%, and **99.499%** of India links,
respectively. Thus the tested cap did not strictly clear 99.5% for India.
This retrospective selection is not a validated candidate rule. The three-
and four-view configurations below the recall guard are rejected as direct
replacements; the six-view rank-64 configuration is a diagnostic only until
it is evaluated against an independent cohort, the full target index's
candidate volume, and the unchanged final matcher.

A separate reverse-direction screen indexed all 2,206,821 training S1 records
and queried only the known-positive target records. Its three-view top-256
union covered 99.0546% of India and 99.8879% of US links. This is an exact
positive-link rank diagnostic for those views with realistic global S1
competition, but it does not measure candidate output size from all targets.
It was not promoted. Raw local JSON: `work/fresh_10k_v1/positive_multiview_rank_screen.json`,
`work/fresh_10k_v1/forward_sparse_sixview_screen.json`.

The next necessary gate is an independently implemented bounded candidate
export with deterministic ordering, followed by frozen-matcher F0.5 and
candidate-volume checks on a disjoint cohort. Do not submit a batched blocker
based on these exposed reduced-pool recall numbers alone.

The same six-view top-64 rule was then exported without changing its
retrieval settings on a second **already exposed** 10k cohort (original
training S1 rows 290,001–300,000; 443,023 reduced targets). It retained
99.5980% of India links (56 misses) and 99.8110% of US links (39 misses),
with mean candidate counts 266.9 and 247.5 respectively. Its 2,552,458
candidate pairs were written in original S1 row order with lexically sorted
target IDs. This repeated blocking result is encouraging but remains a
reduced-pool, development-scope observation. No final matcher score was
inferred from it.
