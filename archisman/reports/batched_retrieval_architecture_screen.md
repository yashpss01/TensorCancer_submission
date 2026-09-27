# Bounded batched retrieval architecture screen

This experiment tested an Akanksha-inspired, reverse TF-IDF sparse retrieval
path as a possible replacement for repeated per-business SQLite FTS searches.
It is an architecture screen on the **already exposed** fresh-v1 cohort, not a
new holdout, Portal score, France result, or full-corpus validation. No full
test TSVs were generated.

## Comparable validation scope

- Original training Source-1 rows 280,001–290,000: 10,000 businesses, 4,045
  India and 5,955 US, with 34,471 known links.
- The same positive-enriched reduced pool of 442,904 targets as the frozen FTS
  evaluation. The batched retriever indexes all 2,206,821 training S1 records
  so the selected 10,000 compete in realistic global rank positions, then
  filters their IDs *after* ranking. It partitions by country; its
  comparison-space denominators therefore differ from the FTS report.
- Retrieval on Modal had no truth file mounted. Candidate shards were
  downloaded and scored against labels locally. The frozen original matcher
  was used unchanged for the narrow v2 candidate set.

| Metric | Frozen FTS | Batched narrow (keep 4, rank 40) | Batched wide (keep 16, rank 80) |
| --- | ---: | ---: | ---: |
| Candidate pairs | 503,552 | 49,930 | 71,532 |
| Candidates per S1, mean / p95 / max | 50.36 / 87 / 106 | 4.99 / 10 / 249 | 7.15 / 15 / 1,575 |
| True links offered / missed | 34,379 / 92 | 33,520 / 951 | 33,792 / 679 |
| Blocking recall, overall | 99.7331% | 97.2412% | 98.0302% |
| Blocking recall, India | 99.5774% | 95.6955% | 96.7197% |
| Blocking recall, US | 99.8391% | 98.2934% | 98.9224% |
| Perfect-decision macro F0.5 ceiling, overall | 99.9179% | 99.1642% | 99.3857% |
| Perfect-decision macro F0.5 ceiling, India | 99.8793% | 98.6714% | 98.9472% |
| Frozen matcher macro F0.5, overall | 98.4139% | 97.7186% | Not run; ceiling is below goal |
| Frozen matcher macro F0.5, India | 98.2916% | 97.1766% | Not run |
| Frozen matcher macro F0.5, US | 98.4970% | 98.0869% | Not run |

The wider configuration recovers 272 more true links than the narrow one, but
still misses 679. Even perfect match decisions could not reach the required
99.5% overall macro F0.5 on this reduced pool. Both batched configurations
are therefore **rejected as direct replacements**. The existing FTS blocker
and its saved models remain the validated pipeline; no production retrieval
rules were changed.

The narrow run's country workers took 613.1 seconds (India) and 673.9 seconds
(US), concurrently, including roughly 443 seconds each to construct their S1
indexes. The wide run took 531.1 and 628.1 seconds respectively, including
367.2 and 434.6 seconds for indexing. The wide app was active from 05:00:29 to
05:11:13 IST. These are candidate-only cloud runtimes. The local frozen FTS
run's 1,451.94 seconds includes retrieval, feature computation, prediction,
and TSV writing, so these numbers are **not** a controlled end-to-end speedup.
Index reuse and full-corpus target size would also change the comparison.

Of the narrow batched path's 951 missed true links, 872 are available in the
original FTS candidate set. A retrospective gate based on low top similarity
would have to invoke FTS on about 93.7% of S1 rows to achieve 99.5% union
blocking recall on this exposed batch. The miss distribution spans low and
high candidate counts; a fallback only for empty candidate lists would not
repair recall. This retrospective rule is a diagnostic, not a selected model
or fresh validation.

Saved machine-readable results: `work/v2_bounded_v1/score.json` and
`work/v2_bounded_v1/wide_retrieval_score.json` (ignored local artifacts).
Cloud apps `ap-vkDZXRbttHAglusuFWo6DS` and
`ap-vIoHHjJbitVdftmp270NFG` have stopped. Their posted combined Modal cost
was about $0.288 at the last billing check; total posted Modal cost for the
project was about $1.164, below the user-approved $5 cap. Billing can lag.

The practical speed improvement already implemented is the candidate-cache
scorer documented in [pipeline_performance_profile.md](pipeline_performance_profile.md).
Further first-pass retrieval redesign needs a new high-recall batched method
and fresh disjoint quality validation. The 99.5% final matching goal also
requires better match decisions: the FTS candidate ceiling is high, but its
frozen matcher scores only 98.41% on this reduced-pool cohort.
