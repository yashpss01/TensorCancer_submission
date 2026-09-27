# Frozen entity-resolution pipeline: measured performance

This report profiles the current `src/infer.py` pipeline before changing its
retrieval rules. The 10,000-business validation pool has 442,904 targets and
is **not** the full 9.97-million-target test index. No quality number here is a
Portal or France result. Wall-clock comparisons on different machines or
cache states are identified separately.

## Execution flow

1. `infer.py index` streams raw S2/S3 TSV rows, normalizes names and addresses,
   and builds a `records` table plus two SQLite FTS5 indexes. The index is
   reusable across matcher experiments.
2. `infer.py run` streams raw S1 TSV rows. Worker processes run the original
   six FTS routes (`evaluate.Retriever`) and two rescue searches plus
   anchor-based rescoring (`inference_retrieval.Improved`) for each S1.
3. The selected candidate IDs are fetched from `records`. `features.py`
   computes 39 base pair features; `inference_features.py` computes 20 extra
   typo/sequence features and 26 candidate-group features. Normalization and
   representation functions have bounded in-process LRU caches.
4. Three saved XGBoost models produce the final probability; threshold 0.74
   selects matches. Two TSVs are written in S1 order, followed by metadata.
5. Research training and evaluation are separate: cached NumPy feature arrays
   under `artifacts/matching_round5/` feed model training. Saved 10k truth is
   opened by an external scorer, not by inference.

## Stage timing

The controlled single-worker profile processed the first 500 fresh S1 rows
against the same 442,904-target reduced pool used in the 10k validation. The
profile did not modify production source, model files, candidate rules, or
evaluation methodology. It selected 25,486 candidate pairs (50.97/S1) from
92,188 unique stage-1 hits and 38,816 unique stage-2 hits. These raw stage
counts overlap; they are not comparisons against every target.

| Stage | Wall time | Share of profiled loop |
| --- | ---: | ---: |
| Original retrieval/ranking | 106.47 s | 44.5% |
| Rescue retrieval/ranking | 120.49 s | 50.4% |
| Candidate selection | 0.007 s | <0.1% |
| Pair/group features plus saved-model scoring | 12.14 s | 5.1% |
| **Total 500-row loop** | **239.12 s** | **100%** |
| TSV serialization in memory, after loop | 0.006 s | separate |

Loading 500 S1 rows and the model bundle happened outside that loop.
Observed throughput was 2.09 S1/s in one process and peak process RSS was
800.9 MB. The production six-worker 10k reduced-pool run took 1,451.94 s
(6.89 S1/s) with 503,552 selected pairs. The associated positive-enriched
comparison space is 10,000 × 442,904 = 4,429,040,000 possible pairs; the code
does **not** enumerate that Cartesian product.

A separate SQL-instrumented 100-row run took 47.74 s. It performed 988 FTS
searches taking 37.55 s (78.7% of the loop), 50,261 vocabulary lookups taking
3.56 s, and 406 record fetches taking 0.05 s. Thus the cost is repeated FTS
posting/rank computation, followed by vocabulary lookups and text features.
The `cProfile` run of the same 100 rows spent 41.38 of 56.54 profiled seconds
inside `sqlite3.Connection.execute`. cProfile overhead makes its absolute
times unsuitable for speedup claims. Its top cumulative operations were:

| Operation | Calls | Cumulative profiled seconds |
| --- | ---: | ---: |
| SQLite `Connection.execute` | 51,655 | 41.38 |
| Rescue `retrieve` | 100 | 27.41 |
| Original `retrieve` | 100 | 23.33 |
| Rescue FTS `search` | 398 | 20.50 |
| Original FTS `search` | 590 | 17.69 |
| `match_batch` | 1 | 5.96 |
| `extra_vector` | 5,069 | 3.07 |
| Indic name `romanize` | 60,663 | 3.06 |
| `phonetic.key` | 39,802 | 3.01 |
| `name_score` | 55,819 | 2.98 |

Some rows are nested cumulative times and must not be added. The first five
especially overlap. Full raw output is in ignored local files
`work/fresh_10k_v1/inference_cprofile_100.txt` and
`work/fresh_10k_v1/inference_profile_100_sql.json`.

The full test target index took 1,608.3 s to build once: 9,969,589 targets,
3.21 GB main SQLite file, 1.18 GB rescue index, and reported peak RSS 458.5 MB.
Cached *training* pair-feature extraction previously took 126.3 s for
1,488,094 pairs and 45.7 s for 493,573 validation pairs; the recorded initial
logistic/XGBoost training took 35.4 s. Later group and augmented model fits
recorded 36.3 and 71.3 s respectively. The hours-long end-to-end latency is
therefore driven by retrieval and, when needlessly repeated, index building;
model fitting is not the primary bottleneck.

## Blocking routes and quality guard

On 100 profiled S1 rows, each original route contributed about 18–20k raw
hits across rows before deduplication. The six routes' hits on the 321 known
true links ranged from 236 to 288. Only 14 true links were exclusive to the
address route; each of name, joint, character, and phonetic was exclusive for
one, while phonetic-joint was exclusive for zero in this tiny sample. This is
**not** enough evidence to delete a route, because rare links are exactly what
the 99.5% recall guard is meant to protect. The full fresh 10k run retained
34,379 of 34,471 true links (99.7331% recall) at 503,552 candidates. The
local-to-Modal candidate difference was one negative swap; true-link coverage
and final scores did not change.

There is no accidental explicit `S1 × targets` Python loop in frozen inference.
The expensive operations are FTS searches and ranking over matching postings,
run roughly ten times per S1. Python-level name/phonetic/sequence loops add
cost but are secondary. The small candidate caps prevent quadratic
all-target feature generation. Indexing and feature arrays are already cached
as separate artifacts, but the original `run` command recomputed retrieval
for every matcher-only experiment.

## Optimization added after profiling

`infer.py score-cached` now scores a previously generated candidate TSV without
repeating retrieval. It verifies candidate digest (when prior metadata is
present), Source-1 order, duplicate IDs and target existence, then writes
fresh TSVs and records the cache/model/index metadata. The original `run`
command and ranking behavior remain available unchanged. On the complete
fresh 10k reduced-pool batch:

| Measure | Original `run` | Cached scorer |
| --- | ---: | ---: |
| Wall time | 1,451.94 s | 227.87 s |
| Candidate pairs | 503,552 | 503,552 |
| Candidate TSV SHA-256 | `009d6a1f...` | same |
| Matching TSV SHA-256 | `369e24f8...` | same |
| Blocking recall | 99.7331% | same exact candidates |
| Final macro F0.5 | 98.4139% | same exact predictions |

This is a **6.37× faster matcher-only iteration**, not a 6.37× faster first
end-to-end run. Existing training arrays already similarly avoid retrieval
when only hyperparameters change. Peak RSS before/after on the *full 10k*
comparison was not measured, so no full-run memory-improvement claim is made.
On the same first 500 S1 rows, a subsequent production `score-cached --limit
500` pass took 12.37 seconds and its process peaked at 585.3 MB sampled RSS.
The earlier read-only original 500-row loop took 239.12 seconds and recorded
800.9 MB peak process RSS. These different runners and measurement methods
make the memory figures indicative, not a controlled before/after percentage.

A read-only A–B–B–A SQLite setting test preserved candidate hash across all
four 100-row reduced-pool passes. Default retrieval took 46.30 and 43.36 s;
larger caches plus 1 GiB mmap took 42.70 and 42.36 s. This ~5% gain is small
and increased peak RSS by roughly 294 MB in that process. It has **not** been
applied to production. A same-query benchmark against the complete target
index was also run before deciding whether it is worthwhile there.

On five real test S1 rows against all 9,969,589 indexed targets, the
default setting took 58.93 s cold and 48.33 s on a warmed repeat. The
larger-cache/mmap setting took 56.14 and 52.66 s respectively. All four
candidate-list hashes matched, at 426 candidates across the five rows. The
tuned process's reported peak RSS rose from 469.5 MB to 1.66 GB. The warm
default was faster than the warm tuned run, so the setting is rejected. A
larger 50-row × four-setting test was stopped after it proved too slow and
caused memory pressure; it produced no usable timing report. These five test
rows are only a throughput warning, not a full-run estimate or quality score.

Another read-only check reconstructed all original and rescue FTS expressions
for 1,000 fresh S1 records without executing their searches. Only 34 of about
7,919 nonempty route queries repeated (0.4%); six of eight routes had zero or
at most two repeats. Memoizing individual FTS result lists would therefore
offer negligible reuse on this cohort and has not been implemented. Raw counts
are in `work/fresh_10k_v1/fts_reuse_1000.json`.

## Remaining work

- Expand the complete-target-index throughput sample only after implementing
  progress reporting and a time bound; do not extrapolate reduced-pool speed as
  if the target universe stayed fixed. The first five test rows show much
  higher per-S1 latency and make the current FTS implementation unsuitable
  for an immediate full-test run on the local Mac.
- A separate [batched sparse retrieval screen](batched_retrieval_architecture_screen.md)
  tested two candidate allowances against the full training S1 index and the
  same reduced target pool. Both missed too many known links; even the wider
  setting had only 98.0302% blocking recall and a 99.3857% perfect-decision
  macro F0.5 ceiling. It is not an accepted replacement. Any redesigned
  candidate generator needs a fresh disjoint quality gate before use.
- Keep the saved index, candidates, and deterministic feature arrays reusable.
  Profiling a full rebuild on every model tweak would spend time without
  changing the model comparison.
- More cloud CPU can parallelize independent S1 shards after sharing a frozen
  read-only index. It will not make the underlying per-S1 FTS queries cheap and
  adds upload, image-build, and merge costs. A GPU is not indicated by this
  profile because model fitting was tens of seconds. No AWS migration is
  justified before a full-index CPU throughput benchmark and quality gate.

The sampled scorer and all 14 existing tests passed. Neither the cached scorer
nor SQLite read-setting experiment changes the evaluation cohort or the
underlying matching decision rule.

## Findings and optimization outcome

The three largest measured sources of elapsed time are (1) repeated FTS5
postings/ranking work on each S1 business, including the rescue searches;
(2) rich matcher text-feature calculation, especially sequence similarity,
when candidate volume grows; and (3) rebuilding the target index if the
reusable 9.97-million-target index is discarded. Training itself took tens of
seconds in the recorded runs, so moving training to a GPU is not the first
performance fix. No explicit all-pairs Cartesian product was found in the
frozen inference path. The historical ~385.7-million comparison-space numbers
in the brief were not reused as benchmark results; all counts above are
recomputed for their stated cohorts and target pools.

The code changes delivered a **6.37× matcher-only iteration speedup** by
reusing verified candidate TSVs (1,451.94 to 227.87 seconds on the same 10k
reduced-pool cohort) and a further **1.15× rich rescore speedup** from bounded
feature caches (302.48 to 262.09 seconds on an exact 10k replay). Both checks
had byte-identical candidate and prediction TSV hashes before and after.
The first-pass full-index runtime did not improve by these same factors;
repeated FTS retrieval remains. The rich cache increased measured peak process
RSS by 67.6 MB, so no memory reduction is claimed. FTS tuning with a larger
SQLite cache/mmap was rejected after its warm run slowed and RSS rose.

A separate six-view batched retriever was measured on two exposed screens and
one disjoint 10k reduced-pool confirmation, then paired against the original
frozen retriever on that same fresh cohort. See
[the paired confirmation](forward_sparse_fresh_confirmation.md). It made
2,554,584 candidates versus the original 508,320. With the **same frozen rich
matcher**, overall per-S1 macro F0.5 was 98.6527% versus 98.6720% for the
original path; India was 98.3201% versus 98.3799%. The batched retriever is
therefore only an experimental architecture screen, not a production
replacement. Its candidate export was fast, but scoring 5.03× as many pairs
consumed 1,481.58 seconds on one local CPU process. The paired FTS run was
contended by an independent inference process; these measurements cannot
support a controlled full-run speedup claim.

The next useful structural optimization is a **selective batched retriever**
that keeps difficult positive links while sharply reducing its candidate
union, followed by a disjoint quality gate and a full-index throughput pilot.
More CPU cores or cloud workers may shorten independent S1 shards once that
path is proven, but scaling the current per-S1 FTS queries multiplies cost
without removing the repeated work. No AWS, SageMaker, EC2, or GPU move is
justified by the current measurements alone. Full-corpus inference, France
performance, and a Portal score remain unmeasured.

## Rich-matcher feature profile and exact-output speedup

A later frozen rich matcher adds Unicode comparisons to the original model.
Its separately controlled 256-S1, 13,449-pair profile on the already exposed
final 10k cohort took 18.25 seconds under `cProfile`. The largest cumulative
operations were `extra_vector` (8.84 s), `difflib.SequenceMatcher.ratio`
(7.84 s, nested within the feature work), `group_matrix` (4.40 s), and
`represent` (3.43 s). These cumulative times overlap; saved-model prediction
was not the primary cost. Exact name, address, and phonetic comparisons repeat
within and across candidate groups, especially where many businesses share
generic tokens.

`inference_features.seq` and `soft` now have bounded 100,000-entry caches.
Group-context scoring reuses the exact `seq` result that pair-feature scoring
already calculated, and prepares invariant query address numbers once per
group. `rich_infer` reuses the existing normalized core and transliterates the
query name/address once per candidate group. These changes leave feature
definitions, candidate lists, model assets, thresholds, and evaluation rows
unchanged.

| Same 10,000-S1 rich replay | Before | After |
| --- | ---: | ---: |
| Candidate pairs | 508,773 | 508,773 |
| Complete rescore wall time | 302.48 s | 262.09 s |
| Peak process RSS | 1,860,354,048 B | 1,927,970,816 B |
| Candidate TSV SHA-256 | `b8b0f5b80a2f...` | identical |
| Matching TSV SHA-256 | `b200fb68f491...` | identical |

The complete bounded replay therefore improved by **1.15×**, with about
67.6 MB more peak memory for the caches. This is a matcher-stage improvement,
not a claim that full-test inference is now fast. On the separate 256-S1
scoring sample, before/after wall time was 6.65/4.99 s, and both frozen and
rich probability-array SHA-256 hashes were identical. The full replay also
verified every frozen baseline decision against its saved TSV.

The complete-target-index 100-S1 six-worker pilot took 263.20 s for 7,720
candidates. It is a small full-index throughput sample, not a reliable total
runtime forecast. The existing per-S1 FTS retrieval remains the main obstacle
to full-test delivery; the prior batched replacement failed the blocking
recall gate. No full-test TSVs or Portal score are implied by this replay.

After the cache change, a separate 100-S1 retrieval-through-TSV run on the
same reduced validation index took 14.72 s with six workers. Its candidate
and matching TSV bytes matched the first 100 rows of the saved 10k baseline
exactly. This checks the original entry point as well as the cached-rich
replay; it is not a complete-target throughput estimate.
