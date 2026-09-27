# Archisman research and inference handoff

This branch's existing `code/` and `output/` are Akanksha's work and remain
unchanged. Archisman's independent entity-resolution package is under
[`archisman/`](archisman/). It mirrors the project layout there: `code/`,
`output/`, `reports/`, `artifacts/`, `tests/`, and methodology documentation.
The snapshot contains Archisman's local `Archisman` commit `417fbcf` plus the
small, explicitly unfinished files listed below. No Akanksha candidate file,
prediction, model, feature cache, or learned dictionary is used by the
Archisman package.

The strongest separately confirmed Archisman **rich matcher** scored 98.7707%
per-Source-1 macro F0.5 on a 10,000-record, positive-enriched *reduced target
pool*. This is below the 99.5% objective and is not a full-corpus, France, or
Portal result. See
[`archisman/reports/rich_pair_fresh_confirmation.md`](archisman/reports/rich_pair_fresh_confirmation.md).
The subsequent paired retrieval experiment did not validate a faster
replacement: on the same fresh reduced-pool cohort, the original FTS
retriever plus frozen rich matcher scored 98.6720%, versus 98.6527% for the
six-view batched retriever. See
[`archisman/reports/forward_sparse_fresh_confirmation.md`](archisman/reports/forward_sparse_fresh_confirmation.md)
and the [performance profile](archisman/reports/pipeline_performance_profile.md).

**Archisman's full test TSVs have not been generated or validated.** The
top-level `output/matching_results.tsv` belongs to Akanksha's separate
pipeline; it must not be presented as Archisman's output. The Archisman
inference instructions are in
[`archisman/code/business_entity_resolution/README.md`](archisman/code/business_entity_resolution/README.md).
They describe the frozen candidate generator, the rich-matcher rescore,
ordered shards, TSV validation, and the unmeasured full-test limitations.
The project-level `archisman/README.md` and
`archisman/Documentation_template.md` describe the package and methodology.

The following files were copied from uncommitted local work for completeness
and should be treated as **draft experiments**, not validated submission
components: `archisman/code/business_entity_resolution/src/run_rich_submission.py`,
`archisman/experiments/decision_context_screen.py`,
`diagnose_dev_errors.py`, `diagnose_fresh_errors.py`,
`diagnose_text_collisions.py`, `listwise_rich_screen.py`,
`missing_address_expert.py`, `modal_bounded_pilot.py`,
`rescore_bounded_candidates.py`, `residual_rerank_screen.py`, and
`script_aware_feature_screen.py`. The `v2/retrieve.py` and
`experiments/modal_v2_candidates.py` copies include uncommitted CLI/process
changes. All these Python files passed syntax compilation; they have not
passed an end-to-end quality or full-test submission check.

No raw challenge data, full target index, generated scratch arrays, Modal
credentials, or Portal upload is included in this handoff.
