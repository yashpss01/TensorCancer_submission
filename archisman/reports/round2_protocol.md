# Round 2 protocol, defined before fresh matching outcomes

Development uses all first 10,000 S1 rows, including the now-inspected former holdout. Those records are no longer untouched evaluation data for round 2. The original code, frozen configuration, index, candidate files, and report are preserved.

Fresh evaluation uses all S1 rows 10,001–20,000 in original file order. Its pool retains all 34,726 labeled targets and 205,086 deterministic distractors, for 239,812 total targets. Distractors use the same 2% SHA256 selection rule as round 1. Both baseline and improved pipelines search exactly this pool. Positives are guaranteed present; most other targets are absent, so this remains an enriched, optimistic reduced-pool benchmark.

A full ground-truth ownership audit found no shared labeled target between either batch or any other S1 group. The two batches share no S1 ID and no exact normalized name/address/country signature. The shared deterministic distractor sample is deliberate, not a labeled training feature. Near-duplicate business identities beyond the supplied truth/exact-text audit are not independently verified.

The improved pipeline preserves the original final candidates and permits only bounded additional candidates. New retrieval uses coarse phonetic bigrams combined with address context, normalized address tokens (abbreviations, ordinal numbers, leading zeros), and up to three strongly linked text-derived anchors. It does not train a matching classifier. Original texts stay intact. Country labels do not filter retrieval; unknown labels remain supported.

Development examines rescue caps 4, 8, 12, 16, and 24 and score cutoffs 0.50–0.80. The starting selection policy is smallest mean candidate count with at least 99.70% development recall overall and in India, if feasible; otherwise report the best bounded configuration and limitations. Every original candidate is retained, so paired US recall cannot decrease. Candidate count, runtime, and memory still need measurement.

Code and configuration hashes are frozen before running either pipeline on fresh matching outcomes. Baseline runs first, improved rescues run from its cached retrieval, and scoring occurs only after both candidate sets are complete. No new tuning follows fresh results. The marker `FRESH_OPENED.json` and frozen hashes record this boundary. Labels may be read during pool construction and ownership audits, but are never retrieval inputs.

Requested aspiration: at least 99.5% recall overall and in India, with small candidate lists. Failure is to be reported honestly. No full-corpus, unseen-France, or final matching-score claim follows from this experiment.
