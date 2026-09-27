# Rich matcher threshold headroom

This diagnostic holds the candidate lists and rich XGBoost features fixed. All
three 10,000-S1 cohorts used below are already exposed training data with
positive-enriched reduced target pools. It is neither a fresh confirmation nor
a full-corpus, France, or Portal result.

For each cross-cohort direction, two entity-disjoint folds of the **training**
cohort selected either one global threshold or separate India/US thresholds.
The fitted model and thresholds were then evaluated on the opposite exposed
cohort. The country is an inference-visible field; no test labels selected
these thresholds.

| Fit → test | Rule | Overall per-S1 macro F0.5 | India | US |
| --- | --- | ---: | ---: | ---: |
| v1 → v2 | Training-selected global threshold 0.700 | 98.7564% | 98.4724% | 98.9457% |
| v1 → v2 | Country thresholds: India 0.750, US 0.700 | 98.7486% | 98.4530% | 98.9457% |
| v2 → v1 | Training-selected global threshold 0.825 | 98.7959% | 98.6375% | 98.9035% |
| v2 → v1 | Country thresholds: India 0.675, US 0.825 | 98.8250% | 98.7095% | 98.9035% |

The country rule helped overall by 0.0291 percentage points in one direction
but hurt by 0.0078 in the other. It does not provide a repeatable basis for
changing the packaged matcher.

To measure threshold *headroom* for the frozen model, we also scored the
already exposed third cohort with its previously frozen checkpoint. We
exhaustively checked every distinct model score as a threshold, treating tied
scores together, and verified each maximizing threshold by independently
recomputing per-S1 macro F0.5. **This uses the exposed cohort's labels to
choose the best threshold and is an optimistic diagnostic, not a deployable
validation result.**

| Fixed score ordering on exposed third cohort | Overall macro F0.5 |
| --- | ---: |
| Previously frozen threshold 0.775 | 98.7707% |
| Best possible single threshold on these labels | 98.7884% |
| Best possible separate India/US thresholds on these labels | 98.7947% |

Even the label-selected country-threshold ceiling is **0.7053 percentage
points below 99.5%** on this reduced pool. It is a ceiling only for threshold
selection on these fixed pair scores and candidates, not for a different
model or joint decision method. The much higher 99.9257% candidate oracle
shows that most of the remaining headroom requires better discrimination
among retrieved candidates. No threshold from this diagnostic was promoted.
Full counts and exact selected values: [rich_country_threshold_screen.json](rich_country_threshold_screen.json).
