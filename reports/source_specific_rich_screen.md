# S2/S3-specific matcher screen

This is a comparison on the already exposed v1 and v2 10,000-S1 training
cohorts and their positive-enriched reduced target pools. The candidate lists
and pair features stayed fixed. It is not an untouched confirmation, full
corpus/France result, or Portal score.

Across the exposed v1, v2, and former-final cohorts, the existing rich
matcher rejected 373/329/340 true S2 candidate links and 494/380/432 true S3
candidate links respectively. That recurring source difference motivated a
separate XGBoost rich model per target source, each fitted only on one exposed
training cohort. We combined their opposite-cohort scores in original
candidate order. The decision thresholds, 0.700 for v1 → v2 and 0.825 for
v2 → v1, had been chosen by entity-disjoint crossfit of the **global** rich
model on the training cohort; neither test's labels selected a threshold.

| Exposed fit → test | Model | Overall per-S1 macro F0.5 | India | US | Overall TP / FP / FN |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 → v2 | Global rich | 98.7564% | 98.4724% | 98.9457% | 33,770 / 185 / 790 |
| v1 → v2 | Separate S2/S3 | 98.7539% | 98.5083% | 98.9177% | 33,757 / 186 / 803 |
| v2 → v1 | Global rich | 98.7959% | 98.6375% | 98.9035% | 33,512 / 133 / 959 |
| v2 → v1 | Separate S2/S3 | 98.7014% | 98.5914% | 98.7761% | 33,472 / 134 / 999 |

At the inherited global thresholds, source-specific fitting offered no
repeatable overall or India gain. The second direction lost 0.0945 overall
F0.5 percentage points, primarily by rejecting more true candidates. This
screen does not rule out source-specific calibration with thresholds selected
on separate training folds. **Do not promote this tested split.** The packaged
matcher and the previously confirmed 98.7707% reduced-pool result stay
unchanged. Complete country counts and candidate-pair counts are in
[source_specific_rich_screen.json](source_specific_rich_screen.json).
