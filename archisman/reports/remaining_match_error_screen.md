# Remaining matcher-error development screen

These checks use only the already exposed original-training S1 cohorts v1 and
v2, 10,000 queries each, with their positive-enriched reduced target pools.
The existing candidate lists were fixed. They do not establish performance on
an untouched holdout, the full corpus, France, or the Portal. The final 10k
confirmation cohort was not used to select either variant.

## Missing target addresses

On the previously confirmed rich matcher, 363 of 878 rejected true candidate
pairs on the final confirmation cohort had an empty target address. This
motivated a specialist trained only on missing-target-address pairs. The gate
uses the target-address flag available at inference time. We fitted an XGBoost
global model and specialist on one exposed cohort, selected thresholds using
two entity-disjoint crossfit folds from that training cohort, and evaluated
on the other exposed cohort. We repeated this in reverse.

| Exposed train → test | Global overall / India / US macro F0.5 | Missing-address specialist overall / India / US |
| --- | --- | --- |
| v1 → v2 | 98.7564% / 98.4724% / 98.9457% | 98.7564% / 98.4747% / 98.9443% |
| v2 → v1 | 98.7959% / 98.6375% / 98.9035% | 98.8366% / 98.6383% / 98.9713% |

A separate threshold for missing-address pairs, without a specialist model,
tied the global result in the first direction and decreased it to 98.7884%
overall in the second. The specialist gain is inconsistent and too small to
justify changing the frozen matcher. Saved values and selected thresholds:
[missing_address_rich_screen.json](missing_address_rich_screen.json).

## Target source and text collisions

Adding a binary S2/S3 origin feature to the same rich feature matrix gave
98.7577% overall / 98.4915% India for v1 → v2, compared with 98.7564% /
98.4724% for the rich baseline. In reverse it gave 98.8258% overall /
98.6504% India, compared with 98.7959% / 98.6375%. The origin is available
from candidate IDs at inference time, but these development gains are small.
The model, thresholds, and full country/source counts are in
[target_source_context_screen.json](target_source_context_screen.json).

## More of the same training pairs

We also held the rich model architecture and decision threshold (0.775)
fixed, fitted it on one exposed cohort, then fitted it on that cohort plus a
second disjoint exposed cohort. Each model was evaluated on the remaining
third cohort. Training volume roughly doubled from 0.50 million to 1.00
million candidate pairs; no candidate retrieval or feature recipe changed.

| Exposed test cohort | One-cohort overall / India macro F0.5 | Two-cohort overall / India |
| --- | --- | --- |
| v1 | 98.8298% / 98.6712% | 98.8449% / 98.7162% |
| v2 | 98.7292% / 98.4355% | 98.7537% / 98.5000% |
| final | 98.7643% / 98.5949% | 98.7707% / 98.5541% |

The pooled model improves overall by only 0.0064–0.0245 points, and India's
result falls on the final cohort. The final row reproduces the earlier frozen
rich score at this fixed threshold; it is **not** a new untouched
confirmation. Full metrics: [rich_training_scale_screen.json](rich_training_scale_screen.json).

To test whether local text collisions force unavoidable errors, we grouped
each query's candidates by exact `(target name, address, country)` and by the
same fields after the existing normalization. In each cohort of roughly
500,000 candidate pairs, **zero** query groups had mixed true/false labels
for byte-identical raw target records. Only **three** query groups in each
cohort had a mixed-label normalized-text class (one true and one false member
per class). This is a narrow diagnostic: other ambiguous but nonidentical
records remain, and it does not prove that labels or text features are
sufficient. Saved counts: [candidate_text_collision_screen.json](candidate_text_collision_screen.json).

## Decision

Do not promote either variant or revise the 98.7707% frozen confirmation
claim. The 99.5% goal remains unmet. The narrow fixes and extra examples here
do not bridge the gap; further improvement needs stronger discriminative
evidence for changed names and address-poor records, evaluated on a newly
reserved disjoint cohort before any quality claim. No new cloud job or Modal
spend was used.
