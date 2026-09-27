# Local multilingual reranker fine-tuning: development screen

This experiment uses **already exposed** original-training S1 cohorts v1 and
v2, each with a positive-enriched reduced target pool. The candidate lists
were fixed. It is not an untouched confirmation, full-corpus or France result,
Portal score, or new claim about the required per-S1 macro F0.5 target.

The base model was
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Its model card
lists Apache-2.0. The downloaded safetensors file contains 567,755,777
parameters, below the challenge's 8-billion-parameter limit. All training and
scoring ran locally on Apple MPS with only the provided training records; no
business identity was sent to an external lookup service, and no Modal job or
new cloud spend was used. This exploratory model was **not** put into the
submission pipeline.

For each direction, one exposed cohort supplied a stable-hash-selected,
balanced training set of 5,000 true and 5,000 false candidate pairs with
frozen score at least 0.01. We fine-tuned the final four encoder layers and
classification head for one epoch (51,435,521 trainable parameters), then
scored the opposite cohort without passing labels into the scoring process.
The pair sample was selected before inspecting its labels: 1,000 candidates
with rich score between 0.05 and 0.95. These pair metrics are **not** the
competition metric.

| Train → exposed test | Pair sample | Rich AUC | Fine-tuned AUC | India rich → fine-tuned AUC | US rich → fine-tuned AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 → v2 | 1,000 | 0.8925 | 0.8531 | 0.9001 → 0.9063 | 0.8864 → 0.8014 |
| v2 → v1 | 1,000 | 0.8759 | 0.8407 | 0.8703 → 0.8790 | 0.8831 → 0.8024 |

The small India-only pair AUC edge repeated, while overall and US AUC were
lower. To test actual entity decisions, we scored **all** rich-uncertain pairs
on each opposite cohort (3,701 and 3,554 pairs out of 500,128 and 503,552
total candidates). Other candidate scores remained exactly as before. We
blended the rich probability with sigmoid of the neural logit, using a
rich-model decision threshold selected on the *training* cohort: 0.700 for
v1 → v2 and 0.825 for v2 → v1. The identity-score control reproduced the
existing rich baseline exactly before evaluating real neural scores.

| Train → exposed test | Blend weight | Overall macro F0.5 | India | US |
| --- | ---: | ---: | ---: | ---: |
| v1 → v2 | Rich only | 98.7564% | 98.4724% | 98.9457% |
| v1 → v2 | 0.10 neural | 98.7905% | 98.5112% | 98.9767% |
| v1 → v2 | 0.30 neural | 98.7808% | 98.6007% | 98.9008% |
| v2 → v1 | Rich only | 98.7959% | 98.6375% | 98.9035% |
| v2 → v1 | 0.05 neural | 98.7910% | 98.6239% | 98.9045% |
| v2 → v1 | 0.10 neural | 98.7405% | 98.5877% | 98.8443% |

The v1 → v2 +0.0341-percentage-point gain at weight 0.10 has a paired
entity-bootstrap 95% interval of **−0.0019 to +0.0761 points**. All tested
weights reduced overall F0.5 in reverse, and even the 0.05 weight lowered
India there. The sample AUC improvement for India does not translate into a
repeatable country-level macro-F0.5 gain. Weight comparisons on exposed data
are also selection-optimistic.

**Decision:** reject this fine-tuned reranker/blend for promotion. Keep the
frozen rich model and its previously confirmed 98.7707% reduced-pool score.
Any different architecture or selected decision rule requires a new disjoint
confirmation cohort before a performance claim. Full numeric counts and
paired intervals are in
[finetuned_reranker_development_screen.json](finetuned_reranker_development_screen.json).
