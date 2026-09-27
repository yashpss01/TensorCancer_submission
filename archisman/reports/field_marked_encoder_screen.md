# Field-marked multilingual entity encoder: bounded development screen

This experiment is on the **already exposed** original-training S1 v1 → v2
cohorts and their positive-enriched reduced target pools. It is not a fresh
confirmation, full-corpus/France result, Portal score, or submission model
change. The candidate list was fixed at 500,128 pairs for 10,000 v2 queries.

Inspired by [Ditto's entity-matching serialization](https://github.com/megagonlabs/ditto),
we marked business-name, address, and country fields using `[COL]` and
`[VAL]`, then fine-tuned the whole multilingual
[DistilBERT base cased model](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
as a pair classifier. The publisher's model card lists Apache-2.0 and 134M
base parameters; after adding field tokens and a classifier our local model
had **135,326,977 trainable parameters**, below the challenge's 8B limit.
We used its pinned revision `45c032ab32cc946ad88a166f7cb282f58c753c2e`.
This is a bounded test of field-marked full-encoder training, **not a full
reimplementation of Ditto** or its data augmentation.

Only provided labeled training rows were used. From exposed v1, a stable
pair-hash selected 5,000 positive and 5,000 negative hard candidate pairs.
The model trained for two epochs locally on Apple MPS. It then scored every
v2 pair where a separate v1-fitted rich matcher gave probability 0.05–0.95:
3,701 uncertain pairs. The scoring process saw no v2 truth labels. The
evaluation script restored the scores at those exact candidate indices and
kept all other rich scores unchanged. The rich threshold of 0.700 had been
selected from training-entity crossfit, before v2 evaluation. No challenge
records were sent to external lookup services, and no Modal job or cloud
spend was used.

| v1 → exposed v2 | Overall per-S1 macro F0.5 | India | US |
| --- | ---: | ---: | ---: |
| Rich only | 98.7564% | 98.4724% | 98.9457% |
| 0.05 neural blend | 98.7675% | 98.5179% | 98.9340% |
| 0.10 neural blend | 98.7658% | 98.5079% | 98.9377% |
| 0.30 neural blend | 98.5035% | 98.2710% | 98.6585% |

The largest observed overall gain was **+0.0112 percentage points** at
weight 0.05; its paired entity-bootstrap 95% interval was **−0.0271 to
+0.0527 points**. Heavier neural weights damaged both countries. This is
selection on an exposed cohort and has no meaningful repeatable advantage
to justify the cost of a reverse or fresh run.

**Decision:** stop this route at the bounded screen; do not promote the model
or alter `rich_infer.py` or default `infer.py run`. The previously frozen
98.7707% reduced-pool confirmation remains the best confirmed matcher score,
below the 99.5% goal. Full counts and tested weights are in
[field_marked_encoder_screen.json](field_marked_encoder_screen.json).
