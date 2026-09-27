# Frozen rich matcher: independent 10,000-record confirmation

The rich pair matcher and threshold 0.775 were frozen and pushed at commit `3f5c790` **before** scoring this cohort. On original training Source 1 rows 300,001–310,000, per-S1 macro F0.5 improved from **98.3538%** for the frozen baseline to **98.7707%** for the rich model. The paired gain is **0.4169 percentage points** (entity-bootstrap 95% interval +0.2939 to +0.5385 points). **It does not meet the 99.5% end goal.**

| Final matching metric | Frozen baseline | Frozen rich model | Difference |
| --- | ---: | ---: | ---: |
| Overall macro F0.5 (10,000 S1) | 98.3538% | **98.7707%** | +0.4169 pp |
| India macro F0.5 (4,087 S1) | 98.2791% | **98.5541%** | +0.2750 pp |
| US macro F0.5 (5,913 S1) | 98.4054% | **98.9204%** | +0.5151 pp |
| Overall matching TP / FP / FN | 33,984 / 374 / 846 | 33,952 / 174 / 878 | 200 fewer FP; 32 more FN |
| India matching TP / FP / FN | 13,734 / 141 / 404 | 13,727 / 78 / 411 | 63 fewer FP; 7 more FN |
| US matching TP / FP / FN | 20,250 / 233 / 442 | 20,225 / 96 / 467 | 137 fewer FP; 25 more FN |

Paired India gain interval: +0.0810 to +0.4712 points; US: +0.3675 to +0.6755 points. This confirms an improvement over the original matcher on this reduced-pool training sample, not full-corpus or Portal performance. The rich model still misses 878 true links, of which 106 were not candidates and 772 were retrieved but not selected. It also selects 174 false links. The remaining overall F0.5 gap to 99.5% is **0.7293 percentage points**.

The following is the **blocking candidate** confusion matrix. It is shared by the baseline and rich matcher because retrieval was unchanged. Candidate precision/F1 must not be confused with final matching precision/F0.5.

| Scope | Comparison space | Candidate TP | FN | FP | TN | Blocking recall | Candidate precision | Reduction ratio | Specificity | Candidate F1 | Candidates mean / p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 4,771,640,000 | 34,724 | 106 | 474,049 | 4,771,131,121 | 99.6957% | 6.8250% | 99.9893% | 99.9901% | 0.1278 | 50.88 / 87 / 106 |
| India | 1,950,169,268 | 14,073 | 65 | 243,626 | 1,949,911,504 | 99.5402% | 5.4610% | 99.9868% | 99.9875% | 0.1035 | 63.05 / 92 / 106 |
| US | 2,821,470,732 | 20,651 | 41 | 230,423 | 2,821,219,617 | 99.8019% | 8.2251% | 99.9911% | 99.9918% | 0.1520 | 42.46 / 80 / 97 |

Blocking recall clears 99.5% overall and by country on this batch, but India does so by only 0.0402 points. The perfect-decision candidate oracle macro F0.5 is 99.9257% overall, 99.8934% India, and 99.9480% US. Those are upper-bound diagnostics, not model results.

## Scope, integrity, and timing

- This new 10,000-S1 batch had zero known target-ID and raw-signature overlap with exposed batches at selection. It contains 4,087 India and 5,913 US records. The target pool was the previous 443,023-target reduced pool plus every new batch known true target, yielding **477,164**. Truth was used to ensure target-pool coverage, **not** to train, select, or tune the frozen matcher. This positive-enriched pool omits most 10.32-million training target records and may be optimistic. There is no France, full-target-corpus, or Portal score here.
- The six-worker frozen baseline retrieval and scoring pass took **1,910.64 seconds** locally and produced 508,773 candidate pairs. Exporting exact baseline pair scores took **239.75 seconds** and reproduced the baseline matching TSV SHA-256 exactly. Label-blind full-training-target core-frequency scan took **73.52 seconds**; rich feature export took **244.50 seconds**; prediction from those features took **0.63 seconds**. These are stage timings in this validation workflow, not a measured production speedup. Comparable peak memory was not measured.
- The baseline match hash was `f6d780ef14a590cbd6cbf1711abcecb42d8da105ac6b22350223d9aaa918a1d2`; the rich-model match hash was `b200fb68f49167599a9c2a47d5b1d3bc0adabd2662b0c2872d72c7f9b800d968`. Both used candidate hash `b8b0f5b80a2fd85c70dfe14f9ffd61433c8b830c468c71402716e8075cff38e0`.
- The later packaged `src/rich_infer.py` replayed all 10,000 rows from the cached candidates and the complete unlabeled training target files. It reproduced **both** candidate and changed-match hashes exactly, and verified the original baseline decisions row by row. Its complete rescore took **302.48 seconds** with **1.86 GB peak RSS** on this machine. A separate 500-row shard reproduced the first 500 rows of both complete TSVs exactly. The challenge validator passed row coverage, output format, and match-subset checks on the 10,000-row replay; its optional full-target ID scan was not run because the scorer already checked every candidate ID against the evaluation index.
- Full paired scores, counts, and intervals: [rich_pair_fresh_confirmation_scores.json](rich_pair_fresh_confirmation_scores.json). The committed [development screen](rich_pair_development_screen.md) and checkpoint manifest document selection before confirmation.

## Decision

The richer matcher is a validated research improvement but remains below the required final F0.5. The default `infer.py run` still uses the original frozen matcher; an optional, self-contained `src/rich_infer.py` bounded-shard command now reproduces this confirmation's matching and candidate TSVs byte-for-byte after candidate generation. This software replay does not turn the reduced-pool result into a full-test, France, or Portal score. Further work needs new discriminative evidence for ambiguous names and missing addresses, with a new disjoint confirmation set if any method is changed after seeing these errors. Faster execution alone will not raise F0.5.
