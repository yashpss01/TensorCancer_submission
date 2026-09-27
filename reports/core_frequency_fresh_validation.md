# Core-frequency matcher: fresh 10,000-record validation

The six-feature core-frequency matcher was frozen at commit `7cd44c0` before scoring the fresh cohort. On original training Source 1 rows 290,001–300,000, per-S1 macro F0.5 rose from **98.3108% to 98.6193%**. The paired gain was **0.3085 percentage points** (entity-bootstrap 95% interval: +0.1931 to +0.4280 points). This is a measured improvement, but **does not meet the 99.5% objective**. Both models used exactly the same 500,128 candidate pairs; blocking quality was unchanged.

| Final matching metric | Frozen baseline | Core-frequency model | Change |
| --- | ---: | ---: | ---: |
| Overall macro F0.5 (10,000 S1) | 98.3108% | 98.6193% | +0.3085 pp |
| India macro F0.5 (4,001 S1) | 98.1273% | 98.4423% | +0.3150 pp |
| US macro F0.5 (5,999 S1) | 98.4332% | 98.7374% | +0.3042 pp |
| Overall matching TP / FP / FN | 33,717 / 354 / 843 | 33,558 / 181 / 1,002 | 173 fewer FP; 159 more FN |
| India matching TP / FP / FN | 13,519 / 139 / 410 | 13,435 / 77 / 494 | 62 fewer FP; 84 more FN |
| US matching TP / FP / FN | 20,198 / 215 / 433 | 20,123 / 104 / 508 | 111 fewer FP; 75 more FN |

The India paired-bootstrap interval for the F0.5 gain was +0.1329 to +0.5106 points; the US interval was +0.1591 to +0.4611 points. These intervals estimate sampling variation for this fixed method and reduced pool, not performance on unseen countries or the full target corpus.

The teammate-format **blocking** matrix is below. These values describe candidate generation, **not** final matching decisions. Both matchers share them.

| Scope | Comparison space | Candidate TP | FN | FP | TN | Blocking recall | Candidate precision | Reduction ratio | Specificity | Candidate F1 | Candidates mean / p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 4,430,230,000 | 34,479 | 81 | 465,649 | 4,429,729,791 | 99.7656% | 6.8940% | 99.9887% | 99.9895% | 0.1290 | 50.01 / 87 / 114 |
| India | 1,772,535,023 | 13,883 | 46 | 235,483 | 1,772,285,611 | 99.6698% | 5.5673% | 99.9859% | 99.9867% | 0.1055 | 62.33 / 91 / 114 |
| US | 2,657,694,977 | 20,596 | 35 | 230,166 | 2,657,444,180 | 99.8304% | 8.2134% | 99.9906% | 99.9913% | 0.1518 | 41.80 / 79 / 101 |

There were 500,128 selected pairs. The blocker exceeds 99.5% recall overall and in both countries on this batch. Its perfect-decision macro F0.5 ceiling is 99.9311% overall, 99.8824% India, and 99.9635% US; the matcher remains the primary quality bottleneck.

## Design, timing, and integrity

- Model selection and training used the already exposed fresh-v1 cohort with entity-disjoint development folds. The fixed XGBoost checkpoint uses the frozen model probability, normalized core-name equality, empty-target-address flag, India flag, and global source/target core-name frequencies. Threshold 0.70 and model hash `394b603c915227024142f51aa6230661bd55dc7aa44cbacf16a319db76586af0` were committed before fresh-v2 truth was scored.
- The fresh cohort contained no S1 IDs, known target IDs, or raw signatures shared with the exposed batches. The pre-scoring selection manifest is in ignored `work/fresh_10k_v2/manifest.json`. The target pool was the old 409,141-target sealed pool plus the new batch's known true targets, yielding 443,023 targets. Adding positives ensures truth coverage but makes this a **positive-enriched reduced-pool** test. It does not establish full 10.32-million-target, France, or Portal performance.
- The frozen baseline retrieval and original matching run took 1,174.94 s with six local workers. Exporting exact pair probabilities from its candidate TSV took 220.50 s and reproduced the baseline matching SHA-256 exactly. Label-blind core frequencies scanned all 10,320,219 training target rows in 57.88 s; changed-model prediction from cached features took 0.335 s. These are stage timings, not an additive clean end-to-end benchmark for the new method, and memory was not measured comparably.
- All score fields and paired intervals are saved in [core_frequency_fresh_v2_scores.json](core_frequency_fresh_v2_scores.json). The baseline and changed matching SHA-256 hashes are `360a40f8b777a2cc88cc620825acd347a743c36d6299c9921070986be24f0b46` and `62fc29f7a70a9dac8cc6e053db90f6ccdf493de49ff38132d4763c1fd8f19a7e`, respectively. Their candidate TSV is byte-identical.

## Remaining error pattern and decision

Of the changed method's 1,002 missed true links, 81 were missed by retrieval and 921 were retrieved but rejected by the matcher. The matcher produced 181 false accepts. Among rejected true candidates, 484 had an empty target address, 140 had a Unicode-status mismatch, and 133 had an address-number conflict; these strata overlap. False accepts also concentrate in sparse evidence: 79 had an empty target address and 67 an exact normalized core name. The current classifier is trading false positives for false negatives. Fixing only either error class is insufficient: the diagnostic F0.5 would be 99.0914% if all FP vanished but FN remained, and 99.4634% if all retrieved positives were recovered but FP remained. These are retrospective ceilings, not achievable model scores.

Keep this model as a research checkpoint. Next, test a structural method that supplies stronger evidence for ambiguous names and missing addresses, using only exposed data for selection. Freeze its exact implementation before using another disjoint confirmation cohort. Do not present this reduced-pool result as the 99.5% end goal or submit it as a full-test result.
