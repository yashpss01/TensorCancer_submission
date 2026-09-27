# Pipeline bottleneck diagnosis — development evidence

This report analyzes **per-Source-1 macro F0.5**, the matching objective. Its
10,000-business development cohort is already exposed; the candidate pool
contains 409,141 targets, not all 10.32 million training targets. The frozen
pipeline and an experimental reranker were compared separately on a new,
disjoint 10,000-business batch, reported below. Nothing here is a Portal or
France score.

## Where the score is lost

| Stage | Result on exposed development cohort |
| --- | ---: |
| True links | 34,416 |
| True links offered as candidates | 34,322 |
| Candidate misses | 94 |
| Best possible macro F0.5 with these candidates and perfect decisions | 99.8794% |
| Frozen matcher macro F0.5 | 98.3658% |
| Final TP / FP / FN | 33,564 / 307 / 852 |

For 10,000 Source-1 businesses, a perfect score has 10,000 entity-score units.
The observed pipeline loses 163.42 units. Candidate retrieval accounts for
12.06 (7.4% of the loss); the decisions after retrieval account for 151.36
(92.6%). The matcher rejected 758 true candidate pairs, and 285 businesses had
at least one false acceptance. A ≥99.5% score allows at most 50 lost units on
this cohort, requiring roughly 69% less total loss than the current pipeline.
This is why another broad retrieval scan alone would not solve the measured
development gap. The candidate oracle is only a ceiling on this reduced pool;
full-corpus distractors can make the actual task harder.

India scored 98.1085% (54 retrieval misses, 414 final missed links, 129 false
accepted links). The US scored 98.5386% (40 retrieval misses, 438 final missed
links, 178 false accepted links). The India gap is mainly in matcher decisions,
not candidate availability in this pool.

## Error concentrations

These categories **overlap** and their losses must not be added together.
Loss units are for all errors in affected groups; the pair counts are for
pairs with that condition. This is association, not causal attribution.
"Script mismatch" below is the model's Unicode-status mismatch feature, a
proxy rather than a verified language or transliteration classification.

| Condition | Affected groups | Matcher loss units | Rejected true candidates | False accepted pairs |
| --- | ---: | ---: | ---: | ---: |
| A true candidate has missing/semantically missing address evidence | 1,387 | 57.28 | 460 | 149 |
| A true candidate has Unicode-status mismatch | 3,154 | 43.70 | 113 | 41 |
| A true candidate has address-number/postcode conflict | 1,014 | 19.59 | 106 | 46 |
| Multiple candidates have the same normalized core name as the query | 6,025 | 91.78 | 146* | 133* |
| True singleton business | 584 | 11.00 | 0 | 11 |

*The pair counts in the exact-core-name row refer to exact-core-name candidate
pairs. The
missing-address condition is especially concentrated: 460 of the 758
retrieved-but-rejected true pairs, and 149 of 307 false accepted pairs, are
there. All 460 rejected true pairs in that condition had a genuinely empty
target address, not just a placeholder. Their median frozen probability was
0.538, below the 0.74 threshold. Lowering the threshold to 0.50 in this
condition would gain 264 true pairs but add 208 false accepts, an unfavorable
tradeoff for F0.5. On India alone, 1,766 groups contain a true candidate with a
Unicode-status mismatch; those groups have 30.59 of India's 68.68 matcher-loss
units, including errors unrelated to the mismatch. The categories overlap.

Exact raw name/address/country duplicates with mixed truth labels occurred in
**zero** candidate groups. Normalization created only five mixed-label groups,
containing three rejected true pairs and two false accepts. Indistinguishable
literal duplicates therefore explain very little of the present error.

## Tested alternatives

The fixed XGBoost group/augmented blend at threshold 0.74 scored 98.3658%.
The development threshold sweep peaked near 0.74: lowering it to 0.65 raised
false accepts from 307 to 447; raising it to 0.85 raised final misses from
852 to 1,288. Two-fold entity-disjoint cross-fitting found no gain from
thresholds conditional on missing address, weak anchor, number conflict, or
India/US.

A new XGBoost missing-address expert trained on old 30,000-group data scored
98.3786% overall / 98.1492% India after an exposed-development threshold
choice, versus 98.3658% / 98.1085% for the frozen model. It traded 13 fewer
false accepts for 43 additional missed links. A separate residual XGBoost
reranker, trained and evaluated in two entity-disjoint development folds, was
worse on its own. Blending 20% of its probability with the frozen score at
the **unchanged** 0.74 threshold gave 98.3888% overall / 98.1591% India,
trading 25 fewer false accepts for 62 additional missed links. This is a
small exploratory improvement, not evidence that the 99.5% target is met.

An equal-feature model-family screen already compared XGBoost, LightGBM,
CatBoost and their simple blends. None beat the frozen blend materially;
details are in [model_family_screen.md](model_family_screen.md). Previous
target-to-target rescue and simple group-context approaches were similarly
below the frozen blend. More training data, strings, and group features have
already produced diminishing returns. The next credible approach must add
**new, reliable discriminating evidence** for missing addresses and ambiguous
names rather than merely rerunning or retuning the same scores.

A bounded label-blind neighbor check found that simply borrowing an address
from another target with the same normalized core name is unlikely to help.
Among 460 rejected true pairs with an empty target address, 147 had a
same-core target with a nonempty address in the development pool and 112 had
address-token Dice similarity above 0.5 to the query. But among 149 false
accepts with an empty target address, 117 had such a neighbor and 93 cleared
the same similarity level. That signal is at least as common in the false
accepts. It should not be added as a naïve rescue rule.

## Paired fresh 10,000-business check

The first fresh batch is original training Source-1 rows 280,001–290,000:
10,000 businesses (4,045 India; 5,955 US), 34,471 true links and 566 true
singletons. Its bounded evaluation pool is the old sealed 409,141-target pool
plus every known positive for this batch: **442,904 targets** total. Retrieval
candidates and the two decision methods were fixed before opening this
batch's labels. The baseline and reranker used exactly the same 503,552
candidate pairs, averaging 50.36 per business (p95 87; maximum 106). The
frozen baseline took 1,452 seconds locally; paired rescoring took about four
minutes. Cloud replication completed with byte-identical matching TSVs and
identical quality scores. One India candidate row swapped a negative target,
so candidate TSVs were not byte-identical across platforms; true-link
coverage was unchanged.

| Final matching metric | Frozen baseline | Experimental 20% residual blend |
| --- | ---: | ---: |
| Overall per-S1 macro F0.5 | 98.4139% | **98.4989%** |
| India macro F0.5 | 98.2916% | **98.4338%** |
| US macro F0.5 | 98.4970% | **98.5432%** |
| Overall TP / FP / FN | 33,665 / 348 / 806 | 33,637 / 307 / 834 |
| India TP / FP / FN | 13,553 / 130 / 409 | 13,545 / 108 / 417 |
| US TP / FP / FN | 20,112 / 218 / 397 | 20,092 / 199 / 417 |

The blend gained 0.0851 percentage points overall and 0.1423 in India, with
41 fewer false accepts at the cost of 28 additional missed links. It still
misses the **99.5%** goal by about 1.001 percentage points overall and 1.066
points in India. The candidate oracle for this pool is 99.9179% overall:
retrieval missed 92 true links and accounts for 8.21 of the blend's 150.11
lost entity-score units. Matcher decisions account for the other 141.89
(94.5%). The pool deliberately omits most full-corpus negatives, so even this
fresh result is not a full-corpus guarantee.

Only 113 of 10,000 businesses changed score (63 improved, 50 worsened). A
deterministic paired business-level bootstrap with 3,000 resamples gives a
95% interval of **+0.033 to +0.140 percentage points** for the overall gain,
and **+0.065 to +0.235** for India. The US interval, **−0.018 to +0.115**,
includes no gain. These intervals describe sampling variation within this
fixed reduced-pool cohort; they do not cover full-corpus or France shift.

The fresh error pattern confirms the development diagnosis. Of 742 true
candidates rejected by the blend, 421 have an empty target address. Of its
307 false accepts, 182 also have an empty target address. In India, 85 of 358
rejected true candidates have a Unicode-status mismatch, versus 14 of 384 in
the US; these are overlapping categories and the Unicode flag is a proxy for
script difference. Thus empty target addresses are the clearest overall
weakness, while name-script evidence appears more important to India's gap;
the latter is an inference from a proxy and needs a direct ablation. This
batch is now exposed after inspection. Two further
disjoint 10,000-business batches remain sealed for later development and
final confirmation. The broad full-universe Modal scan remains stopped.
Two bounded batched-retrieval configurations were subsequently screened on
this same exposed batch and rejected because of reduced blocking recall;
see [batched_retrieval_architecture_screen.md](batched_retrieval_architecture_screen.md).
