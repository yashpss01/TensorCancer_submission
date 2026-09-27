# Matcher model-family screen (development only)

The existing 30,000 Source-1 training groups and disjoint 10,000-group
development split were used to compare model families on the **same 85 numeric
features and 493,573 development candidate pairs**. The 85 features are the
original pair features, extra text features, and out-of-fold group features
excluding the duplicated first-stage probability. No candidate list or
normalization rule changed. Early stopping and the per-S1 macro-F0.5 threshold
were selected on this already exposed development split. These results do not
replace the fresh 10,000-group validation required for a chosen change.

| Model | Overall macro F0.5 | India | US | Pair precision | Pair recall | Fit time | Scoring time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing XGBoost group model | 98.2859% | 98.0817% | 98.4230% | 99.1793% | 97.2599% | 36.3 s | Previously measured |
| LightGBM 4.7.0 | 98.2587% | 98.1230% | 98.3499% | 99.1207% | 97.2803% | 29.6 s | 2.29 s |
| CatBoost 1.2.10 | 98.2657% | 98.0780% | 98.3917% | 98.8682% | 97.7249% | 101.5 s | 0.24 s |
| Existing XGBoost group/augmented blend | **98.3658%** | 98.1085% | 98.5386% | 99.0936% | 97.5244% | Previously measured | Previously measured |

Pair precision and recall are diagnostics; the competition objective is the
per-S1 macro F0.5 column. Training times include array loading for the new
models and therefore are not an exact training-only comparison. New-model peak
RSS was 1.56 GB for LightGBM and 2.05 GB for CatBoost. Both were trained on
1,488,094 pairs. CatBoost used 758 best iterations; LightGBM used 506.

A bounded ensemble ablation mixed each alternative into the existing blend at
25% and 50% weights, with thresholds tuned only on development. None exceeded
the existing blend's 98.3658% overall macro F0.5. The best alternative blend
was 25% LightGBM at 98.3653%, a negligible difference. CatBoost at 50% raised
India to 98.1865% but lowered overall to 98.3524% and US to 98.4639%.

**Decision:** retain the frozen XGBoost blend as the quality reference. A
library swap or simple ensemble is not the structural improvement needed for
the 99.5% goal. Preserve the next untouched labeled batch for a candidate,
feature, or decision-rule change with a meaningful development signal. Test
CatBoost-specific categorical or text features only as a separately scoped
ablation, because adding them would no longer be an equal-feature comparison.

Reproduction script: `experiments/model_family_comparison.py`. The source
feature arrays and baseline predictions are the saved artifacts under
`student_resource/artifacts/matching_round5/`. Experimental models and detailed
threshold grids remain in a local temporary directory; no production model
bundle or inference code was changed.
