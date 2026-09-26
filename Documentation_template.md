# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TensorCancer
**Branch Owner:** Archisman
**Status:** Frozen training-holdout result; test inference and Portal scoring pending
**Date:** 26 September 2026

---

## 1. Executive Summary

This branch records a text-only, multi-route candidate blocker followed by a blended XGBoost matcher. The best frozen configuration scored **98.4117% per-Source-1 macro F0.5** on 20,000 previously untouched labeled training entities in a reduced target pool. The requested 99.5% target was **not met**; there is no Portal score or measured France score.

## 2. Methodology

### 2.1 Problem Analysis

Names vary through spelling, script, word order, legal suffixes, and abbreviations. Addresses may be truncated, missing, or inconsistent in house numbers. Generic business names create hard negatives. The original labeled training data covers India and the US; the unlabeled test adds France. Country is not used as a hard blocker in the evaluated retrieval route.

### 2.2 Solution Strategy

The blocker combines SQLite FTS5 searches over normalized names, addresses, character grams, and phonetic names. A second bounded retrieval pass uses address and name-bigram routes and limited anchor expansion. The frozen selection retains the baseline candidates plus rescues above the chosen score cutoff, capped per route. Candidate pairs receive name/address overlap and numeric features, extra typo/character-IDF features, and group-context features. Three trained XGBoost models are used at inference: a first-stage pair scorer for group context, a group-aware scorer, and an augmented pair scorer. The final probability is `0.4 * group + 0.6 * augmented`; pairs at or above `0.74` are accepted. These weights and the threshold were frozen on development validation before the untouched holdout.

## 3. Candidate Generation (Blocking)

The final 20,000-entity holdout search used an **enriched, reduced pool of 409,141 targets**, containing all known positives for the evaluated entities plus a deterministic 2% distractor sample. It produced **991,322 candidates** (49.57 per S1) and retained **69,171 of 69,330** true links: **99.7707% recall** overall, 99.6437% India, and 99.8570% US. These are blocking figures, separate from final matching F0.5. The reduced pool cannot establish full-corpus blocking recall.

## 4. Matching Model

The pairwise and group-aware features are implemented under `code/business_entity_resolution/round5/`. `artifacts/matching_round5/holdout/frozen_model.json` records the frozen model files, hashes, blend weights, and threshold. The model weights are committed under `artifacts/matching_round5/` at the paths used in that record. Model choice and threshold came from 10,000 development entities; no holdout labels were used to fit or choose them.

## 5. Results & Error Analysis

| Labeled split | S1 entities | Macro F0.5 | India | US |
| --- | ---: | ---: | ---: | ---: |
| Development validation | 10,000 | 98.3658% | 98.1085% | 98.5386% |
| Untouched holdout, S1 rows 60,001–80,000 | 20,000 | **98.4117%** | **98.3497%** | **98.4536%** |

On holdout the matcher predicted 67,723 true links and 637 false links, and missed 1,607 true links. Of those misses, 159 were absent from the candidate list and 1,448 were retrieved but rejected by the matcher. Pair precision was 99.0682%; pair recall was 97.6821%. Most residual error is therefore in the matching decision, especially cases with missing addresses, cross-script names, and similar-looking but distinct businesses. See `reports/matching_round5_holdout_review.md` and `artifacts/matching_round5/holdout/results.json` for the full measured result and caveats.

## 6. Conclusion

The 99.5% macro-F0.5 target remains unmet. The frozen model and holdout evidence are preserved for review. A hard-case reranker is the proposed next experiment, but it has not been trained or evaluated in this branch. A self-contained test inference entry point is included; full test inference, France accuracy, and the Portal score have not been measured.

## Appendix

### A. Code Artefacts

- `code/business_entity_resolution/src/`: initial blocker, reporting code, and self-contained `infer.py` with its inference feature/retrieval modules.
- `code/business_entity_resolution/models/`: three frozen XGBoost models, IDF weights, baseline selection config, and SHA-256 manifest.
- `code/business_entity_resolution/round2/` through `round4/`: blocker improvements and independent validation rounds.
- `code/business_entity_resolution/round5/`: pair features, matcher training, group features, and frozen holdout evaluation.
- `artifacts/matching_round5/`: frozen model weights and compact evidence files. Raw challenge data, multi-gigabyte intermediate arrays, and the reduced SQLite index are not in Git.
- `reports/`: protocols, blocker evaluations, and the final matcher holdout review.

### B. Delivery Boundary

`output/matching_results.tsv` and `output/candidate_pairs.tsv` are required for Portal delivery. They are not present yet. The team lead can run the commands in `code/business_entity_resolution/README.md`, validate TSV format and coverage, then package the code, models, documentation, and generated outputs. A tiny inference fixture passed the challenge validator, including ID checks; this is a software check, not a performance score. The saved holdout score must not be represented as the Portal score.
