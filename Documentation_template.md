# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TensorCancer  
**Team Members:** Akanksha, Yash  
**Submission Date:** 26 September 2026

---

## 1. Executive Summary

We resolve Source-2/3 records to Source-1 entities with a three-part pipeline: (1) rule-based, language-agnostic normalisation of names and addresses, including a token-level Indic-script → English transliteration dictionary learned only from the training pairs; (2) scalable candidate generation with four word-level TF-IDF indexes per country queried by sparse matrix products, trimmed by a transparent rule to ~10 candidates per Source-2/3 record; (3) a two-stage LightGBM matcher (pairwise features, then "competition" features that compare each candidate with the other candidates of the same record and the same entity), followed by a one-to-one assignment and a per-entity decision rule that maximises the expected F0.5. Key innovations are the learned transliteration dictionary (raises Indic-name blocking recall from 92% to 99%), the joint name+address bigram index, and the expected-F0.5 decision that treats singletons and multi-match entities differently.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the exploratory analysis of the training data (2.2M Source-1, 5.0M Source-2, 5.3M Source-3 records):

* **Structure.** Every Source-2/3 record matches at most one Source-1 entity (7.64M matched records, all distinct); 26% of Source-2/3 records are distractors with no Source-1 counterpart; 5.6% of Source-1 entities are singletons; matched entities have on average 3.46 records (up to 11). Country never differs between a record and its entity.
* **Names are not unique.** Only 46% of Indian Source-1 entities have a unique normalised name; the vocabulary of business-name words is small (~50k distinct tokens, ~1.3k distinct Indic tokens), so many entities share a name and the address is the discriminator. 38% of singletons share an exact name with some Source-2/3 record.
* **Script.** 18% of the Indian matched records carry the business name in an Indic script (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Punjabi, Malayalam, Odia). Their addresses are in Latin script but frequently truncated.
* **Name noise.** Case, accents ("Ínc", "Cénter"), legal-suffix variants (Pvt/Private, Ltd/Limited, `[LLC]`, `(ID: 59206)`), dropped or added filler words (Center, Services, Dr, Smt), word-order shuffles ("Parr, Evvy Center"), typos (0 for O, transposed letters), domain-ification ("Summitanimalhospital.Com"), hashtags ("#4443ocean").
* **Address noise.** Abbreviations (Rd/Road, Ave), state name vs. abbreviation (Iowa/IA, Telangana/TG, Indic-script state names), zero-padded or altered house numbers ("001375", "275"→"27"), "City" suffixes, component re-ordering, dropped components (no PIN, no street), "Door No / H.No / ##" prefixes, `<NULL>` / `N/A` tokens, 3.3% of matched records have an empty address.
* **Distractors** are independent businesses, not near-copies of Source-1 entities (0.2% share an exact address with a Source-1 entity), so false merges come from generic names, which addresses must reject.
* **Test set** adds France (15% of Source-1 entities) and a new "small shop" name distribution (Motors, Stores, Bakery, …) that is absent from the training vocabulary, so nothing may be hard-coded to the training countries or words.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + global assignment  
**Core Innovation:** learned token transliteration dictionary; joint name+address TF-IDF bigram index with rule-based trimming; competition features + expected-F0.5 decision.

Pipeline (all steps run per country label, whatever the label is):

1. `normalize.py` — one normalised view set per record: `name_full`, `name_core` (legal words removed), `name_sig` (filler words removed), `name_nospace`, `name_sorted`, legal-form set, phonetic skeleton; `addr_full`, `addr_alpha` (words without state), `addr_nums`, first house number, detected state, empty flag. Indic tokens are mapped through the learned dictionary (`translit.py`) with a small rule-based fallback for trade words; the rest is transliterated with `unidecode`.
2. `blocking.py` — candidate generation (Section 3).
3. `features.py` + `pipeline_core.py` — 66 pairwise features, stage-1 LightGBM, context features, stage-2 LightGBM.
4. Decision — one-to-one assignment on the Source-2/3 side, then per-entity expected-F0.5 subset selection.

---

## 3. Candidate Generation (Blocking)

Every Source-1 record is indexed per country with four word-level TF-IDF vectorisers (sub-linear tf, `max_df` pruning of very common tokens for speed). Every Source-2/3 record of the same country is compared against the whole index with a sparse matrix product (`sparse_dot_topn`), keeping the top-k cosine neighbours of each index:

| index | document | k |
| --- | --- | --- |
| joint | `name_core` + `addr_full`, word uni- and bigrams | 30 |
| jskel | phonetic skeleton of the name + `addr_full` | 15 |
| fullskel | skeletons of name and address words + numbers | 10 |
| name | `name_core` only (rescues empty-address records) | 20 |

The union of the four retrievals is scored with a transparent heuristic `h = max(cos_joint, cos_jskel, cos_fullskel) + 0.25·cos_name` and trimmed to: the top-4 by `h`, plus any candidate within the top-40 that has an identical normalised name (or identical sorted name), the same leading house number, or a name-token Jaccard ≥ 0.5 together with a shared number or address-token Jaccard ≥ 0.3. Nothing in the trimming rule is learned.

- **Blocking keys used:** joint name+address TF-IDF cosine (uni+bigrams), phonetic-skeleton TF-IDF, name-only TF-IDF, exact-name / same-house-number / token-overlap rules.
- **Candidate pairs generated:** training replica 51.8M pairs for 5.16M records (10.0 per record, 46.9 per Source-1 entity, reduction ratio 99.9991%). Test: see Appendix B.
- **How we ensured true matches were not lost:** recall was measured on a 50% replica of the training world (same density of entities and distractors). Pair recall of the final candidate set is 98.38% (India 97.91%, US 98.70%); 96.5% of true matches are the heuristic's rank-1 candidate. The remaining misses are dominated by empty-address records with generic names (inherently ambiguous) and heavily truncated addresses.

Blocking confusion matrix on the training replica:

```
Source-1 entities: 1,103,410   Source-2/3 targets: 5,161,055
Total Comparison Space (S1 x Targets):      5,694,759,697,550
Candidate pairs generated (C):                     51,780,856
True Positives  (TP - Matches Retained):          3,758,225 ( 98.38%)
False Negatives (FN - Matches Missed):               61,904 (  1.62%)
False Positives (FP - Distractor Pairs):         48,022,631 ( 92.74%)
True Negatives  (TN - Pairs Pruned):      5,694,707,854,790 ( 99.9992%)
Recall 98.38% | Precision 7.26% | Reduction Ratio 99.9991% | Specificity 99.9992% | Candidate F1 0.1352
```

---

## 4. Matching Model

**Features used (66, stage 1):**
- Name features: token Jaccard/overlap/first-token equality on `name_core` and `name_sig`; rapidfuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio` on the core name, `ratio` on the full name and on the space-less name, Levenshtein distance (raw and length-normalised), phonetic-skeleton `ratio`/`token_set_ratio`, exact-equality flags (core, sorted, signature), legal-form agreement, non-Latin flag, lengths.
- Address features: token Jaccard/overlap on `addr_alpha`, number-set Jaccard and first-number agreement, `ratio`/`token_sort`/`token_set`/`partial` on the address, number-string similarity, state agreement (same / different / unknown), empty-address flags, lengths.
- Other: the four blocker cosines, heuristic score, heuristic rank and gap to the best candidate, source flag (S2 vs S3).

**Context features (stage 2, 16 + 20 raw):** stage-1 probability and logit; for the same Source-2/3 record: max probability, ratio to max, rank, second-best, gap, candidate count; for the same Source-1 entity: number of other records that already claim it as their best match (total, from S2, from S3), number of strong candidates, mean probability of the other candidates, candidate count.

**Model type:** two LightGBM binary classifiers (255 leaves, lr 0.05, early stopping; ~700 and ~N trees) trained on disjoint Source-1 entity folds so that stage-2 never sees in-sample stage-1 probabilities. A record is assigned to its most probable entity only (one-to-one on the Source-2/3 side).

**Threshold selection method:** per Source-1 entity, the accepted subset of candidates is the prefix (in decreasing probability) that maximises the expected F0.5 under independence, computed exactly by dynamic programming; this is compared against fixed thresholds on the validation folds and the best rule is stored in `models/config.json`.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9824** (v2 models) on 331,029 held-out Source-1 entities of the training replica (entity folds 7-9, never used for fitting or early stopping); India 0.9804, US 0.9838. Pair-level: precision 99.52%, recall 95.85%; singleton accuracy 97.7%. Pairwise AUC 0.9946 (stage 1) → 0.9960 (stage 2), average precision 0.9992. Decision rules compared on the same entities (v2): fixed thresholds 0.3/0.5/0.7/0.8 give 0.9758/0.9807/0.9821/0.9818; the expected-F0.5 rule gives 0.9824; without the one-to-one assignment 0.9806. The v1 models (without name-frequency, tie and house-number-digit features) scored 0.9803. The stage-2 probabilities are calibrated (observed match rate 0.55 in the 0.5–0.6 bin, 0.85 in the 0.8–0.9 bin, 0.9975 above 0.9); isotonic re-calibration and power transforms did not improve the decision.

Loss attribution (1 − F0.5 summed over entities, share of the total 0.0197 macro loss):

| cause | entities | share of loss |
| --- | --- | --- |
| true match retained by blocking but rejected by the model | 29,068 | 46% |
| true match missed by blocking | 15,387 | 23% |
| false positive on a matched entity | 4,904 | 17% |
| false positive on a singleton (scores 0) | 517 | 8% |
| both FP and FN | 1,087 | 5% |
| perfect | 280,066 | — |

- **Common false positives (wrong merges):** (i) a record with an empty (or city-only) address whose name is shared by several Source-1 entities in different places; 62% of FPs are distractors with a colliding name, 38% are records assigned to the wrong same-name entity. The match rate of an exact-name, empty-address pair falls from 74% when the name is unique in Source 1 to 3.5% when ≥11 entities share it, so we added the Source-1 name-frequency and tie-count features in v2. (ii) house-number perturbations on otherwise identical addresses ("1831" vs "1835 Edgewater Drive") where the name is also slightly different, which are indistinguishable from the noise applied to true matches.
- **Common false negatives (missed matches):** (i) empty-address records with a generic name (46% of the model's misses have an empty address); (ii) true matches whose house number was truncated or altered *and* whose name carries a typo ("legacy lfmdhn inc | 61 chandler lane" vs "legacy land inc | 1061 chandler lane"); (iii) blocking misses are 60% empty-address records with a typo in the name or a domain-style concatenated name ("coraliestextiles"), which word-level indexes cannot retrieve — addressed by the optional character-trigram pass (`blocking_extra.py`).

---

## 6. Conclusion

_[filled at the end]_

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`: `normalize.py`, `translit.py`, `preprocess.py`, `make_train_subset.py`, `blocking.py`, `blocking_report.py`, `features.py`, `pipeline_core.py`, `train.py`, `predict.py`; `models/` holds the two LightGBM boosters and `config.json`. `README.md` lists the exact commands (data → transliteration dictionary → normalisation → blocking → training → prediction → validation).

### B. Additional Results

_[test-set candidate statistics, per-country validation scores]_
