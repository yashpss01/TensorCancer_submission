# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TensorCancer  
**Team Members:** Pushkar Sharma, Yash Pratap Singh Solanki, Archisman Choudhury, Akanksha Sharma  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source-2/3 records to Source-1 entities with a three-part pipeline: (1) rule-based, language-agnostic normalisation of names and addresses, including an Indic-script → English token dictionary learned only from the training pairs; (2) scalable candidate generation with four word-level TF-IDF indexes per country queried by sparse matrix products plus a character-trigram pass for hard records, trimmed by a transparent rule; (3) a two-stage LightGBM matcher — pairwise similarity features, then "competition" features that compare each candidate with the other candidates of the same record and of the same entity — followed by a one-to-one assignment and a per-entity decision that maximises the expected F0.5. Models are trained and validated on the full training set at its true density of decoys; the held-out macro F0.5 is 0.9842 (India 0.9822, US 0.9855).

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the exploratory analysis of the training data (2.2M Source-1, 5.0M Source-2, 5.3M Source-3 records):

* **Structure.** Every Source-2/3 record matches at most one Source-1 entity (7.64M matched records, all distinct); 26% of Source-2/3 records have no Source-1 counterpart; 5.6% of Source-1 entities are singletons; matched entities have 3.46 records on average (up to 11). The country label never differs between a record and its entity.
* **Decoys.** Most unmatched records are deliberate near-copies of a Source-1 entity: same street, similar name (often a filler word added, e.g. "… Group Group", "… Co"), house number shifted by a small offset. Among rank-1 false candidates in training, 86% (US) have a different house number and 63% of those differ by 3–21; among true matches only 12% have a different number and 77% of those differ by more than 100 (digit drops such as "1061" → "61", zero padding). The house number is therefore the key discriminator between a true record and a decoy.
* **Names are not unique.** Only 46% of Indian and 53% of US Source-1 entities have a unique normalised name; the name vocabulary is small, so the address decides most matches, and an empty-address record with a common name is inherently ambiguous (the true-match rate of an exact-name, empty-address pair falls from 74% when the name is unique to 3.5% when ≥ 11 entities share it).
* **Script.** 18% of Indian matched records carry the name in an Indic script (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Punjabi, Malayalam, Odia) while the Source-1 name is in Latin script.
* **Name noise.** Case, accents, legal-suffix variants (Pvt/Private, Ltd/Limited, `[LLC]`, `(ID: 59206)`), added or dropped filler words, word-order shuffles, typos (0/O), domain-style names ("summitanimalhospital.com"), hashtags, DBA / trade names (1.8% of true pairs have an unrelated brand-like name at the entity's address).
* **Address noise.** Abbreviations, state name vs. code (including Indic-script state names), zero-padded or altered house numbers, "City" suffixes, component re-ordering, dropped components, "Door No / H.No / ##" prefixes, `<NULL>` / `N/A`; 3.3% of matched records have an empty address.
* **Test set.** Adds France (15% of Source-1 entities, no training data). French Source-1 entities share an address far more often (11.8% share their exact address with another entity, groups of up to 101 at one address, vs. 4–5% in India/US), 86% of French house numbers are below 100, and the test has 5.75 Source-2/3 records per Source-1 entity versus 4.68 in training. India adds a "small shop" vocabulary (Motors, Stores, Bakery …) absent from training. Nothing in the pipeline is hard-coded to the training countries.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + global one-to-one assignment + expected-F0.5 decision  
**Core Innovation:** learned Indic token dictionary; joint name+address bigram TF-IDF blocking; decoy-aware house-number features; competition features computed over complete candidate groups with cross-fitted stage-1 scores; exact expected-F0.5 subset selection per entity.

Pipeline (every step runs per country label, whatever the label is):

1. `translit.py` learns `translit_dict.json` (1,347 Indic tokens) by positional alignment of matched training names; `normalize.py` produces the normalised views of every name and address (`preprocess.py` applies it to all files in parallel).
2. `blocking.py` + `blocking_extra.py` — candidate generation (Section 3).
3. `features.py` + `pipeline_core.py` — 80 pairwise features → stage-1 LightGBM → 56 context features → stage-2 LightGBM (`train_full.py`).
4. `predict.py` — one-to-one assignment on the Source-2/3 side, then per-entity expected-F0.5 subset selection; writes both submission files.

---

## 3. Candidate Generation (Blocking)

Every Source-1 record is indexed per country with four word-level TF-IDF vectorisers (sub-linear tf; very common tokens pruned at 5% document frequency, capped at 30,000 documents). Every Source-2/3 record of the same country is compared against the whole index with a sparse matrix product (`sparse_dot_topn`, multithreaded C++), keeping the top-k cosine neighbours of each index:

| index | document | k |
| --- | --- | --- |
| joint | `name_core` + `addr_full`, word uni- and bigrams | 30 |
| jskel | phonetic skeleton of the name + `addr_full` | 15 |
| fullskel | skeletons of name and address words + numbers | 10 |
| name | `name_core` only (rescues empty-address records) | 20 |

The union is scored with a transparent heuristic `h = max(cos_joint, cos_jskel, cos_fullskel) + 0.25·cos_name` and trimmed to the top-4 by `h` plus any candidate within the top-40 that has an identical normalised (or sorted) name, the same leading house number, or a name-token Jaccard ≥ 0.5 together with a shared number or address-token Jaccard ≥ 0.3. A second pass (`blocking_extra.py`) handles the ~10% of records with an empty address or a single-token name: a character-trigram TF-IDF index over space-less Source-1 names returns up to 10 extra candidates (cosine ≥ 0.3). Nothing in blocking is learned.

- **Blocking keys used:** joint name+address TF-IDF cosine (uni+bigrams), phonetic-skeleton TF-IDF, name-only TF-IDF, character-trigram name TF-IDF for hard records, exact-name / same-house-number / token-overlap rules.
- **Candidate pairs generated:** full training set 120.0M pairs incl. the extra pass (India 46.9M, 53.1 per entity; US 72.7M, 54.9 per entity); test 111.0M pairs (see Appendix B).
- **How we ensured true matches were not lost:** recall measured on the full training set: 98.06% (India) and 98.98% (US) of true pairs are retained; 96.5% of retained true matches are the heuristic's rank-1 candidate. Remaining misses are dominated by empty-address records with generic names and heavily truncated addresses.

Blocking confusion matrix (training replica, main pass only; the full-set numbers above include the extra pass):

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

**Stage-1 features (80):**
- Name: token Jaccard/overlap/first-token equality on `name_core` and `name_sig`; rapidfuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`; full-name and space-less-name ratios; Levenshtein distance (raw and normalised); phonetic-skeleton similarities; exact-equality flags; legal-form agreement; non-Latin flag; lengths; how many Source-1 entities share the candidate's name / signature.
- Address: token Jaccard/overlap on `addr_alpha`; number-set Jaccard; first-number agreement, digit Levenshtein, prefix/suffix relation, absolute offset, offset parity and ratio (decoy signature); address ratios (full, token-sort, token-set, partial); state agreement; empty flags; how many Source-1 entities share the candidate's address (co-location).
- Other: the four blocker cosines, heuristic score, tie-aware heuristic rank, gap to the best candidate, tie count, candidate count, extra-pass flag, source flag (S2 vs S3).

**Stage-2 context features (31 + 25 carried over):** stage-1 probability and logit; for the same record: max probability, ratio to max, rank, second-best, gap, tie count, number of strong candidates, number of same-name candidates, name-similarity gap and rank versus the other candidates (decisive at shared addresses); for the same Source-1 entity: records already claiming it (total / S2 / S3), strong candidates, mean probability of the others, and how many of its confident records agree or disagree with its house number (sibling consistency).

**Model type:** LightGBM binary classifiers (255 leaves, learning rate 0.05, early stopping, up to 2,500 trees). Entity folds: stage 1 on folds 0–3 (7.2M sampled pairs, 1.8M positives); two cross-fitted stage-1 models (folds 0–1 and 2–3) score each other's folds so that stage-2 context is always out-of-sample; context features are computed over the complete candidate group of every record, exactly as at inference; stage 2 on folds 4–5 with early stopping on fold 6; validation on folds 7–9 (662,061 entities, never used for fitting). MIT-licensed, a few tens of MB.

**Threshold selection method:** each record is first assigned to its most probable entity (one-to-one). Per entity, the accepted subset is the prefix (by decreasing probability) that maximises the expected F0.5 under independence, computed exactly by dynamic programming over the Poisson-binomial distribution of true matches (an empty prediction is worth P(no true match)). This beat every fixed threshold on validation (0.9842 vs. 0.9839 at the best threshold 0.7).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), held-out, full density:** **0.9842** (v5; India 0.9822, US 0.9855; pair precision 99.60%, recall 96.15%; singleton accuracy 97.8%; pair AUC 0.9950 → 0.9971 after stage 2). Earlier versions on the same protocol: v4 0.9839.
- **Public leaderboard:** v2 0.9655. The held-out validation of the same approach is ~0.98, so the test set behaves differently from the training distribution; see the diagnostics in Appendix B.

Loss attribution at full density (v4, macro loss 0.0161):

| cause | entities | share of loss |
| --- | --- | --- |
| entity found only partly (some records missed) | 77,038 | 57% |
| entity with records but nothing predicted | 1,823 | 17% |
| false positive on a matched entity | 6,531 | 14% |
| false positive on a singleton (scores 0) | 792 | 7% |
| both FP and FN | 1,353 | 4% |

Of the missed true pairs that reached the matcher, 54% have an empty address (mostly several Source-1 entities share the name), 25% have a different house number (probability ≈ 0.43, below the ≈ 0.75 needed by the F0.5-optimal rule when the entity already has confident matches), 9% are confident name-and-number agreements the model still doubts, 8% are DBA/trade-name records.

- **Common false positives (wrong merges):** decoys with a filler word added and a house number shifted by 3–21 when the true siblings are few; empty-address records assigned to one of several same-name entities.
- **Common false negatives (missed matches):** empty-address records with common names; true records whose house number was altered and whose name also carries a typo; records lost to a same-name entity.

---

## 6. Conclusion

A blocking + two-stage gradient-boosting pipeline with language-agnostic normalisation, a learned transliteration dictionary and decoy-aware features resolves 12M test records on an 8 GB laptop in about four hours end to end, with a held-out macro F0.5 of 0.984 at the true training density. The main lessons: validate at true density with complete candidate groups (otherwise the estimate is optimistic), treat the house number as the decoy discriminator, and let an exact expected-F0.5 rule — not a fixed threshold — decide what to emit.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`: `normalize.py`, `translit.py` (+ `translit_dict.json`), `preprocess.py`, `blocking.py`, `blocking_extra.py`, `blocking_report.py`, `features.py`, `pipeline_core.py`, `train.py` (helpers), `train_full.py` (training + validation), `predict.py` (inference, writes both files); `make_train_subset.py` (replica used in early versions). `models/` holds the two LightGBM boosters and `config.json` with the decision rule and validation results. `README.md` lists the exact commands.

### B. Additional Results

Test-set run (v5 models):

| | France | India | US | all |
| --- | --- | --- | --- | --- |
| Source-1 entities | 259,452 | 809,986 | 663,106 | 1,732,544 |
| candidate pairs (per entity) | 18.8M (72.6) | 51.4M (63.4) | 40.7M (61.5) | 111.0M (64.1) |
| accepted matches per entity | 3.46 | 3.35 | 3.38 | 3.38 |
| entities predicted empty | | | | 5.61% |

Reduction ratio on test: 1 − 110,968,547 / (1,732,544 × 9,969,589) = 99.99936%.

Label-free shift check: re-estimating the match prior among each record's top candidate with the EM procedure of Saerens et al. (which reproduces the training priors exactly) gives 0.861 for India on test versus 0.860 in training (no shift) but 0.798 for France versus 0.857, i.e. France has proportionally more non-matching top candidates than the training countries. French Source-1 entities also share addresses far more often (11.8% vs 4–5%). Both point at France as the main source of the gap between held-out validation and the public leaderboard; France has no training labels, so no France-specific rule was tuned.
