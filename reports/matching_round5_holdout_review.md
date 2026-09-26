# Frozen matcher: held-out phase review

## Result

The competition target of **99.5% per-Source-1 macro F0.5 was not met**. The frozen 0.4 group-model / 0.6 augmented-pair-model blend at threshold 0.74 scored **98.4117%** on 20,000 untouched training Source-1 rows (original rows 60,001–80,000). The model and cutoff were selected on a separate 10,000-row development slice, then frozen before holdout retrieval, feature extraction, or scoring. Development macro F0.5 was 98.3658%.

| Slice | S1 rows | Macro F0.5 | True links predicted | False links predicted | True links missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Holdout overall | 20,000 | 98.4117% | 67,723 | 637 | 1,607 |
| India | 8,061 | 98.3497% | 27,315 | 241 | 754 |
| US | 11,939 | 98.4536% | 40,408 | 396 | 853 |

The holdout has 1,046 true singletons; 1,025 received an empty prediction. Pair-level precision was 99.0682%, and pair-level recall was 97.6821%. These micro figures are diagnostics; the competition score is the macro F0.5 above.

The blocker supplied 991,322 candidate pairs, or 49.57 per S1. It retained 69,171 of 69,330 true links: **99.7707% overall blocking recall**, 99.6437% India, 99.8570% US. The blocker-only oracle macro F0.5 was 99.9198% overall (99.8528% India), assuming every retrieved true link could be selected with no false link. Of the 1,607 missed links, 159 were absent from the candidate list and 1,448 were present but rejected by the matcher. Improving the matcher is the larger opportunity on this pool.

The benchmark pool has **409,141 target records**: every labeled positive for the evaluated rows plus a deterministic 2% distractor sample. It is much easier than searching the full 10.32-million-target training corpus or the separate test corpus. These results do not establish full-corpus blocking recall, final competition accuracy, or France accuracy. France has no training labels.

## Evidence and reproducibility

- Frozen configuration: `artifacts/matching_round5/holdout/frozen_model.json`; it records model/code hashes, weights, threshold, and the development score. A setup-only amendment allowed feature files to share its directory after retrieval; it did not change models or predictions.
- One-time holdout metrics and hashes: `artifacts/matching_round5/holdout/results.json` (`status: PASS`). Retrieval outputs cover exactly 20,000 rows; feature metadata records 991,322 pairs and 69,171 candidate true links.
- The two retrieval passes took 678 and 802 seconds. Pair, extra-text, and group-context feature extraction took about 99, 317, and 196 seconds. Scoring took seconds. The local holdout run consumed roughly 35 minutes end to end, mostly CPU retrieval and text feature work.
- Supplemental model training used 691,760 positive pairs from disjoint S1 rows 80,001–280,000. A target-ID audit found no overlap with the final holdout labels. The augmented pair model alone scored 98.2362% on development; blending it with the group model scored 98.3658%.

## Root cause and next phase

Another small threshold or tree-depth sweep is unlikely to close the 1.09-point gap to 99.5%. On development, the fixed blend rejected 758 retrieved true pairs and accepted 307 false pairs. Among the missed true pairs, 460 had a missing address and 113 had a Unicode name mismatch; these categories overlap. Among the false pairs, 149 had a missing address. Examples also show near-identical names and addresses belonging to different businesses. The model needs a better way to distinguish **missing evidence** from conflicting evidence, resolve cross-script and severe typo variants, and avoid high-confidence false merges among near neighbors. Some records may remain intrinsically ambiguous from the available fields.

Prioritize a **hard-case reranker**, trained on provided labeled groups with difficult negatives, over another broad pairwise tree iteration. Its inputs should include a character-level or multilingual name representation (with deterministic transliteration where valid), structured address/house-number evidence, missing-field indicators, and target-to-target group support. Train it with group-level decisions and calibrate an abstention or empty-result option for ambiguous cases. First run a small, reproducible ablation on the existing development split, then freeze the architecture and evaluate on a newly sealed disjoint holdout (for example, S1 rows after 280,000). A full 10.32-million-target validation pool is needed before claiming operational recall; the enriched reduced pool can hide retrieval competition. Keep a separate France format/open-set stress check, without claiming France accuracy from unlabeled test data.

## Compute and delivery boundary

The current phase ran locally on CPU. The provided dataset is 2.4 GB; the reduced pool artifacts are about 742 MB and matcher artifacts about 1.5 GB. A full training target index could be tens of GB if it scaled linearly from the reduced SQLite/FTS pool; this is an estimate, not a measured size. CPU and SSD throughput dominate current blocking and feature extraction. A GPU becomes useful only for training/inference of a neural reranker. Moving to a larger machine alone will not parallelize the current serial batches: retrieval and feature work must be partitioned by S1 shard while using an immutable shared target index.

A bounded cloud pilot could use about 8–16 physical CPU cores, 32–64 GiB RAM, and 100+ GiB fast disk for index/shard experiments, with an L4-class GPU only if the neural reranker is tested. At [Modal's current standard resource rates](https://modal.com/pricing), 16 physical cores plus 64 GiB RAM is roughly $1.27 per running hour; an L4 GPU is about $0.80 per GPU-hour before associated CPU and memory. A $20–$30 pilot cap is reasonable, excluding any region multipliers, repeated jobs, and data transfer/setup time. No cloud job or spending was started in this phase.

The separate competition test set has not been processed and there has been no Portal score. Before delivery, the next phase needs full test inference including France, `output/matching_results.tsv` and `output/candidate_pairs.tsv` with every S1 exactly once, the repository validator passing, self-contained code and README with pinned requirements, a completed `Documentation_template.md`, and the required submission zip. GitHub push and Portal upload remain separate steps: this directory is not a Git repository, and no destination remote or Portal submission was provided for this task.
