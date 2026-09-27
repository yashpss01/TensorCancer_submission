# Locked stage-one protocol

Scope follows the user's updated request: the **first 10,000 training Source 1 rows**, not all 2,206,821 references. There is no matching classifier in this stage.

- Deterministic SHA256 ordering, seed `blocking-first10k-v1-20260926`: 7,000 development and 3,000 holdout references. Sampled truth groups were audited against the full truth file for shared target ownership; none overlap.
- Pool: every labeled target of those 10,000 references, plus all other S2/S3 records whose SHA256(`distractor-v1` + ID) first 32 bits fall below `floor(0.02 * 2**32)`. Exactly 239,788 targets: 34,752 required positives and 205,036 other distractors. Both development and holdout search the same pool. No country restriction.
- Labels and IDs choose the benchmark and score it. Retrieval sees business names and addresses only. Holdout labels are necessarily used to include its positives in the reduced pool, but holdout outcomes and text errors are not inspected during development tuning.
- Initial four routes were extended during development to six: lexical names, lexical addresses, joint lexical name/address, pairs of rare name trigrams, coarse phonetic-name trigrams, and phonetic-name/address intersection. Each returns at most 40 targets. The union has at most 240 records and is reduced by unsupervised similarity and a configurable cap/cutoff. Final candidate files contain exactly this reduced set.
- Development sweep: caps 5–160; score cutoffs 0–0.70. Select the smallest mean candidate list satisfying development recall at least 99.75%, if feasible. This margin was specified before reading the completed development results. Inspect development misses and improve retrieval when needed before freezing.
- Freeze code/configuration hashes, then evaluate holdout once. Do not revise blocking based on holdout misses. If the target is missed, report it honestly.
- Requested aspiration: blocking FN rate below 0.5% / pair completeness above 99.5%. This is separate from macro per-S1 F0.5. The oracle macro F0.5 retains only true candidate matches and assigns singleton entities an empty list (score 1). It is a ceiling, not classifier performance.
- Pair comparison universe for each reported group: number of evaluated S1 records × **239,788**, including cross-country pairs. Positives are all labeled links belonging to those S1 rows; all other pairs are treated as negative according to provided ground truth.

This benchmark is intentionally optimistic: 98% of nonselected targets are absent, all sample positives are preserved, and first-row sampling need not represent the population. It cannot establish full-corpus retrieval quality, unseen-France performance, a leaderboard score, or achievement of the eventual 99.2% macro-F0.5 ambition. Candidate sizes and computational cost must be remeasured before full deployment.

The local README says candidates are not scored. The newer organizer update supplied by the user supersedes that statement: final candidate set size affects rankings. This experiment therefore explicitly optimizes candidate count while measuring recall.

## Development-stage budget amendment (before holdout)

The initial 99.75% development guardrail required 75.50 candidates/S1 at a cap of 120. Before opening holdout, the final budget decision uses a 99.70% development guardrail to halve this cost: cap 60 at score 0.45, plus the top 12 address results and selective missing-address name rescue. Development recall is 99.70346%, mean 38.09 candidates. The requested 99.5% holdout target is unchanged. This is a documented development choice, not a holdout-driven adjustment. The exact evaluated grid and alternative configurations remain saved.
