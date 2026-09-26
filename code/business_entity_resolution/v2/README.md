# Experimental batched retrieval

This is a separate, label-blind candidate engine inspired by the batched sparse
retrieval in `Akanksha`. It does **not** replace the frozen `Archisman` submission
pipeline or constitute a completed quality result. The final matcher still needs
training and fresh evaluation on the new candidate distribution.

`prepare.py` normalizes Source 1, 2, and 3 TSVs into bounded Parquet shards.
`retrieve.py` builds four TF-IDF views of the complete Source-1 universe for
each country and queries Source 2/3 in batches. The normalized name logic is
adapted from the `Akanksha` branch, but its dictionary learned from training
match labels is excluded. For evaluation, a selected Source-1 batch is filtered
**after** retrieval ranking against all Source-1 competitors. The country
partition is supported by the exposed training-label audit; no France accuracy
is implied by this experiment.

`experiments/modal_v2_candidates.py` orchestrates a bounded Modal CPU run. It
expects a tar containing only the three raw training source TSVs and a separate
one-column selected-ID TSV; **ground truth stays local**. Retrieval workers
write unique candidate shards and manifests to a Volume. After all shards are
downloaded, `evaluate_candidates.py` opens the sealed local truth and reports
candidate confusion counts, blocking recall, candidates per S1, and the oracle
macro F0.5 ceiling. That ceiling assumes perfect decisions among retrieved true
links; it is not a measured matching result. The Modal job also exports the raw
records for retrieved targets, so `score_frozen.py` can apply the original
Archisman model locally at its original 16-candidate cap. That script is
label-blind; it writes prediction and candidate TSVs for the selected 10k only.
They are not complete challenge submission TSVs.

The local smoke fixture verified that filtering selected IDs after global
ranking yields exactly the corresponding pairs from unfiltered retrieval. A
30,097-query, 29,965-index-record US engineering pilot ran in 4.4 seconds after
indexing on the local machine. The new frozen-model scorer reproduced the
original inference function's predictions exactly on a two-entity fixture.
These are not estimates for the full dataset.
Full-train retrieval, quality scoring, and any full-test TSV generation remain
unrun as of this commit.
