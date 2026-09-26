# Business entity resolution: frozen inference

Run from the repository/submission root with CPython 3.13, SQLite with FTS5, and the pinned packages in `code/business_entity_resolution/requirements.txt`. The runnable pipeline is entirely in `code/business_entity_resolution/src/`; its trained assets are in `code/business_entity_resolution/models/`. It reads only the provided test S1/S2/S3 TSVs during inference. It does **not** read training labels, previous evaluation SQLite files, or prior feature arrays.

## Produce both required outputs

Start with an empty index directory and an empty output directory. The index command includes **every** test S2 and S3 record, without filtering to known countries. It may require substantial local disk space and CPU time for the full corpus.

```sh
python3 -m pip install -r code/business_entity_resolution/requirements.txt

python3 code/business_entity_resolution/src/infer.py index \
  --test-dir dataset/test \
  --index-dir inference_index/test

python3 code/business_entity_resolution/src/infer.py run \
  --source1 dataset/test/test_source1.tsv \
  --index-dir inference_index/test \
  --output-dir output \
  --workers 6 \
  --batch-size 256

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

`run` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, each with one row per input S1 in original order. Empty candidate and match lists are written as empty TSV cells. Final matches are a subset of the candidate list for each S1. It also writes `output/inference_meta.json` with counts, hashes, and runtime. The files are first written with `.partial` suffixes and renamed only after successful completion. Existing index/output files are never overwritten; use a new empty directory for a rerun after an interruption.

For **matcher-only iteration** on a dataset whose candidates have already been
generated, use the cached candidate TSV with the same Source-1 order and target
index. This avoids the expensive FTS retrieval pass but does not speed up the
first run. It verifies the candidate digest in the prior run's metadata, checks
every Source-1 ID and candidate uniqueness, and records the source candidate
and model digests in the new metadata. Use a new output directory:

```sh
python3 code/business_entity_resolution/src/infer.py score-cached \
  --source1 dataset/test/test_source1.tsv \
  --candidate-tsv output/candidate_pairs.tsv \
  --index-dir inference_index/test \
  --output-dir rescored_output
```

On a 10,000-S1 reduced-pool validation batch, cached scoring took 228 seconds
versus 1,452 seconds for the original retrieval-and-scoring run (6.37× faster),
and both resulting TSVs were byte-for-byte identical. That timing does not
predict full-corpus performance. Model selection must still use development
labels and a fresh holdout; the unlabeled test output is not a tuning set.

`--limit N` is only for smoke testing and requires a separate output directory. Such output is incomplete and must never be submitted. The full test inference and Portal upload have **not** been run as part of this handoff; the validator must pass on the actual full `output/` files before submission.

For parallel machines or separate long-running jobs, use contiguous, zero-based row ranges. The current provided test S1 file has **1,732,544 data rows** (excluding its header); verify that count before choosing ranges. Every shard needs the same read-only index and model bundle. For example, two halves can run independently:

```sh
python3 code/business_entity_resolution/src/infer.py run \
  --source1 dataset/test/test_source1.tsv --index-dir inference_index/test \
  --output-dir shard_0 --start-row 0 --stop-row 866272

python3 code/business_entity_resolution/src/infer.py run \
  --source1 dataset/test/test_source1.tsv --index-dir inference_index/test \
  --output-dir shard_1 --start-row 866272 --stop-row 1732544

python3 code/business_entity_resolution/src/infer.py merge \
  --source1 dataset/test/test_source1.tsv \
  --shards shard_0 shard_1 --output-dir output
```

`merge` sorts shards by row range, verifies their hashes and contiguous coverage against every S1 ID, checks that matches are subsets of candidates, and writes the two final files. Run the validator after merging. Sharding the S1 work changes throughput only if the target index and enough CPU/SSD capacity are available to the workers. The local 20,000-S1 reduced-pool evaluation took about 35 minutes; a simple linear extrapolation to 1.73 million S1 exceeds two days, and the full target pool may be slower. This is a planning estimate, not a measured full-test runtime.

## Frozen method

Stage 1 uses SQLite FTS5 name, address, character-gram, and phonetic routes; a second text-only rescue pass adds bounded candidates. The final candidate rule preserves baseline candidates and adds up to 16 rescue candidates scoring at least 0.5. `candidate_pairs.tsv` contains exactly these candidates, before the matcher scores them.

Stage 2 computes the frozen 39 base pair features, 20 additional typo/character features, and 26 group-context features. A first XGBoost model supplies pair probabilities for group context. The final probability is `0.4 × group_model + 0.6 × augmented_pair_model`; links at or above 0.74 are selected. Token and character IDF dictionaries are bundled from training so inference uses the same feature definitions. The bundled `models/manifest.json` checks asset hashes and feature order. No S1/S2/S3 ID value enters a matching feature. Country equality is a feature, never a hard country filter, so France rows are processed.

The selected model was frozen before a 20,000-S1 training holdout. On a **reduced, positive-enriched 409,141-target pool**, the holdout scored 98.4117% competition macro F0.5 overall (India 98.3497%, US 98.4536%); blocking recall was 99.7707%. This is below the 99.5% aspiration. The result is **not** a full-corpus or Portal score, and it does not establish France accuracy. The full target corpus may change candidate competition and runtime substantially.

The repository's `round2/` through `round5/` directories and `reports/` hold the research, training, and evaluation provenance. They are not dependencies of `src/infer.py`. To reproduce inference in a submission zip, include `src/`, `models/`, this README, `requirements.txt`, the provided test data at the documented paths, and the challenge's validator. Fill `Documentation_template.md` and include it with both generated output TSVs in the final package.

## Local packaging verification already completed

- Exact candidate-list and final-match parity against frozen saved predictions for 10 untouched training-holdout rows.
- Tiny fresh index from 10 sample S2/S3 targets, then inference for two sample S1 rows, including one labeled `France`; both output TSVs passed `utils/validate_submission.py --check-ids` on that tiny fixture.
- An isolated folder containing only `src/` and `models/` built and scored that fixture. Two independent S1 shards merged to byte-for-byte identical TSVs as the unsharded run, and the merged TSVs passed `--check-ids`.
- No full test inference, Portal upload, or cloud job was performed for this packaging check.
