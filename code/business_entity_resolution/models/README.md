# Frozen matcher assets

The three models used for the measured holdout are `xgboost.json`, `group_extra.json`, and `augmented_0p15.json`. `config.json` records the frozen weights, threshold, and SHA-256 digests. These copies let the inference package carry its models under `code/business_entity_resolution/`.

The same files also retain their original paths under `artifacts/matching_round5/` so the evaluation hashes and provenance remain valid in this repository. Intermediate training arrays, raw datasets, and the 409,141-target holdout index are not committed. The test inference entry point is still being implemented.
