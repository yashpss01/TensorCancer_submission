# Frozen matcher assets

The three models used for the measured holdout are `xgboost.json`, `group_extra.json`, and `augmented_0p15.json`. `manifest.json` records their SHA-256 digests, exact feature order, selection weights, and threshold. `token_idf.joblib`, `char_idf.joblib`, and `baseline_config.json` preserve the training-time feature and retrieval settings. These files let inference run from this package without the original experiment artifacts.

The three XGBoost files also retain their original paths under `artifacts/matching_round5/` so the evaluation hashes and provenance remain valid in this repository. `config.json` is a copy of the frozen holdout configuration. Intermediate training arrays, raw datasets, and the 409,141-target holdout index are not committed. Full test inference has not been run.
