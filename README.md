# TensorCancer — Archisman branch

This branch preserves Archisman's candidate blocking and XGBoost matching work in the same top-level handoff layout used by `Akanksha`: `code/`, `output/`, a filled `Documentation_template.md`, and a submission packaging script.

The frozen labeled holdout scored **98.4117% per-S1 macro F0.5**, below the 99.5% goal. It used 20,000 original training S1 rows and a reduced, enriched 409,141-target pool. This is **not** a Portal or France result. The evidence is in [the holdout review](reports/matching_round5_holdout_review.md) and [the saved result JSON](artifacts/matching_round5/holdout/results.json).

The required `output/matching_results.tsv` and `output/candidate_pairs.tsv` must be generated on the provided test data, validated, and then packaged for submission. They are not pre-filled with training-holdout predictions. **The test inference entry point is still being implemented**, so this commit is an auditable holdout snapshot rather than a runnable Portal submission. See [the output note](output/README.md) for the delivery boundary.

The `reports/` and `artifacts/` directories preserve the experiment audit trail. The final challenge zip should include the runnable `code/business_entity_resolution/` folder, both generated TSVs, and the completed methodology document. The raw challenge dataset and large scratch arrays are intentionally outside Git.
