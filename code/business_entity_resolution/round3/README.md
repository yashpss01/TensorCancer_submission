# Frozen blocking evaluation stopped at 30,000 references

The pool and run were originally prepared for Source 1 rows 20,001–80,000. The user then requested a stop at 30,000 references. The completed evaluation therefore scores exactly original Source 1 rows 20,001–50,000. The original and improved round-2 retrieval code and settings are loaded directly and verified against `artifacts/blocking_round2/frozen_config.json`. The improved configuration remains threshold 0.70, at most four added candidates per S1. There is no configuration selection on this batch.

Run from the workspace root with CPython 3.13 and SQLite FTS5:

```sh
python3 code/business_entity_resolution/round3/evaluate_60k.py prepare
python3 code/business_entity_resolution/round3/evaluate_60k.py extra_index
python3 code/business_entity_resolution/round3/evaluate_60k.py seal
python3 code/business_entity_resolution/round3/evaluate_60k.py baseline
python3 code/business_entity_resolution/round3/evaluate_60k.py improved
python3 code/business_entity_resolution/round3/evaluate_60k.py record_stop
python3 code/business_entity_resolution/round3/evaluate_60k.py score
python3 code/business_entity_resolution/round3/evaluate_60k.py report
```

For the recorded run, `baseline` and `improved` were interrupted after they had each passed 30,000 results. The runner now has a 30,000-row limit for clean reproduction. `record_stop` captures the exact 30,000th-result elapsed time from the logs and verifies both raw files have at least 30,000 rows. Scoring consumes only the first 30,000 rows of each raw file; extra rows written during the recorded shutdown are preserved but excluded. Completed stages refuse to overwrite their outputs.

Both pipelines searched the same newly built 409,141-target pool: all labeled S2/S3 targets for the originally planned 60k S1 references plus a deterministic 2% SHA256 sample of the remaining targets. The evaluated 30k references therefore have all their positives present; positives associated with the unevaluated second half remain in the pool. The pool excludes most full-corpus negatives. Its absolute recall and candidate sizes may differ from a full S2/S3 run or from a pool built specifically for 30k. Consecutive source rows may also differ from other population slices.

Labels are used for benchmark construction, ownership audit, and final scoring only. Record IDs identify records and seed the distractor sample. No ground truth, IDs, or country-based filtering enters retrieval. The first 20k S1 records are prior development/evaluation batches; selected target ownership is checked against the full truth file, and exact normalized name/address/country signatures are compared. The config, source, query, truth, indices, and manifest hashes are sealed before either retrieval run.

Outputs are in `artifacts/blocking_round3/fresh/`; results and explanation are in `reports/blocking_round3_30k_evaluation.md`. Both 30k candidate TSVs are reopened and checked row by row against the selected in-memory sets and target IDs. The improved set must contain every baseline candidate. Neither retrieval pass changes the pool or frozen rules.
