# Round 4: fresh validation of a blocking selection change

This round uses the already inspected Source 1 rows 20,001–50,000 as development data. It changes only the final selection of candidates already retrieved by the frozen round-2 blocking pipeline. The original rule keeps up to four rescues scoring at least 0.70. The new rule keeps up to 16 rescues scoring at least 0.50, while retaining every original candidate. The development sweep and reason for this choice are saved under `artifacts/blocking_round4/`.

Original Source 1 rows 50,001–60,000 are the fresh check. Both rules use the same sealed 409,141-target round-3 pool, containing all labeled targets for this slice plus a deterministic 2% distractor sample. The pool was originally built for S1 rows 20,001–80,000 and therefore also contains positives for other S1 rows. It is an enriched reduced pool, so full-corpus performance is not established.

Run from the workspace root with CPython 3.13 and SQLite FTS5:

```sh
python3 code/business_entity_resolution/round4/evaluate.py development
python3 code/business_entity_resolution/round4/evaluate.py prepare
python3 code/business_entity_resolution/round4/evaluate.py freeze
python3 code/business_entity_resolution/round4/evaluate.py baseline
python3 code/business_entity_resolution/round4/evaluate.py improved
python3 code/business_entity_resolution/round4/evaluate.py score
python3 code/business_entity_resolution/round4/report.py
```

The new rule is frozen before fresh retrieval and scoring. Once finished, the `fresh/frozen/` and `fresh/updated/` directories contain exact candidate TSVs and metrics. Their rows, target membership, subset relation, and source/input hashes are checked. Existing round-2 and round-3 code and artifacts remain unchanged. Labels are only benchmark construction and scoring inputs; IDs and country do not filter retrieval.
