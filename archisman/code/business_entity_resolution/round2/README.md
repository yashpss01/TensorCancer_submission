# Focused blocking improvement, round 2

This round uses first-batch evidence for development and reserves original S1 rows 10,001–20,000 for one fresh paired comparison. It does not build a final matching classifier. Original source code and artifacts under `src/` and `artifacts/blocking/` are read only.

Run from `/Users/archismanchoudhury/Desktop/student_resource` using CPython 3.13 with SQLite FTS5. All runtime code uses the standard library. Three worker processes run retrieval; worker hash seeds are fixed to zero for repeatability. No model settings or external data services are used.

In a fresh workspace containing the original dataset and first-round artifacts:

```sh
python3 code/business_entity_resolution/round2/prepare.py
python3 code/business_entity_resolution/round2/extra_index.py dev
python3 code/business_entity_resolution/round2/extra_index.py fresh
python3 code/business_entity_resolution/round2/run.py run
python3 code/business_entity_resolution/round2/run.py sweep
python3 -m unittest discover -s tests -v
python3 code/business_entity_resolution/round2/run.py freeze
python3 code/business_entity_resolution/round2/run.py run --split fresh --kind baseline
python3 code/business_entity_resolution/round2/run.py run --split fresh --kind improved
python3 code/business_entity_resolution/round2/report.py
```

Optional development pilot: `run.py run --limit 1000`, then `run.py sweep --limit 1000`. Pilot outputs use separate names. They never select a final configuration in place of the full development sweep.

Existing outputs are not overwritten by pool construction or retrieval. Freeze records code, original-baseline configuration, queries, indices, and manifest hashes. Once the fresh evaluation marker exists, development runs and sweeps are prohibited. Both fresh retrieval files must finish before reporting reads matching outcomes. There is no tuning against the new results.

`prepare.py` performs a full label-ownership audit and one streaming S2/S3 scan. It includes all fresh positives plus a deterministic 2% distractor sample. Labels are used solely for benchmark construction/audit/scoring; the retriever accepts only text and cached label-blind baseline retrieval. IDs are used for record identity, output bookkeeping, and the reproducible sample.

`normalize.py` adds coarse cross-script bigrams and conventional address normalization. These are algorithms and abbreviation rules, not external business enrichment. Derived strings are stored in `extra.sqlite`; original names and addresses remain in the original records. All country labels remain eligible, including unfamiliar labels. France accuracy is not measured.

`improve.py` always retains the exact original candidate set. It retrieves up to 40 canonical address results, 40 phonetic/address results, and up to 20 results from each of three strongly linked text-derived anchors. Previously retrieved but filtered baseline records are also eligible for rescue. A development-selected score/cap limits the actual additions. This is blocking, not a trained final classifier or a claimed final match decision.

Outputs are under `artifacts/blocking_round2/`. Development grids, original-baseline metrics, and pilot logs remain separate from fresh results. `fresh/baseline/candidate_pairs.tsv` and `fresh/improved/candidate_pairs.tsv` contain exactly the sets being compared. The original baseline remains runnable without the new modules.

The report is `reports/blocking_round2_evaluation.md`. These are optimistic reduced-pool results: most corpus distractors are absent, consecutive rows may be biased, and neither full-corpus nor France performance is established. Fresh-set comparisons are paired; comparing their absolute scores directly with the old batch confounds population changes.
