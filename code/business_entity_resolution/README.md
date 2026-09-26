# Business entity resolution: research snapshot

This first Archisman-branch commit preserves the completed blocking and matching experiments, frozen models, and measured holdout evidence. **It does not yet provide a test inference entry point or generated Portal TSVs.** The current matcher is in `round5/`; the earlier sections below document the initial blocker and remain useful for reproducing that historical stage. The latest measured outcome is in `reports/matching_round5_holdout_review.md` at the repository root. A self-contained test inference runbook will be added after implementation and validation.

## Historical stage 1: candidate generation and local evaluation

Run from the repository root with Python 3.13 and SQLite 3.45.3 (FTS5 enabled). This implementation uses only the Python standard library. No network, external entity data, pretrained model, or matching classifier is used. `phonetic.py` derives a coarse Indic transliteration from standard Unicode character names; it does not contain a business vocabulary.

```sh
python3 code/business_entity_resolution/src/blocking.py prepare
python3 code/business_entity_resolution/src/blocking.py build
python3 code/business_entity_resolution/src/evaluate.py run dev
python3 code/business_entity_resolution/src/evaluate.py summarize dev
python3 -m unittest discover -s tests -v
```

The first two commands refuse to overwrite the split or index. Existing artifacts can be inspected directly. To reproduce in a fresh workspace, copy code and the original dataset directory, then run these commands. Do not delete or modify the original datasets.

`prepare` takes the first 10,000 training S1 file rows, orders them by SHA256 of a fixed seed plus ID, and assigns 7,000 to development and 3,000 to holdout. It audits all ground-truth rows for shared target ownership involving the sample. IDs are used only for bookkeeping and reproducible selection, never as retrieval features. Labels construct the evaluation pool and score results; the retriever receives text only.

`build` scans all S2/S3 records once. It retains every labeled target for the 10,000 references, plus a deterministic 2% sample of other targets. The index retains original strings, alongside derived normalized fields. Country is never a hard filter, so unseen labels remain eligible. This is an intentionally reduced and enriched target pool: its results cannot establish full-corpus performance, unbiased population accuracy, or generalization to France.

`run dev` evaluates complementary indexed name, address, joint name/address, name-trigram, phonetic-name, and phonetic-name/address routes. Each route returns at most 40 records, and their deduplicated union is ranked using unsupervised weighted token cosine/containment and character/phonetic Dice overlap. Candidate budget sweeps expose the recall/cost tradeoff. This scoring is a blocking filter, not a trained matching model. The exported `candidate_pairs.tsv` is the final stage-one set intended for a future matching model; raw retrieval lists are separate diagnostics.

Run `python3 code/business_entity_resolution/src/tune.py` to sweep caps/cutoffs and propose the smallest development configuration meeting the 99.75% guardrail. Then run `python3 code/business_entity_resolution/src/rescue_sweep.py` to evaluate selective rescues, enrich cached rankings, and choose the final 99.70% development guardrail configuration. This guardrail was amended before holdout to reduce candidate size; see the protocol. Inspect development misses before freezing. After selecting the development configuration, run `python3 code/business_entity_resolution/src/freeze.py` to save its hashes, then run:

```sh
python3 code/business_entity_resolution/src/evaluate.py run holdout
```

The holdout command records a one-time opening marker and refuses subsequent runs against the same benchmark. No tuning may use holdout misses. A future iteration needs another independent benchmark. Only development results support budget sweeps.

Artifacts live under `artifacts/blocking/`. `queries.json` and separate truth files preserve the evaluation sample. `index.sqlite` stores the searchable target pool; `index_meta.json` records its size and construction resources. Split directories hold rankings, runtime, final candidate TSV, metrics, and missed-pair text. No `matching_results.tsv` is produced: final classification is out of scope.

The disk-backed index bounds memory, but rare-token postings may grow substantially in the full corpus. Full-corpus index and query runtime, candidate competition, recall, and candidate budgets remain unvalidated. The earlier full-corpus attempt was stopped following the user's smaller-experiment request; its incomplete artifacts are kept separately in `artifacts/full_corpus_aborted` and never used in results.

Retrieval runs with three local worker processes. Runtime reports distinguish parent peak RSS from maximum child peak RSS; they do not measure aggregate concurrent system memory. The saved `iteration_log.json` distinguishes the invalid compact-index diagnostic, valid baseline, small pilots, and final full development sweep. `reindex.py` can rebuild derived fields from an existing pool without another corpus scan; `build_rescue.py` is an archived experimental separate-index builder and is not needed for the final pipeline.

After the holdout run, verify exact TSVs and generate the report:

```sh
python3 code/business_entity_resolution/src/verify.py
python3 code/business_entity_resolution/src/report.py
```

Verification also writes `artifacts/blocking/candidate_pairs.tsv` containing all first 10,000 references in original file order. Separate split files and metrics distinguish tuned development from untouched holdout. This combined file is not a competition test submission.
