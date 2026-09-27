# Business Entity Resolution — reproducible pipeline

End-to-end pipeline that regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the raw challenge data. It runs on a laptop
(8 GB RAM, 8 cores) in roughly 10 hours end to end, most of it blocking; no
external data, APIs or pretrained models are used.

```
data (tsv) ─▶ translit.py ─▶ preprocess.py ─▶ blocking.py + blocking_extra.py ─▶ train_full.py ─▶ predict.py ─▶ output/*.tsv
```

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 was used; any 3.10+ should work.

## Reproduce end-to-end

Run from `code/business_entity_resolution/src`. `DATA` is the challenge `dataset/`
folder (with `train/` and `test/`); `WORK` is a scratch folder (~25 GB of parquet
intermediates).

```bash
DATA=/path/to/student_resource/dataset
WORK=/path/to/work
MODELS=../models
OUT=../../../output

# 0. learn the Indic-script -> English token dictionary from the training pairs
python translit.py --data-dir $DATA --out translit_dict.json

# 1. normalise every record of every split (parallel, ~5 min)
python preprocess.py --data-dir $DATA --work-dir $WORK

# 2. candidate generation for the full training set and the test set (+ extra pass for hard records)
python blocking.py --work-dir $WORK --split train && python blocking_extra.py --work-dir $WORK --split train
python blocking.py --work-dir $WORK --split test  && python blocking_extra.py --work-dir $WORK --split test

# 3. train stage-1 / stage-2 models on entity folds and validate on held-out entities (full density)
python train_full.py --work-dir $WORK --split train --models-dir $MODELS

# 4. score the test candidates and write both submission files
python predict.py --work-dir $WORK --split test --models-dir $MODELS --out-dir $OUT

# 5. validate the files
cd /path/to/student_resource && python3 utils/validate_submission.py \
    --matching output/matching_results.tsv --test-dir dataset/test
```

`blocking_report.py --split <split>` prints the blocking confusion matrix for a split
with ground truth. `make_train_subset.py` + `train.py` reproduce the earlier
replica-based versions and are not needed for the final result.

## What each module does

| file | role |
| --- | --- |
| `normalize.py` | rule-based cleaning of names/addresses: accent stripping, legal-suffix canonicalisation, abbreviation expansion (US/India/France), state detection, number extraction, Indic-script transliteration |
| `translit.py` | learns `translit_dict.json` (Indic token → English token) by positional alignment of matched training names |
| `preprocess.py` | applies the normalisation to all six files, caches parquet |
| `blocking.py` | four word-level TF-IDF indexes per country (joint name+address uni/bigrams, two phonetic-skeleton variants, name-only), top-k retrieval with sparse matrix products, rule-based trimming; one country and one chunk in memory at a time |
| `blocking_extra.py` | character-trigram name index for records with an empty address or single-token name |
| `blocking_report.py` | blocking confusion matrix / reduction ratio |
| `features.py` | 80 pairwise features (token overlaps, rapidfuzz similarities, phonetic skeletons, house-number agreement/offset/parity, state agreement, name and address frequency in Source 1, blocker cosines) |
| `pipeline_core.py` | candidate loading, chunked feature computation and stage-1 scoring, competition/context features, one-to-one assignment, expected-F0.5 decision, macro-F0.5 evaluation |
| `train_full.py` | trains stage-1 (plus two cross-fitted copies) and stage-2 LightGBM models on disjoint entity folds of the full training set and validates on held-out entities |
| `predict.py` | per-country inference on the test candidates; writes `candidate_pairs.tsv` (exactly the pairs scored) and `matching_results.tsv` |

Models are LightGBM gradient-boosted trees (MIT licence, a few tens of MB), far below
the 8B-parameter limit.
