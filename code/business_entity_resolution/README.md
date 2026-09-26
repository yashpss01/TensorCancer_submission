# Business Entity Resolution — reproducible pipeline

End-to-end pipeline that regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the raw challenge data.  Everything runs on a
laptop (8 GB RAM, 8 cores); no external data, APIs or pretrained models are used.

```
data (tsv) ─▶ preprocess.py ─▶ blocking.py ─▶ train.py (once) ─▶ predict.py ─▶ output/*.tsv
                   ▲
             translit.py  (token dictionary learned from the training pairs)
```

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 was used; any 3.10+ should work.

## Reproduce end-to-end

All commands are run from `code/business_entity_resolution/src`.
`DATA` is the challenge `dataset/` folder (with `train/` and `test/`), `WORK` is a
scratch folder (~15 GB of parquet intermediates).

```bash
DATA=/path/to/student_resource/dataset
WORK=/path/to/work
MODELS=../models
OUT=../../../output

# 0. learn the Indic-script -> English token dictionary from the training pairs
python translit.py --data-dir $DATA --out translit_dict.json

# 1. normalise every record of every split (parallel, ~5 min)
python preprocess.py --data-dir $DATA --work-dir $WORK

# 2. build the training replica (50% of Source-1 entities + their records + 50% of distractors)
python make_train_subset.py --work-dir $WORK --frac 0.5

# 3. candidate generation (blocking) for the replica and for the test set
python blocking.py --work-dir $WORK --split trainsub
python blocking.py --work-dir $WORK --split test
python blocking_report.py --work-dir $WORK --split trainsub     # blocking confusion matrix

# 4. train the two-stage matcher + choose the decision rule on held-out entities
python train.py --work-dir $WORK --split trainsub --models-dir $MODELS

# 5. score the test candidates and write both submission files
python predict.py --work-dir $WORK --split test --models-dir $MODELS --out-dir $OUT

# 6. validate the files
cd /path/to/student_resource && python3 utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## What each module does

| file | role |
| --- | --- |
| `normalize.py` | rule-based cleaning of names/addresses: accent stripping, legal-suffix canonicalisation, abbreviation expansion (US/India/France), state detection, number extraction, Indic-script transliteration |
| `translit.py` | learns `translit_dict.json` (Indic token → English token) by positional alignment of matched training names |
| `preprocess.py` | applies the normalisation to all six files, caches parquet |
| `make_train_subset.py` | scaled-down replica of the training world used for fitting and validation |
| `blocking.py` | four word-level TF-IDF indexes per country (joint name+address uni/bigrams, phonetic-skeleton variants, name-only), top-k retrieval with sparse matrix products, rule-based trimming |
| `blocking_report.py` | blocking confusion matrix / reduction ratio |
| `features.py` | ~65 pairwise features (token overlaps, rapidfuzz string similarities, phonetic skeletons, number/state agreement, blocker cosines) |
| `pipeline_core.py` | chunked feature computation, stage-1 scoring, competition/context features, stage-2 scoring, one-to-one assignment, expected-F0.5 decision, macro-F0.5 evaluation |
| `train.py` | trains stage-1 and stage-2 LightGBM models on disjoint entity folds and validates on a third fold |
| `predict.py` | inference on the test candidates; writes `candidate_pairs.tsv` (exactly the pairs scored) and `matching_results.tsv` |

Models are plain LightGBM gradient-boosted trees (MIT licence, a few MB), far below
the 8B-parameter limit.
