#!/usr/bin/env bash
# Package the exact challenge layout after inference and validation.
set -euo pipefail
cd "$(dirname "$0")"

TEAM=${1:-TensorCancer}
DATASET_DIR=${DATASET_DIR:-}
if [[ -z "$DATASET_DIR" ]]; then
  echo 'Set DATASET_DIR to the challenge dataset directory containing test/.' >&2
  exit 1
fi
if [[ ! -f output/matching_results.tsv || ! -f output/candidate_pairs.tsv ]]; then
  echo 'Generate both output TSVs before packaging.' >&2
  exit 1
fi

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "$DATASET_DIR/test"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/output" "$stage/code/business_entity_resolution"
cp output/matching_results.tsv output/candidate_pairs.tsv "$stage/output/"
cp -R code/business_entity_resolution/src code/business_entity_resolution/models "$stage/code/business_entity_resolution/"
cp code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt "$stage/code/business_entity_resolution/"
cp Documentation_template.md "$stage/"

archive="$(pwd)/${TEAM}_submission.zip"
rm -f "$archive"
(cd "$stage" && zip -qr "$archive" .)
echo "Created $archive"
