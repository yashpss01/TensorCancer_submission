#!/usr/bin/env bash
# Build <team>_submission.zip with the layout required by the challenge.
set -euo pipefail
cd "$(dirname "$0")"
TEAM=${1:-TensorCancer}
STUDENT_RESOURCE=${STUDENT_RESOURCE:-/Users/akanksha/Downloads/student_resource}

test -f output/matching_results.tsv || { echo "output/matching_results.tsv missing"; exit 1; }
test -f output/candidate_pairs.tsv || { echo "output/candidate_pairs.tsv missing"; exit 1; }

echo "== validating submission files"
( cd "$STUDENT_RESOURCE" && python3 utils/validate_submission.py \
    --matching "$OLDPWD/output/matching_results.tsv" \
    --candidate "$OLDPWD/output/candidate_pairs.tsv" \
    --test-dir dataset/test )

STAGE=$(mktemp -d)
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' code/business_entity_resolution/src "$STAGE/code/business_entity_resolution/"
rsync -a code/business_entity_resolution/models "$STAGE/code/business_entity_resolution/"
cp code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt "$STAGE/code/business_entity_resolution/"
cp Documentation_template.md "$STAGE/"
rm -f "${TEAM}_submission.zip"
( cd "$STAGE" && zip -qr "$OLDPWD/${TEAM}_submission.zip" . )
rm -rf "$STAGE"
echo "== wrote ${TEAM}_submission.zip"
unzip -l "${TEAM}_submission.zip" | tail -n +1 | head -40
