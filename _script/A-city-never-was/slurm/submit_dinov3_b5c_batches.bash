#!/usr/bin/env bash
# One city per task; bounded arrays. Reuse RUN_TAG only while inputs stay unchanged.
set -euo pipefail
export ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
export UVI_SAMPLE_REPO_DIR="${UVI_SAMPLE_REPO_DIR:-${ROOTFOLDER}/uvi-time-machine/_script/A-city-never-was}"
cd "$UVI_SAMPLE_REPO_DIR"
export VENV_PYTHON="${VENV_PYTHON:-$(cd ../.. && pwd)/.venv/bin/python}"
export CITY_META="${CITY_META:-${UVI_SAMPLE_REPO_DIR%/*}/city_meta.csv}"
export RESOLUTION="${RESOLUTION:-8}"
export PAIRWISE_ROOT="${PAIRWISE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair}"
export H3_MEMBERSHIP_ROOT="${H3_MEMBERSHIP_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"
export SIMILARITY_EXPORT_FOLDER="${SIMILARITY_EXPORT_FOLDER:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_res=${RESOLUTION}}"
RUN_TAG="${RUN_TAG:?Set RUN_TAG; reuse it to resume unchanged inputs}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_TAG' >&2; exit 2; }
BATCH_SIZE="${BATCH_SIZE:-20}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
for value in "$BATCH_SIZE" "$ARRAY_CONCURRENCY"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch size and concurrency must be positive integers' >&2; exit 2; }
done
export B5C_RUN_DIR="$SIMILARITY_EXPORT_FOLDER/_batches/$RUN_TAG"
export B5C_CITY_MANIFEST="$B5C_RUN_DIR/cities.txt"
mkdir -p logs/slurm "$B5C_RUN_DIR"
# Keep the manifest stable across retries so city checkpoint identities cannot shift.
"$VENV_PYTHON" -c '
import pandas as pd, pathlib, sys
cities = sorted(pd.read_csv(sys.argv[1])["City"].dropna().astype(str).unique())
if not cities or any("\n" in c or "\r" in c for c in cities):
    raise SystemExit("Empty or invalid city list")
p = pathlib.Path(sys.argv[2]); content = "".join(c + "\n" for c in cities)
if p.exists() and p.read_text() != content:
    raise SystemExit("City list changed: use a new RUN_TAG")
p.write_text(content)
' "$CITY_META" "$B5C_CITY_MANIFEST"
COUNT=$(wc -l < "$B5C_CITY_MANIFEST")
for ((start=1; start<=COUNT; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1)); (( end > COUNT )) && end=$COUNT
  export CITY_OFFSET=$((start - 1))
  echo "Submitting B5c cities $start-$end/$COUNT at res=$RESOLUTION"
  # --wait returns a failure if any array task fails; do not silently continue.
  sbatch --wait --export=ALL --array="1-$((end-start+1))%${ARRAY_CONCURRENCY}" \
    slurm/dinov3_04_b5c_array.cmd
done
echo 'All B5c batches completed successfully.'
