#!/usr/bin/env bash
# Submit pairwise DINOv3 H3 tasks in bounded arrays to respect account job caps.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-dinov3_city={city}_res_exclude=None.parquet}"
RESOLUTION="${RESOLUTION:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if (( BATCH_SIZE < 1 || ARRAY_CONCURRENCY < 1 || POLL_SECONDS < 1 )); then
  printf 'BATCH_SIZE, ARRAY_CONCURRENCY, and POLL_SECONDS must be positive.\n' >&2
  exit 2
fi
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/dinov3_similarity
MANIFEST="${PAIR_MANIFEST:-logs/dinov3_similarity/pairs_res=${RESOLUTION}_$(date +%Y%m%d_%H%M%S).txt}"
"${PYTHON}" slurm/generate_dinov3_pair_manifest.py --city-meta "${CITY_META}" --source-root "${SOURCE_ROOT}" --input-template "${INPUT_TEMPLATE}" --resolution "${RESOLUTION}" --output "${MANIFEST}"
PAIR_COUNT="$(wc -l < "${MANIFEST}")"
for ((start=1; start<=PAIR_COUNT; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1)); (( end > PAIR_COUNT )) && end="${PAIR_COUNT}"
  job_id="$(sbatch --parsable --export=ALL,PAIR_MANIFEST="${MANIFEST}" --array="${start}-${end}%${ARRAY_CONCURRENCY}" slurm/dinov3_03_pairwise_array.cmd)"
  printf 'Submitted pair tasks %s-%s/%s: job %s\n' "${start}" "${end}" "${PAIR_COUNT}" "${job_id}"
  while squeue -h -j "${job_id}" -o "%T" | grep -q .; do
    sleep "${POLL_SECONDS}"
  done
done
