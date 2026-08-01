#!/usr/bin/env bash
# Submit only unfinished DINOv3 pairwise tasks in bounded Slurm arrays.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"
RESOLUTION="${RESOLUTION:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair_res=${RESOLUTION}}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"
if [[ -z "${INPUT_TEMPLATE}" ]]; then
  INPUT_TEMPLATE='dinov3_city={city}_res_exclude=None.parquet'
fi
BATCH_SIZE="${BATCH_SIZE:-20}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if (( BATCH_SIZE < 1 || ARRAY_CONCURRENCY < 1 || POLL_SECONDS < 1 )); then
  printf 'BATCH_SIZE, ARRAY_CONCURRENCY, and POLL_SECONDS must be positive.\n' >&2
  exit 2
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm logs/dinov3_similarity
PAIR_MANIFEST="${PAIR_MANIFEST:-logs/dinov3_similarity/pairs_res=${RESOLUTION}_$(date +%Y%m%d_%H%M%S).txt}"
PENDING_MANIFEST="${PAIR_MANIFEST%.txt}_pending.txt"
"${PYTHON}" slurm/generate_dinov3_pair_manifest.py --city-meta "${CITY_META}" --source-root "${SOURCE_ROOT}" --input-template "${INPUT_TEMPLATE}" --resolution "${RESOLUTION}" --output "${PAIR_MANIFEST}"

: > "${PENDING_MANIFEST}"
DONE=0
PENDING=0
while IFS='|' read -r CITY1 CITY2; do
  [[ -z "${CITY1}" || -z "${CITY2}" ]] && continue
  SHARD="${OUTPUT_ROOT}/optimized/temp/city1=${CITY1}/city2=${CITY2}/part_res=${RESOLUTION}.parquet"
  if [[ -s "${SHARD}" ]]; then
    DONE=$((DONE + 1))
    continue
  fi
  printf '%s|%s\n' "${CITY1}" "${CITY2}" >> "${PENDING_MANIFEST}"
  PENDING=$((PENDING + 1))
done < "${PAIR_MANIFEST}"

printf 'Previously complete: %s; pending: %s\n' "${DONE}" "${PENDING}"
if (( PENDING == 0 )); then
  exit 0
fi

for ((start=1; start<=PENDING; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1)); (( end > PENDING )) && end="${PENDING}"
  task_count=$((end - start + 1))
  job_id="$(sbatch --parsable --export=ALL,PAIR_MANIFEST="${PENDING_MANIFEST}",PAIR_OFFSET=$((start - 1)) --array="1-${task_count}%${ARRAY_CONCURRENCY}" slurm/dinov3_03_pairwise_array.cmd)"
  printf 'Submitted pending pair tasks %s-%s/%s: job %s\n' "${start}" "${end}" "${PENDING}" "${job_id}"
  while squeue -h -j "${job_id}" -o "%T" | grep -q .; do
    sleep "${POLL_SECONDS}"
  done
done
