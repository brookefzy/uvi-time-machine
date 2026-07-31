#!/usr/bin/env bash
# Resume direct DINOv3 H3 pairwise processing without overwriting completed shards.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"
RESOLUTION="${RESOLUTION:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair_res=${RESOLUTION}}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"
if [[ -z "${INPUT_TEMPLATE}" ]]; then
  INPUT_TEMPLATE='dinov3_city={city}_res_exclude=None.parquet'
fi
LOG_DIR="${LOG_DIR:-logs/dinov3_similarity}"
MANIFEST="${PAIR_MANIFEST:-${LOG_DIR}/pairs_res=${RESOLUTION}_resume_$(date +%Y%m%d_%H%M%S).txt}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2
  exit 127
fi
if [[ ! "${RESOLUTION}" =~ ^[0-9]+$ ]]; then
  printf 'RESOLUTION must be a non-negative integer: %s\n' "${RESOLUTION}" >&2
  exit 2
fi

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

"${PYTHON}" slurm/generate_dinov3_pair_manifest.py \
  --city-meta "${CITY_META}" \
  --source-root "${SOURCE_ROOT}" \
  --input-template "${INPUT_TEMPLATE}" \
  --resolution "${RESOLUTION}" \
  --output "${MANIFEST}"

DONE=0
PENDING=0
while IFS='|' read -r CITY1 CITY2; do
  [[ -z "${CITY1}" || -z "${CITY2}" ]] && continue
  SHARD="${OUTPUT_ROOT}/optimized/temp/city1=${CITY1}/city2=${CITY2}/part_res=${RESOLUTION}.parquet"

  if [[ -s "${SHARD}" ]]; then
    printf 'DONE — skipping %s|%s\n' "${CITY1}" "${CITY2}"
    DONE=$((DONE + 1))
    continue
  fi

  printf 'PENDING — running %s|%s\n' "${CITY1}" "${CITY2}"
  PENDING=$((PENDING + 1))
  "${PYTHON}" B5b_compute_similarity_pairwise-optimized.py \
    --city-pair "${CITY1}|${CITY2}" \
    --resolution "${RESOLUTION}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --input-template "${INPUT_TEMPLATE}" \
    --feature-prefix e_ \
    --threshold "${DINO_THRESHOLD:--1.0}" \
    --row-block-size "${ROW_BLOCK_SIZE:-1000}" \
    --memory-limit "${B5B_MEMORY_LIMIT:-96GB}" \
    --res-exclude None \
    --log-dir "${LOG_DIR}"
done < "${MANIFEST}"

printf 'Previously complete: %s; processed now: %s\n' "${DONE}" "${PENDING}"
