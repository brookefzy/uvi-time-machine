#!/usr/bin/env bash
# Run every eligible DINOv3 H3 city pair directly, without Slurm.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"
RESOLUTION="${RESOLUTION:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair_res=${RESOLUTION}}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-dinov3_city={city}_res_exclude=None.parquet}"
LOG_DIR="${LOG_DIR:-logs/dinov3_similarity}"
MANIFEST="${PAIR_MANIFEST:-${LOG_DIR}/pairs_res=${RESOLUTION}_direct_$(date +%Y%m%d_%H%M%S).txt}"

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

PAIR_COUNT=0
while IFS='|' read -r CITY1 CITY2; do
  [[ -z "${CITY1}" || -z "${CITY2}" ]] && continue
  PAIR_COUNT=$((PAIR_COUNT + 1))
  printf 'Running %s|%s (%s)\n' "${CITY1}" "${CITY2}" "${PAIR_COUNT}"

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

printf 'Completed direct processing of %s pairs.\n' "${PAIR_COUNT}"
