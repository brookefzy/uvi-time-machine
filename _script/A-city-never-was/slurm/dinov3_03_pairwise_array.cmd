#!/usr/bin/env bash
#SBATCH --job-name=dinov3_pair
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --export=ALL

set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"
if [[ -z "${INPUT_TEMPLATE}" ]]; then
  INPUT_TEMPLATE='dinov3_city={city}_res_exclude=None.parquet'
fi
PAIR_MANIFEST="${PAIR_MANIFEST:?Set PAIR_MANIFEST to a CITY1|CITY2 manifest.}"
PAIR_OFFSET="${PAIR_OFFSET:-0}"
if [[ ! "${PAIR_OFFSET}" =~ ^[0-9]+$ ]]; then
  printf 'PAIR_OFFSET must be a non-negative integer: %s\n' "${PAIR_OFFSET}" >&2
  exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PAIR_LINE_NUMBER=$((PAIR_OFFSET + SLURM_ARRAY_TASK_ID))
PAIR="$(sed -n "${PAIR_LINE_NUMBER}p" "${PAIR_MANIFEST}")"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2
  exit 127
fi
if [[ -z "${PAIR}" || "${PAIR}" != *"|"* ]]; then
  printf 'Invalid or empty pair manifest row %s in %s\n' "${PAIR_LINE_NUMBER}" "${PAIR_MANIFEST}" >&2
  exit 2
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm
"${PYTHON}" "${REPO_DIR}/B5b_compute_similarity_pairwise-optimized.py" \
  --city-pair "${PAIR}" \
  --resolution "${RESOLUTION:-8}" \
  --source-root "${SOURCE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}" \
  --output-root "${OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair}" \
  --input-template "${INPUT_TEMPLATE}" \
  --feature-prefix e_ \
  --threshold "${DINO_THRESHOLD:--1.0}" \
  --row-block-size "${ROW_BLOCK_SIZE:-1000}" \
  --memory-limit "${B5B_MEMORY_LIMIT:-96GB}" \
  --res-exclude None \
  --log-dir "${LOG_DIR:-logs/dinov3_similarity}"
