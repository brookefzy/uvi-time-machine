#!/usr/bin/env bash
#SBATCH --job-name=dinov3_h3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-127%24

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage aggregate \
  --city-meta "${CITY_META}" \
  --city-index "${SLURM_ARRAY_TASK_ID}" \
  --repo-dir "${REPO_DIR}" \
  --res-exclude "${RES_EXCLUDE:-11}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  --execute
