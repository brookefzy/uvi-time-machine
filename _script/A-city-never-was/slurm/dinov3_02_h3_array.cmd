#!/usr/bin/env bash
#SBATCH --job-name=dinov3_h3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-8%4

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
DEFAULT_CITY_META="${REPO_DIR%/*}/city_meta.csv"
CITY_META="${CITY_META:-${DEFAULT_CITY_META}}"
if [[ "${CITY_META}" == "${REPO_DIR}/city_meta.csv" ]]; then
  CITY_META="${DEFAULT_CITY_META}"
fi
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage aggregate \
  --city-meta "${CITY_META}" \
  --city-index "${SLURM_ARRAY_TASK_ID}" \
  --repo-dir "${REPO_DIR}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  --execute
