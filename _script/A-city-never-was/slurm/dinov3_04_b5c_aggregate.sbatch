#!/usr/bin/env bash
#SBATCH --job-name=dinov3_b5c
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage b5c \
  --city-meta "${CITY_META}" \
  --repo-dir "${REPO_DIR}" \
  --resolution "${RESOLUTION:-8}" \
  --b5c-memory-limit "${B5C_MEMORY_LIMIT:-64GB}" \
  --b5c-threads "${B5C_THREADS:-8}" \
  --execute
