#!/usr/bin/env bash
#SBATCH --job-name=dinov3_pairwise
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage pairwise \
  --city-meta "${CITY_META}" \
  --repo-dir "${REPO_DIR}" \
  --resolution "${RESOLUTION:-8}" \
  --res-exclude "${RES_EXCLUDE:-11}" \
  --threshold "${DINO_THRESHOLD:--1.0}" \
  --row-block-size "${ROW_BLOCK_SIZE:-1000}" \
  --b5b-memory-limit "${B5B_MEMORY_LIMIT:-96GB}" \
  --execute
