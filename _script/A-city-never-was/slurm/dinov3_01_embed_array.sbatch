#!/usr/bin/env bash
#SBATCH --job-name=dinov3_embed
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=1-127%8

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage embed \
  --city-meta "${CITY_META}" \
  --city-index "${SLURM_ARRAY_TASK_ID}" \
  --repo-dir "${REPO_DIR}" \
  --model-name "${MODEL_NAME}" \
  --backend "${DINO_BACKEND:-transformers}" \
  --device "${DINO_DEVICE:-cuda}" \
  --batch-size "${DINO_BATCH_SIZE:-64}" \
  --execute
