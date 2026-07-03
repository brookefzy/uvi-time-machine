#!/usr/bin/env bash
#SBATCH --job-name=dinov3_smoke
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
PYTHON="${PYTHON:-python}"
SMOKE_CITY="${SMOKE_CITY:-Hong Kong}"
MISMATCH_ARGS=()
if [[ "${DINO_IGNORE_MISMATCHED_SIZES:-0}" == "1" || "${DINO_IGNORE_MISMATCHED_SIZES:-0}" == "true" ]]; then
  MISMATCH_ARGS+=(--ignore-mismatched-sizes)
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage smoke \
  --city "${SMOKE_CITY}" \
  --city-meta "${CITY_META}" \
  --repo-dir "${REPO_DIR}" \
  --model-name "${MODEL_NAME}" \
  --backend "${DINO_BACKEND:-transformers}" \
  --device "${DINO_DEVICE:-cuda}" \
  --smoke-batch-size "${SMOKE_BATCH_SIZE:-2}" \
  --smoke-limit "${SMOKE_LIMIT:-2}" \
  "${MISMATCH_ARGS[@]}" \
  --execute
