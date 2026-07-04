#!/usr/bin/env bash
#SBATCH --job-name=dinov3_embed
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=1-8%2

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
PYTHON="${PYTHON:-python}"
MISMATCH_ARGS=()
if [[ "${DINO_IGNORE_MISMATCHED_SIZES:-0}" == "1" || "${DINO_IGNORE_MISMATCHED_SIZES:-0}" == "true" ]]; then
  MISMATCH_ARGS+=(--ignore-mismatched-sizes)
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm
printf 'host: %s\n' "$(hostname)"
printf 'SLURM_ARRAY_TASK_ID: %s\n' "${SLURM_ARRAY_TASK_ID:-unset}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader || true

"${PYTHON}" dinov3_pipeline.py \
  --stage embed \
  --city-meta "${CITY_META}" \
  --city-index "${SLURM_ARRAY_TASK_ID}" \
  --repo-dir "${REPO_DIR}" \
  --model-name "${MODEL_NAME}" \
  --backend "${DINO_BACKEND:-transformers}" \
  --device "${DINO_DEVICE:-cuda}" \
  --batch-size "${DINO_BATCH_SIZE:-64}" \
  "${MISMATCH_ARGS[@]}" \
  --execute
