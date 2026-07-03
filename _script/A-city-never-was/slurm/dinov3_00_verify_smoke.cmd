#!/usr/bin/env bash
#SBATCH --job-name=dinov3_verify_smoke
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
PYTHON="${PYTHON:-python}"
SMOKE_CITY="${SMOKE_CITY:-Hong Kong}"
SMOKE_ROOT="${SMOKE_ROOT:-/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed_smoke}"
MIN_SMOKE_ROWS="${MIN_SMOKE_ROWS:-1}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" verify_dinov3_smoke.py \
  --city "${SMOKE_CITY}" \
  --smoke-root "${SMOKE_ROOT}" \
  --expected-model-name "${MODEL_NAME}" \
  --min-rows "${MIN_SMOKE_ROWS}"
