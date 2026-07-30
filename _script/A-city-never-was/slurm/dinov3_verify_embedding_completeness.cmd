#!/usr/bin/env bash
#SBATCH --job-name=dinov3_embed_check
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# This is a single global audit job, rather than an array, so it consumes one
# submitted/running-job slot and respects the account concurrency cap.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
VENV_PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint/dinov3-vitb16-pretrain-lvd1689m}"

if [[ ! -d "${REPO_DIR}" ]]; then
  printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -x "${VENV_PYTHON}" ]]; then
  printf 'Python interpreter is not executable: %s\n' "${VENV_PYTHON}" >&2
  exit 127
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${VENV_PYTHON}" "${REPO_DIR}/verify_dinov3_embedding_completeness.py" \
  --city-meta "${CITY_META}" \
  --valfolder "${VALFOLDER:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}" \
  --output-root "${OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}" \
  --year-metadata-root "${YEAR_METADATA_ROOT:-${ROOTFOLDER}}" \
  --expected-model-name "${MODEL_NAME}" \
  --output-csv "${OUTPUT_CSV:-logs/dinov3_embedding_completeness.csv}" \
  --output-json "${OUTPUT_JSON:-logs/dinov3_embedding_completeness.json}"
