#!/usr/bin/env bash
#SBATCH --job-name=dino_r7_h3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${CITY_MANIFEST:?CITY_MANIFEST is required}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${PYTHON:-python}"
EMBED_ROOT="${EMBED_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}"
RECOVERY_H3_ROOT="${RECOVERY_H3_ROOT:?RECOVERY_H3_ROOT is required}"
TRAIN_TEST_FOLDER="${TRAIN_TEST_FOLDER:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8}"
ROW="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CITY_MANIFEST}")"
IFS='|' read -r CITY CITY_STEM <<< "${ROW}"
if [[ -z "${CITY}" || -z "${CITY_STEM}" ]]; then
  printf 'Invalid city manifest row %s: %s\n' "${SLURM_ARRAY_TASK_ID}" "${ROW}" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "${REPO_DIR}"
mkdir -p logs/slurm "${RECOVERY_H3_ROOT}"

"${PYTHON}" B5e_dinov3_vector_summary.py \
  --city "${CITY}" \
  --city-file-stem "${CITY_STEM}" \
  --rootfolder "${ROOTFOLDER}" \
  --input-root "${EMBED_ROOT}" \
  --output-root "${RECOVERY_H3_ROOT}" \
  --train-test-folder "${TRAIN_TEST_FOLDER}" \
  --res-exclude None \
  --log-level "${LOG_LEVEL:-INFO}"
