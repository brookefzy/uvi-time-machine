#!/usr/bin/env bash
#SBATCH --job-name=dino_r7_embed
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

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${CITY_MANIFEST:?CITY_MANIFEST is required}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${PYTHON:-python}"
VALFOLDER="${VALFOLDER:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}"
EMBED_ROOT="${EMBED_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
ROW="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CITY_MANIFEST}")"
IFS='|' read -r CITY CITY_INDEX_STEM CITY_INDEX_ROOT CITY_OUTPUT_STEM <<< "${ROW}"
if [[ -z "${CITY}" || -z "${CITY_INDEX_STEM}" || -z "${CITY_INDEX_ROOT}" || -z "${CITY_OUTPUT_STEM}" ]]; then
  printf 'Invalid city manifest row %s: %s\n' "${SLURM_ARRAY_TASK_ID}" "${ROW}" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" B5d_dinov3_embed_city.py \
  --city "${CITY}" \
  --city-file-stem "${CITY_INDEX_STEM}" \
  --output-city-file-stem "${CITY_OUTPUT_STEM}" \
  --valfolder "${CITY_INDEX_ROOT}" \
  --output-root "${EMBED_ROOT}" \
  --year-metadata-root "${ROOTFOLDER}" \
  --model-name "${MODEL_NAME}" \
  --backend "${DINO_BACKEND:-transformers}" \
  --batch-size "${DINO_BATCH_SIZE:-64}" \
  --device "${DINO_DEVICE:-cuda}" \
  --local-files-only
