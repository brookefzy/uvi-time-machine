#!/usr/bin/env bash
#SBATCH --job-name=dinov3_gallery
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
VENV_PYTHON="${VENV_PYTHON:-${REPO_DIR}/../../.venv/bin/python}"
PYTHON="${PYTHON:-${VENV_PYTHON}}"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python interpreter is not executable: %s\nSet VENV_PYTHON or PYTHON explicitly.\n' "${PYTHON}" >&2
  exit 127
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm sample_similar_pairs/output

"${PYTHON}" sample_similar_pairs/build_image_pair_gallery.py \
  --pairs "${PAIRS:-sample_similar_pairs/output/dinov3_image_pairs.parquet}" \
  --image-index-root "${IMAGE_INDEX_ROOT:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}" \
  --output-dir "${OUTPUT_DIR:-sample_similar_pairs/output/image_gallery}"
