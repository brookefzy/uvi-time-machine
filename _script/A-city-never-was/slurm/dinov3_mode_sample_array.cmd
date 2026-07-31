#!/usr/bin/env bash
#SBATCH --job-name=dinov3_mode_sample
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"
EMBEDDING_ROOT="${EMBEDDING_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}"
TRAIN_TEST_FOLDER="${TRAIN_TEST_FOLDER:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8}"
MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50}"
if [[ ! -d "${REPO_DIR}" ]]; then printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2; exit 2; fi
if [[ ! -x "${PYTHON}" ]]; then printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2; exit 127; fi
cd "${REPO_DIR}"; mkdir -p logs/slurm
CITY="$("${PYTHON}" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], newline=""))); print(rows[int(sys.argv[2])]["City"])' "${CITY_META}" "${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID}")"
[[ -n "${CITY}" ]] || { printf 'Invalid city array index\n' >&2; exit 2; }
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/01_sample_h3_images.py" --city "${CITY}" --embedding-root "${EMBEDDING_ROOT}" --rootfolder "${ROOTFOLDER}" --train-test-folder "${TRAIN_TEST_FOLDER}" --output "${MODE_OUTPUT_ROOT}/sampled_images/city=${CITY}.parquet"
