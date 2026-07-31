#!/usr/bin/env bash
#SBATCH --job-name=dinov3_mode_js
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"; PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"
MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50}"; SELECTED_MODEL="${SELECTED_MODEL:-${MODE_OUTPUT_ROOT}/selected_model.json}"
if [[ ! -d "${REPO_DIR}" ]]; then printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2; exit 2; fi
if [[ ! -x "${PYTHON}" ]]; then printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2; exit 127; fi
cd "${REPO_DIR}"; mkdir -p logs/slurm
MODEL_ID="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' "${SELECTED_MODEL}")"; PAIR_MANIFEST="${PAIR_MANIFEST:-${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/pair_manifest.txt}"
PAIR="$("${PYTHON}" -c 'import sys; rows=open(sys.argv[1]).read().splitlines(); print(rows[int(sys.argv[2])-1])' "${PAIR_MANIFEST}" "${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID}")"; IFS='|' read -r CITY_1 CITY_2 <<< "${PAIR}"
[[ -n "${CITY_1}" && -n "${CITY_2}" && "${CITY_1}" != "${CITY_2}" ]] || { printf 'Invalid city pair: %s\n' "${PAIR}" >&2; exit 2; }
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/07_compute_h3_mode_js_similarity.py" --source "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms/city=${CITY_1}.parquet" --target "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms/city=${CITY_2}.parquet" --output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_similarity/city_1=${CITY_1}/city_2=${CITY_2}/part_res=8.parquet" --threshold "${SIMILARITY_THRESHOLD:--1}" --row-block-size "${SIMILARITY_ROW_BLOCK_SIZE:-64}" --target-block-size "${SIMILARITY_TARGET_BLOCK_SIZE:-2048}"
