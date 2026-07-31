#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${UVI_SAMPLE_REPO_DIR:?Set UVI_SAMPLE_REPO_DIR}"; REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"; PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then exit 127; fi
MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:?Set MODE_OUTPUT_ROOT}"; SELECTED_MODEL="${SELECTED_MODEL:-${MODE_OUTPUT_ROOT}/selected_model.json}"; PAIR_MANIFEST="${PAIR_MANIFEST:?Set PAIR_MANIFEST}"
PAIR="$("${PYTHON}" -c 'import sys; rows=open(sys.argv[1]).read().splitlines(); print(rows[int(sys.argv[2])-1])' "${PAIR_MANIFEST}" "${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID}")"
IFS='|' read -r CITY_1 CITY_2 <<< "${PAIR}"
if [[ -z "${CITY_1}" || -z "${CITY_2}" || "${CITY_1}" == "${CITY_2}" ]]; then printf 'Invalid city pair: %s\n' "${PAIR}" >&2; exit 2; fi
MODEL_ID="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' "${SELECTED_MODEL}")"
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/07_compute_h3_mode_js_similarity.py" --source "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms/city=${CITY_1}.parquet" --target "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms/city=${CITY_2}.parquet" --output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_similarity/city_1=${CITY_1}/city_2=${CITY_2}/part_res=8.parquet" --threshold "${SIMILARITY_THRESHOLD:--1}" --row-block-size "${SIMILARITY_ROW_BLOCK_SIZE:-64}" --target-block-size "${SIMILARITY_TARGET_BLOCK_SIZE:-2048}"
