#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${UVI_SAMPLE_REPO_DIR:?Set UVI_SAMPLE_REPO_DIR}"; REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"; PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then exit 127; fi
CITY_META="${CITY_META:?Set CITY_META}"; MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:?Set MODE_OUTPUT_ROOT}"; SELECTED_MODEL="${SELECTED_MODEL:-${MODE_OUTPUT_ROOT}/selected_model.json}"
CITY="$("${PYTHON}" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], newline=""))); print(rows[int(sys.argv[2])]["City"])' "${CITY_META}" "${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID}")"
MODEL_ID="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' "${SELECTED_MODEL}")"
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/06_build_h3_mode_histograms.py" --input "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/assignments/city=${CITY}.parquet" --output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms/city=${CITY}.parquet"
