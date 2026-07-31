#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${UVI_SAMPLE_REPO_DIR:?}"; REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"; PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then exit 127; fi
CITY_META="${CITY_META:?Set CITY_META}"; CITY="$("${PYTHON}" -c 'import csv,sys; print(list(csv.DictReader(open(sys.argv[1], newline="")))[int(sys.argv[2])]["City"])' "${CITY_META}" "${SLURM_ARRAY_TASK_ID:?}")"
if [[ -z "${CITY}" || "${CITY}" == "City" ]]; then printf 'Invalid city array index\n' >&2; exit 2; fi
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/01_sample_h3_images.py" --city "${CITY}" --embedding-root "${EMBEDDING_ROOT:?}" --rootfolder "${ROOTFOLDER:?}" --train-test-folder "${TRAIN_TEST_FOLDER:-}" --output "${MODE_OUTPUT_ROOT:?}/sampled_images/city=${CITY}.parquet"
