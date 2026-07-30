#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${UVI_SAMPLE_REPO_DIR:?}"; REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"; PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then exit 127; fi
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/01_sample_h3_images.py" "$@"
