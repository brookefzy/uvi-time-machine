#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "${UVI_SAMPLE_REPO_DIR:?}/../.." && pwd)"; PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then exit 127; fi
"${PYTHON}" "${UVI_SAMPLE_REPO_DIR}/stage2_dino_modality/02_fit_evaluate_codebooks.py" "$@"
