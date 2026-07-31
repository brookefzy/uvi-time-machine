#!/usr/bin/env bash
# Submit global-mode stages in dependency order.  Array submitters wait for each batch.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:?Set UVI_SAMPLE_REPO_DIR}"
cd "${REPO_DIR}"
CITY_META="${CITY_META:?Set CITY_META}"
MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:?Set MODE_OUTPUT_ROOT}"
LAST_CITY="${LAST_CITY:-$(( $(wc -l < "${CITY_META}") - 1 ))}"
SELECTED_MODEL="${SELECTED_MODEL:-${MODE_OUTPUT_ROOT}/selected_model.json}"

LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_sample_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
sbatch --wait slurm/dinov3_mode_fit_codebooks.cmd --input "${MODE_OUTPUT_ROOT}/sampled_images" --output-root "${MODE_OUTPUT_ROOT}"

if [[ -z "${SELECTED_K:-}" ]]; then
  printf 'Candidate scorecard is ready at %s. Review galleries and rerun with SELECTED_K set.\n' "${MODE_OUTPUT_ROOT}/scorecard.parquet"
  exit 0
fi

MODEL_ID="$("${VENV_PYTHON:-$(cd ../.. && pwd)/.venv/bin/python}" -c 'import pandas as pd,sys; x=pd.read_parquet(sys.argv[1]); row=x[(x.k==int(sys.argv[2])) & (x.status=="ok")]; assert len(row)==1, "selected K must have exactly one valid scorecard row"; print(row.iloc[0].model_id)' "${MODE_OUTPUT_ROOT}/scorecard.parquet" "${SELECTED_K}")"
sbatch --wait slurm/dinov3_mode_select.cmd --scorecard "${MODE_OUTPUT_ROOT}/scorecard.parquet" --selected-k "${SELECTED_K}" --model-id "${MODEL_ID}" --output "${SELECTED_MODEL}"

LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_assign_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_histogram_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash

PAIR_MANIFEST="${PAIR_MANIFEST:-${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/pair_manifest.txt}"
"${VENV_PYTHON:-$(cd ../.. && pwd)/.venv/bin/python}" slurm/generate_dinov3_mode_pair_manifest.py --city-meta "${CITY_META}" --histogram-root "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms" --output "${PAIR_MANIFEST}"
PAIR_COUNT="$(wc -l < "${PAIR_MANIFEST}")"
if (( PAIR_COUNT > 0 )); then
  PAIR_COUNT="${PAIR_COUNT}" PAIR_MANIFEST="${PAIR_MANIFEST}" JOB_SCRIPT=slurm/dinov3_mode_similarity_array.cmd bash slurm/submit_dinov3_mode_similarity_batches.bash
fi
sbatch --wait slurm/dinov3_mode_city_summary.cmd --input "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_similarity" --output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/city_pair_summary.parquet"
