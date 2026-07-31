#!/usr/bin/env bash
# Coordinator intentionally pauses after gallery/model recommendation for visual review.
set -euo pipefail
REPO_DIR="${UVI_SAMPLE_REPO_DIR:?Set UVI_SAMPLE_REPO_DIR}"
cd "${REPO_DIR}"
CITY_META="${CITY_META:?Set CITY_META}"; LAST_CITY="${LAST_CITY:-$(( $(wc -l < "${CITY_META}") - 1 ))}"
LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_sample_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
sbatch slurm/dinov3_mode_fit_codebooks.cmd --input "${MODE_OUTPUT_ROOT:?}/sampled_images" --output-root "${MODE_OUTPUT_ROOT}"
sbatch slurm/dinov3_mode_gallery.cmd --representatives "${MODE_OUTPUT_ROOT}/representatives.parquet" --output "${MODE_OUTPUT_ROOT}/mode_gallery/index.html"
if [[ -z "${SELECTED_K:-}" ]]; then
  printf 'Gallery/recommendation submitted. Review it, then rerun with SELECTED_K set.\n'
  exit 0
fi
sbatch slurm/dinov3_mode_select.cmd --selected-k "${SELECTED_K}"
bash slurm/submit_dinov3_mode_city_batches.bash
bash slurm/submit_dinov3_mode_similarity_batches.bash
sbatch slurm/dinov3_mode_city_summary.cmd
