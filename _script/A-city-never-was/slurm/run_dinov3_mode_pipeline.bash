#!/usr/bin/env bash
# Coordinator intentionally pauses after gallery/model recommendation for visual review.
set -euo pipefail
REPO_DIR="${UVI_SAMPLE_REPO_DIR:?Set UVI_SAMPLE_REPO_DIR}"
cd "${REPO_DIR}"
bash slurm/submit_dinov3_mode_city_batches.bash
sbatch slurm/dinov3_mode_fit_codebooks.cmd
sbatch slurm/dinov3_mode_gallery.cmd
if [[ -z "${SELECTED_K:-}" ]]; then
  printf 'Gallery/recommendation submitted. Review it, then rerun with SELECTED_K set.\n'
  exit 0
fi
sbatch slurm/dinov3_mode_select.cmd --selected-k "${SELECTED_K}"
bash slurm/submit_dinov3_mode_city_batches.bash
bash slurm/submit_dinov3_mode_similarity_batches.bash
sbatch slurm/dinov3_mode_city_summary.cmd
