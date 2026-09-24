#!/usr/bin/env bash
#SBATCH --job-name=dinov3_b5c_city
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
cd "${UVI_SAMPLE_REPO_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROW=$(( ${CITY_OFFSET:-0} + ${SLURM_ARRAY_TASK_ID:?} ))
CITY=$(sed -n "${ROW}p" "${B5C_CITY_MANIFEST:?}")
[[ -n "$CITY" ]] || { echo "Empty city at row $ROW" >&2; exit 2; }
STATE="$B5C_RUN_DIR/city_$ROW"
mkdir -p "$STATE"
"${VENV_PYTHON:?}" B5c_pairwise_agg_optimized.py \
  --resolution "${RESOLUTION:?}" --city-meta "${CITY_META:?}" --city "$CITY" \
  --pairwise-root "${PAIRWISE_ROOT:?}" --h3-membership-root "${H3_MEMBERSHIP_ROOT:?}" \
  --export-folder "${SIMILARITY_EXPORT_FOLDER:?}" \
  --agg-progress-file "$STATE/progress.json" --audit-report "$STATE/audit.json" \
  --unresolved-folder "$STATE/unresolved" --duckdb-temp-dir "$STATE/spill" \
  --duckdb-memory-limit "${B5C_MEMORY_LIMIT:-64GB}" --duckdb-threads 1 --resume
