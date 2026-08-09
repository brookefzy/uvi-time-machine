#!/usr/bin/env bash
#SBATCH --job-name=dinov3_b5c
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
PYTHON="${PYTHON:-python}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
RESOLUTION="${RESOLUTION:-7}"
PAIRWISE_ROOT="${PAIRWISE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair"
SIMILARITY_EXPORT_FOLDER="${SIMILARITY_EXPORT_FOLDER:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_res=${RESOLUTION}}"
H3_MEMBERSHIP_ROOT="${H3_MEMBERSHIP_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" dinov3_pipeline.py \
  --stage b5c \
  --city-meta "${CITY_META}" \
  --repo-dir "${REPO_DIR}" \
  --resolution "${RESOLUTION}" \
  --pairwise-root "${PAIRWISE_ROOT}" \
  --hex-root "${H3_MEMBERSHIP_ROOT}" \
  --similarity-export-folder "${SIMILARITY_EXPORT_FOLDER}" \
  --b5c-memory-limit "${B5C_MEMORY_LIMIT:-64GB}" \
  --b5c-threads "${B5C_THREADS:-1}" \
  --execute
