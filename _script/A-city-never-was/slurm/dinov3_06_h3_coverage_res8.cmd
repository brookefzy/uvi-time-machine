#!/usr/bin/env bash
#SBATCH --job-name=dinov3_h3_cov8
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# This is deliberately a single Slurm job (not an array), so it uses one
# submitted/running-job slot and remains within the account concurrency cap.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"

if [[ ! -d "${REPO_DIR}" ]]; then
  printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2
  exit 127
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm

"${PYTHON}" "${REPO_DIR}/summarize_dinov3_h3_coverage.py" \
  --city-meta "${CITY_META}" \
  --h3-root "${H3_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}" \
  --res-exclude None \
  --resolutions 8 \
  --output-csv "${OUTPUT_CSV:-logs/dinov3_h3_coverage_res8.csv}" \
  --output-json "${OUTPUT_JSON:-logs/dinov3_h3_coverage_res8.json}" \
  --allow-missing
