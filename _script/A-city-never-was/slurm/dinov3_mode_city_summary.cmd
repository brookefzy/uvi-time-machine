#!/usr/bin/env bash
#SBATCH --job-name=dinov3_mode_summary
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"
if [[ ! -d "${REPO_DIR}" ]]; then printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2; exit 2; fi
if [[ ! -x "${PYTHON}" ]]; then printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2; exit 127; fi
cd "${REPO_DIR}"; mkdir -p logs/slurm
"${PYTHON}" "${REPO_DIR}/stage2_dino_modality/08_summarize_mode_citypairs.py" "$@"
