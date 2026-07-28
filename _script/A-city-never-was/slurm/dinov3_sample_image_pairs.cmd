#!/usr/bin/env bash
#SBATCH --job-name=dinov3_img_pairs
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=ALL

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
if [[ ! -d "${REPO_DIR}" ]]; then
  printf 'Repository directory does not exist: %s\nSet REPO_DIR explicitly.\n' "${REPO_DIR}" >&2
  exit 2
fi
REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
PYTHON="${VENV_PYTHON}"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python interpreter is not executable: %s\nSet VENV_PYTHON explicitly.\n' "${PYTHON}" >&2
  exit 127
fi
printf 'Using Python: %s\n' "${PYTHON}"
CITY_PAIRS_DEFAULT=(
  "Paris|London"
  "London|Hong Kong"
  "Hong Kong|Singapore"
  "London|Sydney"
  "New York|London"
)
if [[ -n "${CITY_PAIRS:-}" ]]; then
  IFS=';' read -r -a CITY_PAIR_ARGS <<< "${CITY_PAIRS}"
else
  CITY_PAIR_ARGS=("${CITY_PAIRS_DEFAULT[@]}")
fi

cd "${REPO_DIR}"
mkdir -p logs/slurm sample_similar_pairs/output
"${PYTHON}" -c 'import faiss; print("FAISS", getattr(faiss, "__version__", "available"))'

"${PYTHON}" "${REPO_DIR}/sample_similar_pairs/sample_image_pairs_faiss.py" \
  --city-pairs "${CITY_PAIR_ARGS[@]}" \
  --embedding-root "${EMBEDDING_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}" \
  --rootfolder "${ROOTFOLDER}" \
  --train-test-folder "${TRAIN_TEST_FOLDER:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8}" \
  --res-exclude "${RES_EXCLUDE:-None}" \
  --h3-resolution "${H3_RESOLUTION:-8}" \
  --max-images-per-h3 "${MAX_IMAGES_PER_H3:-100}" \
  --max-images-per-city "${MAX_IMAGES_PER_CITY:-0}" \
  --top-k "${TOP_K:-30}" \
  --threshold "${DINO_THRESHOLD:--1.0}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-2048}" \
  --max-pairs-per-source-image "${MAX_PAIRS_PER_SOURCE_IMAGE:-0}" \
  --max-pairs-per-hex-pair "${MAX_PAIRS_PER_HEX_PAIR:-0}" \
  --pairs-per-city-pair "${PAIRS_PER_CITY_PAIR:-10}" \
  --output "${OUTPUT:-sample_similar_pairs/output/dinov3_image_pairs.parquet}"
