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

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${PYTHON:-python}"
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

"${PYTHON}" sample_similar_pairs/sample_image_pairs_faiss.py \
  --city-pairs "${CITY_PAIR_ARGS[@]}" \
  --embedding-root "${EMBEDDING_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}" \
  --rootfolder "${ROOTFOLDER}" \
  --train-test-folder "${TRAIN_TEST_FOLDER:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8}" \
  --res-exclude "${RES_EXCLUDE:-None}" \
  --h3-resolution "${H3_RESOLUTION:-8}" \
  --max-images-per-h3 "${MAX_IMAGES_PER_H3:-20}" \
  --max-images-per-city "${MAX_IMAGES_PER_CITY:-50000}" \
  --top-k "${TOP_K:-30}" \
  --threshold "${DINO_THRESHOLD:-0.85}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-2048}" \
  --max-pairs-per-source-image "${MAX_PAIRS_PER_SOURCE_IMAGE:-1}" \
  --max-pairs-per-hex-pair "${MAX_PAIRS_PER_HEX_PAIR:-1}" \
  --pairs-per-city-pair "${PAIRS_PER_CITY_PAIR:-100}" \
  --output "${OUTPUT:-sample_similar_pairs/output/dinov3_image_pairs.parquet}"
