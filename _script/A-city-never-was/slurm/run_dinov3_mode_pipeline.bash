#!/usr/bin/env bash
# Submit global-mode stages in dependency order.  Array submitters wait for each batch.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50}"
IMAGE_INDEX_ROOT="${IMAGE_INDEX_ROOT:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}"
LANDUSE_TIERS_PATH="${LANDUSE_TIERS_PATH:-${ROOTFOLDER}/_curated/03_similarity_grid/landuse_tiers/h3_landuse_tiers_pct10_res=8.csv}"
MODE_MAX_TRAINING_IMAGES_PER_CITY="${MODE_MAX_TRAINING_IMAGES_PER_CITY:-2000}"
MODE_MAX_TRAINING_IMAGES_PER_H3="${MODE_MAX_TRAINING_IMAGES_PER_H3:-5}"
MODE_STRATUM_WEIGHTS="${MODE_STRATUM_WEIGHTS:-core=.4,suburban=.3,occupied_rural=.2,no_poi=.1}"
MODE_TRAINING_SAMPLING_SEED="${MODE_TRAINING_SAMPLING_SEED:-42}"
MODE_K_VALUES="${MODE_K_VALUES:-32 64 128}"
MODE_GALLERY_MEMORY="${MODE_GALLERY_MEMORY:-128G}"
read -r -a MODE_K_ARGS <<< "${MODE_K_VALUES}"
(( ${#MODE_K_ARGS[@]} > 0 )) || { printf 'MODE_K_VALUES must contain at least one K.\n' >&2; exit 2; }
export SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:--1}"
PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"
[[ -d "${REPO_DIR}" ]] || { printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2; exit 127; }
[[ -f "${LANDUSE_TIERS_PATH}" ]] || { printf 'POI-derived land-use tier CSV does not exist: %s\n' "${LANDUSE_TIERS_PATH}" >&2; exit 2; }
cd "${REPO_DIR}"; mkdir -p logs/slurm
LAST_CITY="${LAST_CITY:-$(( $(wc -l < "${CITY_META}") - 2 ))}"
SELECTED_MODEL="${SELECTED_MODEL:-${MODE_OUTPUT_ROOT}/selected_model.json}"
TRAINING_POOL_AUDIT="${MODE_OUTPUT_ROOT}/training_pool_audit.json"

training_config_matches() {
  "${PYTHON}" -c 'import json,sys; from pathlib import Path; from stage2_dino_modality.landuse_tiers import expected_training_config,training_config_matches; audit=json.load(open(sys.argv[1])); expected=expected_training_config(Path(sys.argv[2]),max_images_per_city=int(sys.argv[3]),max_images_per_h3=int(sys.argv[4]),stratum_weights=sys.argv[5],sampling_seed=int(sys.argv[6]),requested_k_values=[int(x) for x in sys.argv[7].split()]); sys.exit(not training_config_matches(audit,expected))' "${TRAINING_POOL_AUDIT}" "${LANDUSE_TIERS_PATH}" "${MODE_MAX_TRAINING_IMAGES_PER_CITY}" "${MODE_MAX_TRAINING_IMAGES_PER_H3}" "${MODE_STRATUM_WEIGHTS}" "${MODE_TRAINING_SAMPLING_SEED}" "${MODE_K_VALUES}"
}

all_city_artifacts_exist() {
  "${PYTHON}" -c 'import csv,sys; from pathlib import Path; root=Path(sys.argv[2]); cities=[r["City"] for r in csv.DictReader(open(sys.argv[1],newline=""))]; sys.exit(not all((root/f"city={city}.parquet").exists() for city in cities))' "${CITY_META}" "$1"
}

all_pair_artifacts_exist() {
  "${PYTHON}" -c 'import sys; from pathlib import Path; root=Path(sys.argv[2]); pairs=[x.split("|") for x in Path(sys.argv[1]).read_text().splitlines() if x]; sys.exit(not all((root/f"city_1={a}"/f"city_2={b}"/"part_res=8.parquet").exists() for a,b in pairs))' "${PAIR_MANIFEST}" "$1"
}

if [[ "${RESUME:-1}" != "1" ]] || ! all_city_artifacts_exist "${MODE_OUTPUT_ROOT}/sampled_images"; then
  LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_sample_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
fi
if [[ "${RESUME:-1}" == "1" && -f "${MODE_OUTPUT_ROOT}/scorecard.parquet" ]]; then
  [[ -f "${TRAINING_POOL_AUDIT}" ]] || { printf 'Existing scorecard has no POI-stratified training audit. Use a new MODE_OUTPUT_ROOT.\n' >&2; exit 2; }
  training_config_matches || { printf 'Existing scorecard uses a different POI tier file or training configuration. Use a new MODE_OUTPUT_ROOT.\n' >&2; exit 2; }
else
  sbatch --wait slurm/dinov3_mode_fit_codebooks.cmd \
    --input "${MODE_OUTPUT_ROOT}/sampled_images" \
    --output-root "${MODE_OUTPUT_ROOT}" \
    --landuse-tiers "${LANDUSE_TIERS_PATH}" \
    --k "${MODE_K_ARGS[@]}" \
    --max-training-images-per-city "${MODE_MAX_TRAINING_IMAGES_PER_CITY}" \
    --max-training-images-per-h3 "${MODE_MAX_TRAINING_IMAGES_PER_H3}" \
    --stratum-weights "${MODE_STRATUM_WEIGHTS}" \
    --training-sampling-seed "${MODE_TRAINING_SAMPLING_SEED}"
fi

if [[ -z "${SELECTED_K:-}" ]]; then
  if [[ -n "${IMAGE_INDEX_ROOT}" ]]; then
    for CENTROIDS in "${MODE_OUTPUT_ROOT}"/codebook_candidates/k=*/centroids.parquet; do
      [[ -f "${CENTROIDS}" ]] || continue
      K="$(basename "$(dirname "${CENTROIDS}")")"; K="${K#k=}"
      sbatch --wait --mem="${MODE_GALLERY_MEMORY}" slurm/dinov3_mode_gallery.cmd --sampled "${MODE_OUTPUT_ROOT}/sampled_images" --centroids "${CENTROIDS}" --image-index "${IMAGE_INDEX_ROOT}" --landuse-tiers "${LANDUSE_TIERS_PATH}" --output "${MODE_OUTPUT_ROOT}/mode_gallery/k=${K}/index.html"
    done
  fi
  printf 'Candidate scorecard is ready at %s. Review galleries and rerun with SELECTED_K set.\n' "${MODE_OUTPUT_ROOT}/scorecard.parquet"
  exit 0
fi

MODEL_ID="$("${PYTHON}" -c 'import pandas as pd,sys; x=pd.read_parquet(sys.argv[1]); row=x[(x.k==int(sys.argv[2])) & (x.status=="ok")]; assert len(row)==1, "selected K must have exactly one valid scorecard row"; print(row.iloc[0].model_id)' "${MODE_OUTPUT_ROOT}/scorecard.parquet" "${SELECTED_K}")"
if [[ -f "${SELECTED_MODEL}" ]]; then
  EXISTING_MODEL_ID="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' "${SELECTED_MODEL}")"
  [[ "${EXISTING_MODEL_ID}" == "${MODEL_ID}" ]] || { printf 'Selected model conflicts with requested K/model: %s\n' "${SELECTED_MODEL}" >&2; exit 2; }
elif [[ "${RESUME:-1}" != "1" || ! -f "${SELECTED_MODEL}" ]]; then
  sbatch --wait slurm/dinov3_mode_select.cmd --scorecard "${MODE_OUTPUT_ROOT}/scorecard.parquet" --selected-k "${SELECTED_K}" --model-id "${MODEL_ID}" --output "${SELECTED_MODEL}"
fi

if [[ "${RESUME:-1}" != "1" ]] || ! all_city_artifacts_exist "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/assignments"; then
  LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_assign_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
fi
HISTOGRAM_ROOT="${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms"
EXISTING_HISTOGRAMS=()
if [[ "${ALLOW_MISSING_CITIES:-0}" == "1" ]]; then
  shopt -s nullglob
  EXISTING_HISTOGRAMS=("${HISTOGRAM_ROOT}"/city=*.parquet)
  shopt -u nullglob
fi
if [[ "${ALLOW_MISSING_CITIES:-0}" == "1" && "${RESUME:-1}" == "1" && ${#EXISTING_HISTOGRAMS[@]} -gt 0 ]]; then
  printf 'Accepting existing histogram subset because ALLOW_MISSING_CITIES=1 (%s files).\n' "${#EXISTING_HISTOGRAMS[@]}"
elif [[ "${RESUME:-1}" != "1" ]] || ! all_city_artifacts_exist "${HISTOGRAM_ROOT}"; then
  LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_histogram_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
fi

PAIR_MANIFEST="${PAIR_MANIFEST:-${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/pair_manifest.txt}"
MANIFEST_ARGS=(--city-meta "${CITY_META}" --histogram-root "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_histograms" --expected-model-id "${MODEL_ID}" --output "${PAIR_MANIFEST}")
if [[ "${ALLOW_MISSING_CITIES:-0}" == "1" ]]; then
  MANIFEST_ARGS+=(--allow-missing --available-cities-output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/available_cities.txt" --skipped-cities-output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/skipped_cities.txt")
fi
"${PYTHON}" slurm/generate_dinov3_mode_pair_manifest.py "${MANIFEST_ARGS[@]}"
PAIR_COUNT="$(wc -l < "${PAIR_MANIFEST}")"
if (( PAIR_COUNT > 0 )); then
  if [[ "${RESUME:-1}" != "1" ]] || ! all_pair_artifacts_exist "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_similarity"; then
    PAIR_COUNT="${PAIR_COUNT}" PAIR_MANIFEST="${PAIR_MANIFEST}" JOB_SCRIPT=slurm/dinov3_mode_similarity_array.cmd bash slurm/submit_dinov3_mode_similarity_batches.bash
  fi
fi
sbatch --wait slurm/dinov3_mode_city_summary.cmd --input "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/h3_similarity" --pair-manifest "${PAIR_MANIFEST}" --output "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/city_pair_summary.parquet"
