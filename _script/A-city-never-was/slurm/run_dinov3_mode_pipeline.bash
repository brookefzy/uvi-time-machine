#!/usr/bin/env bash
# Submit global-mode stages in dependency order.  Array submitters wait for each batch.
set -euo pipefail

REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50}"
IMAGE_INDEX_ROOT="${IMAGE_INDEX_ROOT:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}"
PYTHON="${VENV_PYTHON:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python}"
[[ -d "${REPO_DIR}" ]] || { printf 'Repository directory does not exist: %s\n' "${REPO_DIR}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { printf 'Python interpreter is not executable: %s\n' "${PYTHON}" >&2; exit 127; }
cd "${REPO_DIR}"; mkdir -p logs/slurm
LAST_CITY="${LAST_CITY:-$(( $(wc -l < "${CITY_META}") - 1 ))}"
SELECTED_MODEL="${SELECTED_MODEL:-${MODE_OUTPUT_ROOT}/selected_model.json}"

all_city_artifacts_exist() {
  "${PYTHON}" -c 'import csv,sys; from pathlib import Path; root=Path(sys.argv[2]); cities=[r["City"] for r in csv.DictReader(open(sys.argv[1],newline=""))]; sys.exit(not all((root/f"city={city}.parquet").exists() for city in cities))' "${CITY_META}" "$1"
}

all_pair_artifacts_exist() {
  "${PYTHON}" -c 'import sys; from pathlib import Path; root=Path(sys.argv[2]); pairs=[x.split("|") for x in Path(sys.argv[1]).read_text().splitlines() if x]; sys.exit(not all((root/f"city_1={a}"/f"city_2={b}"/"part_res=8.parquet").exists() for a,b in pairs))' "${PAIR_MANIFEST}" "$1"
}

if [[ "${RESUME:-1}" != "1" ]] || ! all_city_artifacts_exist "${MODE_OUTPUT_ROOT}/sampled_images"; then
  LAST_CITY="${LAST_CITY}" JOB_SCRIPT=slurm/dinov3_mode_sample_array.cmd bash slurm/submit_dinov3_mode_city_batches.bash
fi
if [[ "${RESUME:-1}" != "1" || ! -f "${MODE_OUTPUT_ROOT}/scorecard.parquet" ]]; then
  sbatch --wait slurm/dinov3_mode_fit_codebooks.cmd --input "${MODE_OUTPUT_ROOT}/sampled_images" --output-root "${MODE_OUTPUT_ROOT}"
fi

if [[ -z "${SELECTED_K:-}" ]]; then
  if [[ -n "${IMAGE_INDEX_ROOT}" ]]; then
    for CENTROIDS in "${MODE_OUTPUT_ROOT}"/codebook_candidates/k=*/centroids.parquet; do
      [[ -f "${CENTROIDS}" ]] || continue
      K="$(basename "$(dirname "${CENTROIDS}")")"; K="${K#k=}"
      sbatch --wait slurm/dinov3_mode_gallery.cmd --sampled "${MODE_OUTPUT_ROOT}/sampled_images" --centroids "${CENTROIDS}" --image-index "${IMAGE_INDEX_ROOT}" --output "${MODE_OUTPUT_ROOT}/mode_gallery/k=${K}/index.html"
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
