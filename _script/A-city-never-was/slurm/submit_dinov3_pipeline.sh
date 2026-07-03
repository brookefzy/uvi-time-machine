#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
MODE="${MODE:-city-batch}"
CITY_ARRAY_RANGE="${CITY_ARRAY_RANGE:-1-8%2}"
H3_ARRAY_RANGE="${H3_ARRAY_RANGE:-${CITY_ARRAY_RANGE}}"
RUN_SMOKE="${RUN_SMOKE:-0}"
VERIFY_SMOKE="${VERIFY_SMOKE:-1}"

export REPO_DIR CITY_META MODEL_NAME

cd "${REPO_DIR}"
mkdir -p logs/slurm

if [[ "${MODE}" == "city-batch" ]]; then
  smoke_job="skipped"
  verify_job="skipped"
  embed_dependency=()

  if [[ "${RUN_SMOKE}" == "1" || "${RUN_SMOKE}" == "true" ]]; then
    smoke_job="$(sbatch --parsable slurm/dinov3_00_smoke.cmd)"
    verify_dependency=(--dependency=afterok:${smoke_job})
  else
    verify_dependency=()
  fi

  if [[ "${VERIFY_SMOKE}" == "1" || "${VERIFY_SMOKE}" == "true" ]]; then
    verify_job="$(sbatch --parsable "${verify_dependency[@]}" slurm/dinov3_00_verify_smoke.cmd)"
    embed_dependency=(--dependency=afterok:${verify_job})
  elif [[ "${RUN_SMOKE}" == "1" || "${RUN_SMOKE}" == "true" ]]; then
    embed_dependency=(--dependency=afterok:${smoke_job})
  fi

  embed_job="$(sbatch --parsable --array="${CITY_ARRAY_RANGE}" "${embed_dependency[@]}" slurm/dinov3_01_embed_array.cmd)"
  h3_job="$(sbatch --parsable --array="${H3_ARRAY_RANGE}" --dependency=afterok:${embed_job} slurm/dinov3_02_h3_array.cmd)"

  printf 'Submitted DINOv3 city batch jobs:\n'
  printf '  city array range: %s\n' "${CITY_ARRAY_RANGE}"
  printf '  h3 array range:   %s\n' "${H3_ARRAY_RANGE}"
  printf '  smoke:            %s\n' "${smoke_job}"
  printf '  verify smoke:     %s\n' "${verify_job}"
  printf '  embed:            %s\n' "${embed_job}"
  printf '  h3:               %s\n' "${h3_job}"
  printf '\nAfter all city batches finish successfully, run:\n'
  printf '  MODE=global bash slurm/submit_dinov3_pipeline.sh\n'
  exit 0
fi

if [[ "${MODE}" == "global" ]]; then
  pairwise_job="$(sbatch --parsable slurm/dinov3_03_pairwise.cmd)"
  b5c_job="$(sbatch --parsable --dependency=afterok:${pairwise_job} slurm/dinov3_04_b5c_aggregate.cmd)"
  summary_job="$(sbatch --parsable --dependency=afterok:${b5c_job} slurm/dinov3_05_summary.cmd)"

  printf 'Submitted DINOv3 global jobs:\n'
  printf '  pairwise: %s\n' "${pairwise_job}"
  printf '  b5c:      %s\n' "${b5c_job}"
  printf '  summary:  %s\n' "${summary_job}"
  exit 0
fi

printf 'Unsupported MODE=%s. Use MODE=city-batch or MODE=global.\n' "${MODE}" >&2
exit 2
