#!/usr/bin/env bash
# Audit and recover affected DINOv3 resolution-7 cities without mutating unaffected artifacts.
set -euo pipefail

EXECUTE=0
RUN_ROOT_OVERRIDE=""
while (($#)); do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --run-root)
      shift
      RUN_ROOT_OVERRIDE="${1:?--run-root requires a path}"
      ;;
    --help|-h)
      printf 'Usage: %s [--execute] [--run-root PATH]\n' "$0"
      printf 'Without --execute, run the forensic preflight and write manifests only.\n'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
ROOTFOLDER="${ROOTFOLDER:-/lustre1/g/geog_pyloo/05_timemachine}"
PYTHON="${PYTHON:-python}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
RESOLUTION=7
INDEX_ROOTS="${INDEX_ROOTS:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}"
VALFOLDER="${VALFOLDER:-${INDEX_ROOTS%%:*}}"
EMBED_ROOT="${EMBED_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_embed}"
ORIGINAL_H3_ROOT="${ORIGINAL_H3_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_hex_summary}"
ORIGINAL_PAIRWISE_ROOT="${ORIGINAL_PAIRWISE_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_by_pair_res=7}"
ORIGINAL_AGG_ROOT="${ORIGINAL_AGG_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_similarity_res=7}"
REQUIRED_H3_ROOT="${REQUIRED_H3_ROOT:-}"
CORE_H3_ROOT="${CORE_H3_ROOT:-}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${ROOTFOLDER}/_tmp/dinov3_res7_recovery/${RUN_STAMP}}"
MANIFEST_DIR="${RUN_ROOT}/manifests"
AUDIT_DIR="${RUN_ROOT}/audit"
LOG_DIR="${RUN_ROOT}/logs"
RECOVERY_H3_ROOT="${RUN_ROOT}/h3_recovered"
H3_OVERLAY_ROOT="${RUN_ROOT}/h3_overlay"
RECOVERY_PAIRWISE_ROOT="${RUN_ROOT}/pairwise_recovered"
PAIRWISE_OVERLAY_ROOT="${RUN_ROOT}/pairwise_overlay"
FINAL_ROOT="${RUN_ROOT}/final"
JOB_LOG="${LOG_DIR}/slurm_jobs.tsv"
COMMAND_LOG="${LOG_DIR}/commands.log"
PRE_AUDIT="${AUDIT_DIR}/before.json"
SOURCE_AUDIT="${AUDIT_DIR}/after_index_recovery.json"
POST_EMBED_AUDIT="${AUDIT_DIR}/after_embedding.json"
POST_H3_AUDIT="${AUDIT_DIR}/after_h3.json"
RECOVERED_INDEX_ROOT="${RUN_ROOT}/recovered_indices"
EMBED_MANIFEST="${MANIFEST_DIR}/embed.txt"
H3_MANIFEST="${MANIFEST_DIR}/h3.txt"
ABSENT_CITIES="${MANIFEST_DIR}/source_imagery_absent.txt"
AFFECTED_PAIRS="${MANIFEST_DIR}/affected_pairs.txt"
PENDING_PAIRS="${MANIFEST_DIR}/affected_pairs_pending.txt"
AFFECTED_CITIES=(Amsterdam Gombe Kampala Kozhikode Malegaon Sitapur Vijayawada)

mkdir -p "${MANIFEST_DIR}" "${AUDIT_DIR}" "${LOG_DIR}" "${RECOVERY_H3_ROOT}" \
  "${H3_OVERLAY_ROOT}" "${RECOVERY_PAIRWISE_ROOT}" "${PAIRWISE_OVERLAY_ROOT}" "${FINAL_ROOT}"
cd "${REPO_DIR}"
if [[ ! -s "${JOB_LOG}" ]]; then
  printf 'timestamp_utc\tphase\tjob_id\tcommand\n' > "${JOB_LOG}"
fi
touch "${COMMAND_LOG}"

quote_command() {
  printf '%q ' "$@" >> "${COMMAND_LOG}"
  printf '\n' >> "${COMMAND_LOG}"
}

run() {
  quote_command "$@"
  "$@"
}

audit_args() {
  local output="$1"
  local h3_root="${2:-${ORIGINAL_H3_ROOT}}"
  local pairwise_root="${3:-${ORIGINAL_PAIRWISE_ROOT}}"
  local aggregate_root="${4:-${ORIGINAL_AGG_ROOT}}"
  local args=(
    "${PYTHON}" dinov3_res7_recovery.py audit
    --root "${ROOTFOLDER}"
    --embed-root "${EMBED_ROOT}"
    --h3-root "${h3_root}"
    --pairwise-root "${pairwise_root}"
    --aggregate-root "${aggregate_root}"
    --resolution "${RESOLUTION}"
    --output-json "${output}"
    --output-csv "${output%.json}.csv"
  )
  local city root
  for city in "${AFFECTED_CITIES[@]}"; do args+=(--city "${city}"); done
  IFS=':' read -r -a index_roots <<< "${INDEX_ROOTS}"
  for root in "${index_roots[@]}"; do args+=(--index-root "${root}"); done
  if [[ -n "${CITY_STEM_OVERRIDES:-}" ]]; then
    IFS=';' read -r -a stem_overrides <<< "${CITY_STEM_OVERRIDES}"
    local override
    for override in "${stem_overrides[@]}"; do args+=(--city-stem "${override}"); done
  fi
  [[ -n "${REQUIRED_H3_ROOT}" ]] && args+=(--required-h3-root "${REQUIRED_H3_ROOT}")
  [[ -n "${CORE_H3_ROOT}" ]] && args+=(--core-h3-root "${CORE_H3_ROOT}")
  run "${args[@]}"
}

record_job() {
  local phase="$1" job_id="$2"; shift 2
  local rendered
  printf -v rendered '%q ' "$@"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${phase}" "${job_id}" "${rendered}" >> "${JOB_LOG}"
}

wait_job() {
  local job_id="$1" phase="$2"
  while squeue -h -j "${job_id}" -o '%T' | grep -q .; do
    sleep "${POLL_SECONDS:-60}"
  done
  local accounting="${LOG_DIR}/${phase}_${job_id}_sacct.txt"
  sacct -n -X -j "${job_id}" --format=JobIDRaw,State,ExitCode -P > "${accounting}"
  if awk -F'|' 'NF >= 3 {seen=1; if ($2 !~ /^COMPLETED/ || $3 !~ /^0:0/) bad=1} END {exit (!seen || bad)}' "${accounting}"; then
    return 0
  fi
  printf 'Slurm phase %s failed; inspect %s\n' "${phase}" "${accounting}" >&2
  return 1
}

submit_array() {
  local phase="$1" count="$2" concurrency="$3" script="$4" export_vars="$5"
  local command=(sbatch --parsable --array="1-${count}%${concurrency}" --export="ALL,${export_vars}" "${script}")
  quote_command "${command[@]}"
  local job_id
  job_id="$("${command[@]}")"
  job_id="${job_id%%;*}"
  record_job "${phase}" "${job_id}" "${command[@]}"
  printf 'Submitted %s as %s\n' "${phase}" "${job_id}"
  wait_job "${job_id}" "${phase}"
}

audit_args "${PRE_AUDIT}"
recover_index_args=(
  "${PYTHON}" dinov3_res7_recovery.py recover-indices
  --audit-json "${PRE_AUDIT}" --root "${ROOTFOLDER}"
  --output-root "${RECOVERED_INDEX_ROOT}"
  --output-json "${AUDIT_DIR}/recovered_indices.json"
)
if [[ "${ALLOW_GSV_INDEX_REBUILD:-0}" == "1" ]]; then
  recover_index_args+=(--allow-gsv-rebuild)
fi
run "${recover_index_args[@]}"
INDEX_ROOTS="${RECOVERED_INDEX_ROOT}:${INDEX_ROOTS}"
audit_args "${SOURCE_AUDIT}"
run "${PYTHON}" dinov3_res7_recovery.py recovery-manifests \
  --audit-json "${SOURCE_AUDIT}" --embed-manifest "${EMBED_MANIFEST}" \
  --h3-manifest "${H3_MANIFEST}" --absent-cities "${ABSENT_CITIES}"

if ((EXECUTE == 0)); then
  printf 'Preflight complete. Audits: %s and %s\n' "${PRE_AUDIT}" "${SOURCE_AUDIT}"
  printf 'Review ambiguous aliases and set REQUIRED_H3_ROOT/CORE_H3_ROOT before --execute.\n'
  exit 0
fi
if [[ -z "${REQUIRED_H3_ROOT}" || -z "${CORE_H3_ROOT}" ]]; then
  printf 'REQUIRED_H3_ROOT and CORE_H3_ROOT are required in --execute mode.\n' >&2
  exit 2
fi

embed_count="$(wc -l < "${EMBED_MANIFEST}" | tr -d ' ')"
if ((embed_count > 0)); then
  submit_array embed "${embed_count}" "${EMBED_CONCURRENCY:-2}" \
    slurm/dinov3_res7_embed.cmd \
    "CITY_MANIFEST=${EMBED_MANIFEST},REPO_DIR=${REPO_DIR},ROOTFOLDER=${ROOTFOLDER},PYTHON=${PYTHON},VALFOLDER=${VALFOLDER},EMBED_ROOT=${EMBED_ROOT}"
fi

audit_args "${POST_EMBED_AUDIT}"
run "${PYTHON}" dinov3_res7_recovery.py recovery-manifests \
  --audit-json "${POST_EMBED_AUDIT}" --embed-manifest "${EMBED_MANIFEST}" \
  --h3-manifest "${H3_MANIFEST}" --absent-cities "${ABSENT_CITIES}"
h3_count="$(wc -l < "${H3_MANIFEST}" | tr -d ' ')"
if ((h3_count > 0)); then
  submit_array h3 "${h3_count}" "${H3_CONCURRENCY:-4}" \
    slurm/dinov3_res7_h3.cmd \
    "CITY_MANIFEST=${H3_MANIFEST},REPO_DIR=${REPO_DIR},ROOTFOLDER=${ROOTFOLDER},PYTHON=${PYTHON},EMBED_ROOT=${EMBED_ROOT},RECOVERY_H3_ROOT=${RECOVERY_H3_ROOT}"
fi

audit_args "${POST_H3_AUDIT}" "${RECOVERY_H3_ROOT}" "${RECOVERY_PAIRWISE_ROOT}" "${FINAL_ROOT}"
check_h3_args=(
  "${PYTHON}" dinov3_res7_recovery.py check-h3
  --audit-json "${POST_H3_AUDIT}" --output-json "${AUDIT_DIR}/h3_validation.json"
)
while IFS= read -r city; do
  [[ -n "${city}" ]] && check_h3_args+=(--allowed-missing-city "${city}")
done < "${ABSENT_CITIES}"
run "${check_h3_args[@]}"

overlay_args=(
  "${PYTHON}" dinov3_res7_recovery.py build-overlays
  --original-h3-root "${ORIGINAL_H3_ROOT}"
  --recovered-h3-root "${RECOVERY_H3_ROOT}"
  --h3-overlay-root "${H3_OVERLAY_ROOT}"
  --original-pairwise-root "${ORIGINAL_PAIRWISE_ROOT}"
  --recovered-pairwise-root "${RECOVERY_PAIRWISE_ROOT}"
  --pairwise-overlay-root "${PAIRWISE_OVERLAY_ROOT}"
  --resolution "${RESOLUTION}"
  --output-json "${AUDIT_DIR}/overlay_before_pairwise.json"
)
for city in "${AFFECTED_CITIES[@]}"; do overlay_args+=(--affected-city "${city}"); done
run "${overlay_args[@]}"

manifest_args=(
  "${PYTHON}" dinov3_res7_recovery.py manifest --city-meta "${CITY_META}"
  --h3-root "${H3_OVERLAY_ROOT}" --output "${AFFECTED_PAIRS}" --resolution "${RESOLUTION}"
)
for city in "${AFFECTED_CITIES[@]}"; do manifest_args+=(--affected-city "${city}"); done
run "${manifest_args[@]}"

: > "${PENDING_PAIRS}"
while IFS='|' read -r city1 city2; do
  [[ -z "${city1}" || -z "${city2}" ]] && continue
  shard="${RECOVERY_PAIRWISE_ROOT}/optimized/temp/city1=${city1}/city2=${city2}/part_res=${RESOLUTION}.parquet"
  [[ -s "${shard}" ]] || printf '%s|%s\n' "${city1}" "${city2}" >> "${PENDING_PAIRS}"
done < "${AFFECTED_PAIRS}"

pending_count="$(wc -l < "${PENDING_PAIRS}" | tr -d ' ')"
batch_size="${PAIR_BATCH_SIZE:-1000}"
for ((offset=0; offset<pending_count; offset+=batch_size)); do
  count=$((pending_count - offset)); ((count > batch_size)) && count="${batch_size}"
  submit_array "pairwise_${offset}" "${count}" "${PAIR_CONCURRENCY:-4}" \
    slurm/dinov3_03_pairwise_array.cmd \
    "PAIR_MANIFEST=${PENDING_PAIRS},PAIR_OFFSET=${offset},RESOLUTION=${RESOLUTION},SOURCE_ROOT=${H3_OVERLAY_ROOT},OUTPUT_ROOT=${RECOVERY_PAIRWISE_ROOT},UVI_SAMPLE_REPO_DIR=${REPO_DIR},VENV_PYTHON=${PYTHON},LOG_DIR=${LOG_DIR}/pairwise"
done

run "${PYTHON}" dinov3_res7_recovery.py check-pairs \
  --manifest "${AFFECTED_PAIRS}" --pairwise-root "${RECOVERY_PAIRWISE_ROOT}" \
  --resolution "${RESOLUTION}" --output-json "${AUDIT_DIR}/pairwise_validation.json"

run "${overlay_args[@]/overlay_before_pairwise/overlay_after_pairwise}"

b5c_command=(
  sbatch --parsable
  --export="ALL,REPO_DIR=${REPO_DIR},CITY_META=${CITY_META},PYTHON=${PYTHON},ROOTFOLDER=${ROOTFOLDER},RESOLUTION=${RESOLUTION},PAIRWISE_ROOT=${PAIRWISE_OVERLAY_ROOT},H3_MEMBERSHIP_ROOT=${H3_OVERLAY_ROOT},SIMILARITY_EXPORT_FOLDER=${FINAL_ROOT}"
  slurm/dinov3_04_b5c_aggregate.cmd
)
quote_command "${b5c_command[@]}"
b5c_job="$("${b5c_command[@]}")"; b5c_job="${b5c_job%%;*}"
record_job b5c "${b5c_job}" "${b5c_command[@]}"
wait_job "${b5c_job}" b5c

validate_args=(
  "${PYTHON}" dinov3_res7_recovery.py validate --export-root "${FINAL_ROOT}"
  --membership-root "${H3_OVERLAY_ROOT}" --resolution "${RESOLUTION}"
  --duckdb-temp-dir "${RUN_ROOT}/validator_duckdb"
  --duckdb-memory-limit "${VALIDATION_MEMORY_LIMIT:-32GB}"
  --output-json "${AUDIT_DIR}/final_validation.json"
)
while IFS= read -r city; do [[ -n "${city}" ]] && validate_args+=(--allowed-missing-city "${city}"); done < "${ABSENT_CITIES}"
while IFS= read -r city; do validate_args+=(--required-city "${city}"); done < <(
  "${PYTHON}" -c 'import pandas as pd,sys; print("\n".join(pd.read_csv(sys.argv[1])["City"].dropna().astype(str)))' "${CITY_META}"
)
run "${validate_args[@]}"
touch "${RUN_ROOT}/READY"
printf 'Recovery validated and ready: %s\n' "${RUN_ROOT}"
