#!/usr/bin/env bash
# Re-submit only city indices marked missing or invalid by the H3 coverage audit.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
COVERAGE_CSV="${COVERAGE_CSV:-logs/dinov3_h3_coverage.csv}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if (( BATCH_SIZE < 1 || ARRAY_CONCURRENCY < 1 || POLL_SECONDS < 1 )); then
  printf 'Invalid settings: BATCH_SIZE=%s ARRAY_CONCURRENCY=%s POLL_SECONDS=%s\n' \
    "${BATCH_SIZE}" "${ARRAY_CONCURRENCY}" "${POLL_SECONDS}" >&2
  exit 2
fi

export REPO_DIR CITY_META
cd "${REPO_DIR}"

if [[ ! -f "${COVERAGE_CSV}" ]]; then
  printf 'Coverage CSV not found: %s\n' "${COVERAGE_CSV}" >&2
  exit 2
fi
if [[ ! -f "${CITY_META}" ]]; then
  printf 'City metadata CSV not found: %s\n' "${CITY_META}" >&2
  exit 2
fi

city_indices_text="$(python3 - "${COVERAGE_CSV}" "${CITY_META}" <<'PY'
import csv
import sys

coverage_path, city_meta_path = sys.argv[1:]
statuses = {"missing", "error"}
with open(coverage_path, newline="", encoding="utf-8") as handle:
    coverage_rows = list(csv.DictReader(handle))
if not coverage_rows or not {"city", "status"}.issubset(coverage_rows[0]):
    raise SystemExit("Coverage CSV must contain city and status columns")

retry_cities = {
    row["city"].strip()
    for row in coverage_rows
    if row.get("status", "").strip().lower() in statuses and row.get("city", "").strip()
}
ignored_cities = {
    row["city"].strip()
    for row in coverage_rows
    if row.get("status", "").strip().lower() == "ignored_no_images" and row.get("city", "").strip()
}
if ignored_cities:
    print(f"Skipping {len(ignored_cities)} cities marked ignored_no_images", file=sys.stderr)
with open(city_meta_path, newline="", encoding="utf-8") as handle:
    city_rows = list(csv.DictReader(handle))
if not city_rows:
    raise SystemExit("City metadata is empty")
city_column = next((name for name in ("City", "city", "city_name", "name") if name in city_rows[0]), None)
if city_column is None:
    raise SystemExit("City metadata must contain City, city, city_name, or name")

seen = set()
city_to_index = {}
for row in city_rows:
    city = (row.get(city_column) or "").strip()
    if city and city not in seen:
        seen.add(city)
        city_to_index[city] = len(seen)

unknown = sorted(retry_cities.difference(city_to_index))
if unknown:
    raise SystemExit(f"Coverage cities absent from city metadata: {unknown}")
for city, index in city_to_index.items():
    if city in retry_cities:
        print(index)
PY
)"

if [[ -z "${city_indices_text}" ]]; then
  printf 'No cities with status=missing or status=error in %s. Nothing to submit.\n' "${COVERAGE_CSV}"
  exit 0
fi

mapfile -t city_indices <<< "${city_indices_text}"
printf 'Re-running H3 aggregation for %s cities from %s.\n' "${#city_indices[@]}" "${COVERAGE_CSV}"
mkdir -p logs/slurm

for ((offset=0; offset<${#city_indices[@]}; offset+=BATCH_SIZE)); do
  batch=("${city_indices[@]:offset:BATCH_SIZE}")
  array_spec="$(IFS=,; printf '%s' "${batch[*]}")"
  job_id="$(sbatch --parsable --array="${array_spec}%${ARRAY_CONCURRENCY}" slurm/dinov3_02_h3_array.cmd)"
  printf 'Submitted H3 retry city indices %s: job %s\n' "${array_spec}" "${job_id}"
  while squeue -h -j "${job_id}" -o "%T" | grep -q .; do
    printf 'Waiting %ss for H3 retry job %s before submitting the next batch...\n' "${POLL_SECONDS}" "${job_id}"
    sleep "${POLL_SECONDS}"
  done
done
