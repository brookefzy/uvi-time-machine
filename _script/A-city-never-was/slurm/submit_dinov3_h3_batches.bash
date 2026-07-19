#!/usr/bin/env bash
# Submit the DINOv3 H3 array in bounded batches to respect MaxSubmitJobsPerAccount.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-${REPO_DIR%/*}/city_meta.csv}"
FIRST_CITY="${FIRST_CITY:-1}"
LAST_CITY="${LAST_CITY:-127}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if (( FIRST_CITY < 1 || LAST_CITY < FIRST_CITY || BATCH_SIZE < 1 || ARRAY_CONCURRENCY < 1 || POLL_SECONDS < 1 )); then
  printf 'Invalid settings: FIRST_CITY=%s LAST_CITY=%s BATCH_SIZE=%s ARRAY_CONCURRENCY=%s POLL_SECONDS=%s\n' \
    "${FIRST_CITY}" "${LAST_CITY}" "${BATCH_SIZE}" "${ARRAY_CONCURRENCY}" "${POLL_SECONDS}" >&2
  exit 2
fi

export REPO_DIR CITY_META
cd "${REPO_DIR}"
mkdir -p logs/slurm

for ((start=FIRST_CITY; start<=LAST_CITY; start+=BATCH_SIZE)); do
  end=$(( start + BATCH_SIZE - 1 ))
  if (( end > LAST_CITY )); then
    end="${LAST_CITY}"
  fi
  job_id="$(sbatch --parsable --array="${start}-${end}%${ARRAY_CONCURRENCY}" slurm/dinov3_02_h3_array.cmd)"
  printf 'Submitted H3 city indices %s-%s: job %s\n' "${start}" "${end}" "${job_id}"
  while squeue -h -j "${job_id}" -o "%T" | grep -q .; do
    printf 'Waiting %ss for H3 job %s before submitting the next batch...\n' "${POLL_SECONDS}" "${job_id}"
    sleep "${POLL_SECONDS}"
  done
done
