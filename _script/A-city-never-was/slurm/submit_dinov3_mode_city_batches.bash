#!/usr/bin/env bash
set -euo pipefail
BATCH_SIZE="${BATCH_SIZE:-20}"; ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"; FIRST_CITY="${FIRST_CITY:-1}"; LAST_CITY="${LAST_CITY:?Set LAST_CITY}"; JOB_SCRIPT="${JOB_SCRIPT:?Set JOB_SCRIPT}"
for ((start=FIRST_CITY; start<=LAST_CITY; start+=BATCH_SIZE)); do end=$((start+BATCH_SIZE-1)); ((end>LAST_CITY)) && end=$LAST_CITY; job_id="$(sbatch --parsable --array="${start}-${end}%${ARRAY_CONCURRENCY}" "${JOB_SCRIPT}")"; while squeue -h -j "${job_id}" -o "%T" | grep -q .; do sleep "${POLL_SECONDS:-60}"; done; done
