#!/usr/bin/env bash
set -euo pipefail
BATCH_SIZE="${BATCH_SIZE:-20}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
PAIR_COUNT="${PAIR_COUNT:?Set PAIR_COUNT}"
JOB_SCRIPT="${JOB_SCRIPT:?Set JOB_SCRIPT}"

for ((start=1; start<=PAIR_COUNT; start+=BATCH_SIZE)); do
  end=$((start+BATCH_SIZE-1))
  ((end>PAIR_COUNT)) && end=$PAIR_COUNT
  batch_count=$((end-start+1))
  pair_index_offset=$((start-1))
  job_id="$(
    sbatch --parsable \
      --export="ALL,PAIR_INDEX_OFFSET=${pair_index_offset}" \
      --array="1-${batch_count}%${ARRAY_CONCURRENCY}" \
      "${JOB_SCRIPT}"
  )"
  while squeue -h -j "${job_id}" -o "%T" | grep -q .; do
    sleep "${POLL_SECONDS:-60}"
  done
done
