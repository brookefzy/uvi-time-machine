#!/usr/bin/env bash
set -euo pipefail
BATCH_SIZE="${BATCH_SIZE:-20}"; ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
while squeue -h -j "${job_id:-0}" -o "%T" | grep -q .; do sleep 60; done
