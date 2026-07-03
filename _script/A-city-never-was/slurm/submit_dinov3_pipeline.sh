#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"
CITY_META="${CITY_META:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}"

export REPO_DIR CITY_META MODEL_NAME

cd "${REPO_DIR}"
mkdir -p logs/slurm

smoke_job="$(sbatch --parsable slurm/dinov3_00_smoke.cmd)"
embed_job="$(sbatch --parsable --dependency=afterok:${smoke_job} slurm/dinov3_01_embed_array.cmd)"
h3_job="$(sbatch --parsable --dependency=afterok:${embed_job} slurm/dinov3_02_h3_array.cmd)"
pairwise_job="$(sbatch --parsable --dependency=afterok:${h3_job} slurm/dinov3_03_pairwise.cmd)"
b5c_job="$(sbatch --parsable --dependency=afterok:${pairwise_job} slurm/dinov3_04_b5c_aggregate.cmd)"
summary_job="$(sbatch --parsable --dependency=afterok:${b5c_job} slurm/dinov3_05_summary.cmd)"

printf 'Submitted DINOv3 pipeline jobs:\n'
printf '  smoke:   %s\n' "${smoke_job}"
printf '  embed:   %s\n' "${embed_job}"
printf '  h3:      %s\n' "${h3_job}"
printf '  pairwise:%s\n' "${pairwise_job}"
printf '  b5c:     %s\n' "${b5c_job}"
printf '  summary: %s\n' "${summary_job}"
