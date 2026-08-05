# Global DINOv3 modes

This stage builds a globally shared DINOv3 codebook, assigns balanced H3
samples to its visual modes, and summarizes cross-city H3
Jensen--Shannon similarity. Run it on the remote Slurm cluster in two passes:
first create and review codebook candidates, then select `K` and submit the
expensive downstream stages.

## Remote setup

The Slurm jobs use the Lustre paths below by default, so no exports are needed
on the remote server. From the repository's `A-city-never-was` directory:

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was
git pull

bash slurm/run_dinov3_mode_pipeline.bash
```

The defaults are `ROOTFOLDER=/lustre1/g/geog_pyloo/05_timemachine`,
`CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv`,
and the repository `.venv` under that checkout. `CITY_META` must have a `City`
column. `IMAGE_INDEX_ROOT` is the
existing image-index directory: it contains `<resolved-city-stem>.parquet`
shards (for example, `hongkong.parquet`) with `path` and, optionally, `name`.
The new `city=<city>.parquet` convention is also accepted.

## E2E smoke run

Before submitting every city, use a separate two- or three-city `CITY_META`
file. It exercises data access, Slurm resources, gallery assets, and the full
pairwise flow without creating the all-city cross product.

### Pass 1: sample, fit, and review candidates

```bash
unset SELECTED_K
RESUME=0 bash slurm/run_dinov3_mode_pipeline.bash
```

Inspect candidate metrics:

```bash
VENV_PYTHON=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python
"$VENV_PYTHON" -c '
import os, pandas as pd
root = os.environ["MODE_OUTPUT_ROOT"]
print(pd.read_parquet(f"{root}/scorecard.parquet").to_string(index=False))
'
```

Review the generated HTML galleries:

```text
$MODE_OUTPUT_ROOT/mode_gallery/k=64/index.html
$MODE_OUTPUT_ROOT/mode_gallery/k=128/index.html
$MODE_OUTPUT_ROOT/mode_gallery/k=256/index.html
$MODE_OUTPUT_ROOT/mode_gallery/k=512/index.html
```

### Pass 2: select K and run assignments through summary

Choose a reviewed candidate. For a smoke run, use a restrictive threshold;
`-1` retains every H3-by-H3 result and can create a very large output.

```bash
export SELECTED_K=128
export SIMILARITY_THRESHOLD=0.80

RESUME=1 bash slurm/run_dinov3_mode_pipeline.bash
```

Inspect the final summary:

```bash
MODEL_ID="$("$VENV_PYTHON" -c '
import json, os
print(json.load(open(os.path.join(os.environ["MODE_OUTPUT_ROOT"], "selected_model.json")))["model_id"])
')"

MODEL_ID="$MODEL_ID" "$VENV_PYTHON" -c '
import os, pandas as pd
root = os.environ["MODE_OUTPUT_ROOT"]
model = os.environ["MODEL_ID"]
print(pd.read_parquet(f"{root}/model={model}/city_pair_summary.parquet").to_string(index=False))
'
```

## Full run, resumption, and clean reruns

After the smoke run succeeds, set `CITY_META` to the complete city list, retain
the selected `K`, choose an intentional similarity threshold/block size, and
run the second pass again with `RESUME=1`. The coordinator reuses complete
artifacts and refuses to reuse a `selected_model.json` whose model ID conflicts
with the requested `K`.

Useful optional settings:

```bash
export SIMILARITY_THRESHOLD=-1       # retain all pairs; assess storage first
export SIMILARITY_ROW_BLOCK_SIZE=64
export SIMILARITY_TARGET_BLOCK_SIZE=2048
```

### Check a cancelled run before resuming

Run these commands on the Slurm cluster. `squeue` shows surviving jobs;
`sacct` records the cancelled and failed tasks when Slurm accounting is
available. Search the per-task logs before resubmitting a job that failed for a
reason other than cancellation.

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was

squeue -u "$USER"
sacct -X -u "$USER" --starttime today \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,Start,End
rg -n -i 'error|exception|cancel|killed|traceback' logs/slurm/dinov3_mode_*.err
```

The authoritative progress markers are the artifacts under
`$MODE_OUTPUT_ROOT`:

- `sampled_images/city=<city>.parquet`
- `model=<model_id>/assignments/city=<city>.parquet`
- `model=<model_id>/h3_histograms/city=<city>.parquet`
- `model=<model_id>/h3_similarity/city_1=<city_1>/city_2=<city_2>/part_res=8.parquet`
- `model=<model_id>/city_pair_summary.parquet`

Sample and pair-similarity outputs are atomically published, so an existing
file from those stages is complete. Assignment, histogram, fit, and summary
Parquet files are written directly; validate them with `pandas.read_parquet`
after a cancellation rather than relying only on file existence.

### Resume an interrupted run

Use exactly the same output root, selected K, threshold, and block sizes as the
interrupted run. A fresh invocation of the coordinator is preferable to
requeueing its cancelled parent job:

```bash
export MODE_OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_global_modes/res=8/sample=50
export CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv
export SELECTED_K=128                 # must agree with selected_model.json
export SIMILARITY_THRESHOLD=0.80      # must agree with existing pair shards

RESUME=1 bash slurm/run_dinov3_mode_pipeline.bash
```

Resume is safe but coarse-grained: if any output in a city-array or pair-array
stage is absent, the coordinator resubmits that entire stage's arrays. Existing
sample and pair shards are replaced atomically; existing assignment and
histogram files are overwritten by their re-run tasks. If every expected file
exists but one is unreadable, rerun the affected array task explicitly (or
archive the root and do a clean rerun), because the coordinator's completeness
check is existence-based.

### Complete rerun without replacing prior results (recommended)

Use a new, timestamped `MODE_OUTPUT_ROOT`. This recomputes samples, codebooks,
selection, assignments, histograms, pair shards, and the summary while keeping
the cancelled run available for comparison and recovery:

```bash
export RUN_TAG="$(date +%Y%m%d-%H%M%S)"
export MODE_OUTPUT_ROOT="/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_global_modes/res=8/sample=50-rerun-${RUN_TAG}"
export CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv
export SELECTED_K=128
export SIMILARITY_THRESHOLD=0.80
export SIMILARITY_ROW_BLOCK_SIZE=64
export SIMILARITY_TARGET_BLOCK_SIZE=2048

RESUME=0 bash slurm/run_dinov3_mode_pipeline.bash
```

For a new experiment, omit `SELECTED_K` and run the first pass with `RESUME=0`.
Review its scorecard and galleries, then set the chosen K and use `RESUME=1`
for the downstream stages.

### Complete rerun in the standard output location

Do this only when replacing the standard root is intentional. First archive it
with `mv` (recoverable on the same Lustre filesystem), then run with
`RESUME=0`. Do not use `rm -rf` for this workflow.

```bash
export MODE_OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_global_modes/res=8/sample=50
export RUN_TAG="$(date +%Y%m%d-%H%M%S)"
mv "$MODE_OUTPUT_ROOT" "${MODE_OUTPUT_ROOT}.before-rerun-${RUN_TAG}"

export SELECTED_K=128
export SIMILARITY_THRESHOLD=0.80
RESUME=0 bash slurm/run_dinov3_mode_pipeline.bash
```

This last command assumes the same `CITY_META`, selection, and similarity
settings as the prior production run. Changing any of them makes this a new
experiment and should normally use a new output root instead.
