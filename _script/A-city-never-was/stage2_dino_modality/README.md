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

Codebook evaluation uses a deterministic hash-stratified holdout within every
city. By default, five models are fitted with seeds 42 through 46 and
`stability` is the median adjusted Rand score across all ten seed pairs. The
scorecard also records `stability_mean`, `stability_min`, `stability_max`,
`stability_std`, seed/pair counts, holdout strategy, and train/holdout city
counts. The versioned stability strategy is `all_pairs_ari_median_v1`. The
primary seed (42 by default) supplies the saved centroids; all
other fits are evaluation models.

To rerun only codebook evaluation against completed samples, use a fresh output
root and link the immutable sample shards into it:

```bash
export SOURCE_MODE_OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_global_modes/res=8/sample=50
export MODE_OUTPUT_ROOT="${SOURCE_MODE_OUTPUT_ROOT}-stratified-5seed-v1"
mkdir -p "$MODE_OUTPUT_ROOT"
ln -s "$SOURCE_MODE_OUTPUT_ROOT/sampled_images" "$MODE_OUTPUT_ROOT/sampled_images"

JOB_ID="$(sbatch --parsable slurm/dinov3_mode_fit_codebooks.cmd \
  --input "$MODE_OUTPUT_ROOT/sampled_images" \
  --output-root "$MODE_OUTPUT_ROOT" \
  --k 4 8 16 32 64 \
  --holdout-fraction 0.20 \
  --holdout-split-seed 42 \
  --seed 42 \
  --stability-seed-count 5 \
  --niter 100)"
echo "Submitted job: $JOB_ID"
```

Five-seed evaluation performs five full FAISS fits per K and therefore takes
roughly 2.5 times as much fitting work as the former two-seed evaluation.
Monitor the submitted job and inspect its peak memory after completion:

```bash
squeue -j "$JOB_ID" -o "%.18i %.28j %.10T %.10M %.6D %R"
tail -f "logs/slurm/dinov3_mode_fit_${JOB_ID}.out"

sacct -j "$JOB_ID" --units=G \
  --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS,MaxVMSize
```

Inspect the expanded scorecard after the job succeeds:

```bash
export VENV_PYTHON=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python
"$VENV_PYTHON" -c '
import os, pandas as pd
x = pd.read_parquet(os.environ["MODE_OUTPUT_ROOT"] + "/scorecard.parquet")
columns = [
    "k", "status", "held_out_mean_cohesion", "held_out_p05_cohesion",
    "min_mode_share", "near_empty_mode_count", "stability",
    "stability_mean", "stability_min", "stability_max", "stability_std",
    "stability_seed_count", "stability_pair_count", "training_city_count",
    "holdout_city_count", "model_id",
]
print(x[[column for column in columns if column in x]].to_string(index=False))
'
```

For the default five-seed run, each valid scorecard row should report
`stability_seed_count=5`, `stability_pair_count=10`,
`holdout_strategy=city_stratified_hash_v1`, and
`stability_strategy=all_pairs_ari_median_v1`.

### Pass 2: select K and run assignments through summary

Choose a reviewed candidate. For a smoke run, use a restrictive threshold;
`-1` retains every H3-by-H3 result and can create a very large output.

```bash
export SELECTED_K=128
export SIMILARITY_THRESHOLD=0.80
export CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv

# City array jobs index the CSV data rows from zero.
export FIRST_CITY=0
export LAST_CITY=$(( $(wc -l < "$CITY_META") - 2 ))

RESUME=1 bash slurm/run_dinov3_mode_pipeline.bash
```

### Continue while auditing cities with missing histograms

Strict mode is the default: a city listed in `CITY_META` without a histogram
stops pair-manifest generation. If missing histograms are known and the
remaining cities should continue through similarity and summary, opt in
explicitly:

```bash
export ALLOW_MISSING_CITIES=1
unset PAIR_MANIFEST
unset SELECTED_MODEL

RESUME=1 bash slurm/run_dinov3_mode_pipeline.bash
```

Only absent `city=<city>.parquet` histogram files are skipped. An existing
histogram that is empty, has the wrong resolution, or belongs to a different
model remains a fatal validation error. When at least one histogram exists,
allow-missing mode accepts that existing subset without resubmitting and
overwriting the complete histogram array. The model directory records the
exact city population used:

```text
model=<model_id>/available_cities.txt
model=<model_id>/skipped_cities.txt
model=<model_id>/pair_manifest.txt
```

Inspect the audit before using the final statistics:

```bash
MODEL_ID="$("$VENV_PYTHON" -c '
import json, os
print(json.load(open(os.path.join(
    os.environ["MODE_OUTPUT_ROOT"], "selected_model.json"
)))["model_id"])
')"

cat "$MODE_OUTPUT_ROOT/model=$MODEL_ID/skipped_cities.txt"
wc -l "$MODE_OUTPUT_ROOT/model=$MODEL_ID/available_cities.txt"
wc -l "$MODE_OUTPUT_ROOT/model=$MODEL_ID/pair_manifest.txt"
```

The pair manifest becomes authoritative for similarity and summary. Stale
similarity partitions for excluded cities are ignored, and a city pair with
no rows above `SIMILARITY_THRESHOLD` remains in the summary with
`pair_count_observed=0` and null similarity statistics. The summary is rebuilt
after every manifest regeneration so it cannot retain rows for newly excluded
cities.

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
