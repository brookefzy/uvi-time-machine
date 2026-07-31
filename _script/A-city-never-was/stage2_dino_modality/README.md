# Global DINOv3 modes

This stage builds a globally shared DINOv3 codebook, assigns balanced H3
samples to its visual modes, and summarizes cross-city H3
Jensen--Shannon similarity. Run it on the remote Slurm cluster in two passes:
first create and review codebook candidates, then select `K` and submit the
expensive downstream stages.

## Remote setup

From the repository's `A-city-never-was` directory:

```bash
cd /path/to/uvi-time-machine/_script/A-city-never-was
git pull

export UVI_SAMPLE_REPO_DIR="$PWD"
export VENV_PYTHON="/path/to/uvi-time-machine/.venv/bin/python"

export ROOTFOLDER="/path/to/data-root"
export EMBEDDING_ROOT="/path/to/dinov3-embeddings"
export TRAIN_TEST_FOLDER="/path/to/train-test-metadata"
export CITY_META="/path/to/city_meta.csv"
export IMAGE_INDEX_ROOT="/path/to/image-index-root"

export MODE_OUTPUT_ROOT="${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50"

# Tune these for the remote account and Slurm cluster.
export BATCH_SIZE=20
export ARRAY_CONCURRENCY=2
export POLL_SECONDS=60
export SBATCH_ACCOUNT="your-account"
export SBATCH_PARTITION="your-partition"
export SBATCH_TIME="24:00:00"
export SBATCH_CPUS_PER_TASK=4
export SBATCH_MEM="32G"

"$VENV_PYTHON" -c 'import faiss, h3, numpy, pandas, pyarrow, sklearn; print("environment OK")'
```

`CITY_META` must be a CSV with a `City` column. `IMAGE_INDEX_ROOT` must contain
`city=<city>.parquet` shards with `path` and, optionally, `name`; the city name
may be supplied by the shard filename instead of a column.

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

## Full run and resumption

After the smoke run succeeds, set `CITY_META` to the complete city list, retain
the selected `K`, choose an intentional similarity threshold/block size, and
run the second pass again with `RESUME=1`. The coordinator reuses complete
artifacts and submits only missing stages. It refuses to reuse a
`selected_model.json` whose model ID conflicts with the requested `K`.

Useful optional settings:

```bash
export SIMILARITY_THRESHOLD=-1       # retain all pairs; assess storage first
export SIMILARITY_ROW_BLOCK_SIZE=64
export SIMILARITY_TARGET_BLOCK_SIZE=2048
```
