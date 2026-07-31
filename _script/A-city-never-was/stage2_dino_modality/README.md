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
