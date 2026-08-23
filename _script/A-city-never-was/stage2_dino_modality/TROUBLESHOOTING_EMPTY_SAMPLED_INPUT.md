# Troubleshooting an empty sampled-image input

Use this runbook when `run_dinov3_mode_pipeline.bash` reaches the mode-fit job
and fails with:

```text
ValueError: No objects to concatenate
```

The fitting error is a downstream symptom. It means that
`$MODE_OUTPUT_ROOT/sampled_images` contained no Parquet files discoverable by
the fitting process. Common causes include an incorrect output root, a broken
sample-directory symlink, failed sampling arrays, or a source directory that
does not contain completed samples.

The current city-array submitter waits for jobs to leave `squeue`, but does not
verify their final exit status. A failed sampling batch can therefore be
followed by an attempted fit against an empty directory.

## 1. Establish the remote environment

Run these commands on the Slurm server:

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was

export ROOTFOLDER=/lustre1/g/geog_pyloo/05_timemachine
export VENV_PYTHON=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python
export CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv
```

Set `MODE_OUTPUT_ROOT` to the actual experiment directory. Do not use the
literal placeholder `/path/to/the/lower-k/output-root`.

```bash
export MODE_OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_global_modes/res=8/sample=50-stratified-5seed-v1
```

## 2. Inspect the configured paths

```bash
printf 'MODE_OUTPUT_ROOT=%s\n' "$MODE_OUTPUT_ROOT"
ls -ld "$MODE_OUTPUT_ROOT" "$MODE_OUTPUT_ROOT/sampled_images"
readlink -f "$MODE_OUTPUT_ROOT/sampled_images" || true
```

Count the sampled Parquet files, following a directory symlink if present:

```bash
printf 'Sample Parquet count: '
find -L "$MODE_OUTPUT_ROOT/sampled_images" \
  -type f -name 'city=*.parquet' 2>/dev/null |
  wc -l
```

Compare that count with the city metadata:

```bash
printf 'Expected city count: '
"$VENV_PYTHON" -c '
import csv, os
print(len(list(csv.DictReader(open(os.environ["CITY_META"], newline="")))))
'
```

A lower sample count can be intentional only when the experiment explicitly
accepts unavailable cities. A count of zero must be fixed before fitting.

## 3. Check sampling job outcomes

Show recent sampling jobs and their final states:

```bash
sacct -X -u "$USER" --starttime today \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,ReqMem,MaxRSS |
  rg 'dinov3_mode_sample|JobID'
```

Replace `today` with an earlier date when the failed run is older.

Search sampling error logs:

```bash
rg -n -i \
  'error|exception|traceback|killed|oom|no such file|permission denied' \
  logs/slurm \
  --glob 'dinov3_mode_sample_*.err' |
  tail -100
```

The first sampling traceback normally identifies the actual cause. Resolve
that error before resubmitting the fitter.

## 4. Reuse a completed sampled-image directory

If the samples already exist under an earlier output root, link them into a
fresh codebook-evaluation root:

```bash
export SOURCE_MODE_OUTPUT_ROOT="$ROOTFOLDER/_curated/c_city_dinov3_global_modes/res=8/sample=50"
export MODE_OUTPUT_ROOT="${SOURCE_MODE_OUTPUT_ROOT}-stratified-5seed-v1"

ls -ld "$SOURCE_MODE_OUTPUT_ROOT/sampled_images"
find "$SOURCE_MODE_OUTPUT_ROOT/sampled_images" \
  -type f -name 'city=*.parquet' |
  wc -l

mkdir -p "$MODE_OUTPUT_ROOT"

if [[ ! -e "$MODE_OUTPUT_ROOT/sampled_images" ]]; then
  ln -s "$SOURCE_MODE_OUTPUT_ROOT/sampled_images" \
    "$MODE_OUTPUT_ROOT/sampled_images"
fi
```

If `sampled_images` is already a broken or incorrect symlink, inspect it before
replacing it. Preserve or move incorrect paths rather than deleting data
recursively.

## 5. Apply the mandatory pre-fit gate

Do not submit or resume the fit until this command reports a nonzero file
count:

```bash
"$VENV_PYTHON" -c '
import os
from pathlib import Path

root = Path(os.environ["MODE_OUTPUT_ROOT"]) / "sampled_images"
files = sorted(root.rglob("*.parquet"))
print("Input:", root)
print("Resolved:", root.resolve())
print("Parquet files:", len(files))
if not files:
    raise SystemExit("STOP: no sampled Parquet files")
print("First file:", files[0])
'
```

Also verify that the files are readable:

```bash
"$VENV_PYTHON" -c '
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["MODE_OUTPUT_ROOT"]) / "sampled_images"
file = next(root.rglob("*.parquet"))
frame = pd.read_parquet(file)
print(file)
print("Rows:", len(frame))
print("Columns:", len(frame.columns))
'
```

## 6. Resume the pipeline

After the input gate succeeds:

```bash
RESUME=1 bash slurm/run_dinov3_mode_pipeline.bash \
  2>&1 | tee logs/dinov3_mode_fit_resume.log
```

If this is a candidate-fitting pass, keep `SELECTED_K` unset. If a valid
scorecard already exists and the run should continue downstream, export the
reviewed `SELECTED_K` before resuming.

## 7. If sampling must be regenerated

Use valid zero-based city-array bounds:

```bash
export FIRST_CITY=0
export LAST_CITY=$(( $(wc -l < "$CITY_META") - 2 ))
```

Then rerun the coordinator with the intended `MODE_OUTPUT_ROOT`. Monitor the
sampling arrays rather than assuming that disappearance from `squeue` means
success:

```bash
squeue -u "$USER" \
  -o "%.18i %.28j %.10T %.10M %.6D %R" |
  rg 'dinov3_mode_sample|JOBID'

sacct -X -u "$USER" --starttime today \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,ReqMem,MaxRSS |
  rg 'dinov3_mode_sample|JobID'
```

Do not proceed to fitting unless the pre-fit gate in section 5 succeeds.
