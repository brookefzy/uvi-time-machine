# Direct DINOv3 pairwise runners

These scripts run B5b city-pair similarity calculations directly from a shell,
without submitting Slurm job arrays. They default to H3 resolution 6 and write
to a resolution-specific output root.

## `run_dinov3_pairwise_direct.bash`

Generates the eligible city-pair manifest and processes every pair
sequentially. It recomputes any existing temporary shard for a pair. Use it
only for a clean output root or when intentional recomputation is desired.

```bash
RESOLUTION=6 \
OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair_res=6 \
bash pipeline/run_dinov3_pairwise_direct.bash
```

## `run_dinov3_pairwise_resume.bash`

Generates the same manifest but skips a pair when its non-empty atomic B5b
shard already exists at:

```text
optimized/temp/city1=<CITY1>/city2=<CITY2>/part_res=<RESOLUTION>.parquet
```

Use this after interrupted array or direct work to process only missing pairs.
A leftover `.tmp` shard is treated as incomplete and is recomputed.

```bash
RESOLUTION=6 \
OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair_res=6 \
bash pipeline/run_dinov3_pairwise_resume.bash
```

Both scripts accept the same useful environment overrides: `VENV_PYTHON`,
`CITY_META`, `SOURCE_ROOT`, `INPUT_TEMPLATE`, `B5B_MEMORY_LIMIT`,
`ROW_BLOCK_SIZE`, `DINO_THRESHOLD`, `LOG_DIR`, and `PAIR_MANIFEST`.

Run them on an interactive compute allocation or another suitable compute host,
not a shared login node. They run one pair at a time and use a default DuckDB
memory limit of 96GB.

## `slurm/submit_dinov3_pairwise_resume_batches.bash`

Use this submitter after cancelled or interrupted arrays. It regenerates the
eligible-pair manifest and submits only pairs whose final atomic shard is
missing or empty; a non-empty shard is considered complete. It waits for each
bounded array to drain before submitting the next one.

```bash
RESOLUTION=8 \
OUTPUT_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair_res=8 \
BATCH_SIZE=20 ARRAY_CONCURRENCY=2 POLL_SECONDS=120 \
bash slurm/submit_dinov3_pairwise_resume_batches.bash
```
