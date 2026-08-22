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

## `run_dinov3_res7_recovery.bash`

Use this forensic recovery runner for the seven downstream resolution-7 gaps:
Amsterdam, Gombe, Kampala, Kozhikode, Malegaon, Sitapur, and Vijayawada. The
default mode is a read-only preflight (apart from its timestamped audit folder):

```bash
REQUIRED_H3_ROOT=/path/to/current/stage3/res7/all_cells \
CORE_H3_ROOT=/path/to/current/stage3/res7/core_cells \
bash pipeline/run_dinov3_res7_recovery.bash
```

Review `audit/before.csv`, `audit/after_index_recovery.csv`, and the JSON detail.
If an alias is ambiguous, provide an explicit semicolon-delimited override, for
example `CITY_STEM_OVERRIDES='Kozhikode=calicut;Vijayawada=vijayawada_ap'`.
Multiple candidate validation-index roots are colon-delimited in `INDEX_ROOTS`.
If the audit finds raw GSV image files but no validation index, it stops before
using them. Set `ALLOW_GSV_INDEX_REBUILD=1` only after confirming those files
are the intended DINO validation-image corpus; the recovered index stays under
the run root.

After the preflight resolves every alias, execute with a stable run root so the
same command can safely resume:

```bash
export REQUIRED_H3_ROOT=/path/to/current/stage3/res7/all_cells
export CORE_H3_ROOT=/path/to/current/stage3/res7/core_cells
export RUN_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_tmp/dinov3_res7_recovery/manual_20260821

bash pipeline/run_dinov3_res7_recovery.bash \
  --run-root "${RUN_ROOT}" \
  --execute
```

The runner resumes embeddings by image name and never overwrites an existing
embedding shard. Rebuilt H3 summaries, affected pairwise shards, B5c outputs,
manifests, job IDs, commands, `sacct` records, and validation reports remain
under `RUN_ROOT`. The H3 and pairwise inputs to B5c are symlink overlays:
unaffected artifacts link to the originals, affected artifacts link only to the
recovery root. Old affected shards are deliberately excluded.

The run stops on ambiguous aliases, failed/cancelled Slurm tasks, a missing
source-backed Stage-3 core cell, an absent affected pair shard, or any final
schema/membership/null/zero-sentinel/range/duplicate/coverage violation. Only a
fully validated run gets `RUN_ROOT/READY`. Cities proven to have no index,
candidate GSV stem, physical image file, or existing embedding are listed in
`manifests/source_imagery_absent.txt`; they remain absent from output rather than
being filled with zero similarities.

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
