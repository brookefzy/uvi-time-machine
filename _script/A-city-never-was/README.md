# Development environment
```
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/
source .venv/bin/activate
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was
uv pip install -r requirements.txt

```
# Usage
* TO-DO: add previous steps
1.  summarizes the all probablity vector at resolution 6,7,8 for each city
```python B5a_prob_vector_summary.py --city all```
2. summarizes the pairwire similarity index, saved by each city.
```python B5b_compute_similarity_pairwise.py --res 6``` 
   Optimized resumable version:
```bash
python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --resolution 6 \
  --row-block-size 1000
```

   Resume behavior:
   - The optimized script preserves the original global city-pair behavior. It still evaluates every unique `(city1, city2)` pair, but computes each pair in smaller row blocks to reduce memory use.
   - If `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair/optimized/_progress_res=6_optimized.json` exists, the script resumes from its pending city-pair list.
   - If that JSON file does not exist, the script scans existing `similarity_city=*_res=<resolution>_optimized.parquet` files in the `optimized/` output folder and skips already completed `city1` outputs.
   - Use `--fresh-start` to ignore previous progress and recompute from scratch.

   Disk usage:
   - Intermediate exact results are written under `optimized/temp/city1=<city>/city2=<city>/`.
   - Those temp shards are merged into final `similarity_city=*_res=<resolution>_optimized.parquet` outputs after processing finishes.

   Local debug logs:
   - Default log folder: `/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/logs`
   - Optional override: add `--log-dir /path/to/logs`

3. summarize similarity index for each city.
```python B5c_pairwise_agg.py``` 
   Pairing:
   - Use `B5c_pairwise_agg.py` with `B5b_compute_similarity_pairwise.py`
   - Use `B5c_pairwise_agg_optimized.py` with `B5b_compute_similarity_pairwise-optimized.py`

   Optimized aggregation example:
```bash
python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --resolution 8 \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_optimized_res=8 \
  --progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair/optimized/_progress_res=8_optimized.json \
  --resume \
  --agg-progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_optimized_progress_res=8.json \
  --duckdb-memory-limit 32GB \
  --duckdb-temp-dir /lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_city_similarity \
  --parquet-file-size 512MB
```

   Notes:
   - `B5c_pairwise_agg_optimized.py` reads temp shards from `optimized/temp/city1=<city>/city2=*/part_res=<resolution>.parquet`
   - If those temp shards have already been cleaned up after a completed `B5b_compute_similarity_pairwise-optimized.py` run, it falls back to merged files at `optimized/similarity_city=<city>_res=<resolution>_optimized.parquet`
   - It does not require merged per-city optimized parquet files to exist first, but can use them when temp shards are no longer present
   - `--resume` skips cities that already have aggregated output files, and if `--agg-progress-file` is provided it also resumes from that city-level progress JSON
   - Resume recognizes both a single output parquet file and a chunked parquet dataset directory at `similarity_intracity_city=<city>_res=<resolution>.parquet`
   - Use an explicit `--export-folder` when resuming. The script's default export folder is date-based, so a bare `python B5c_pairwise_agg_optimized.py --resume` on a later day will look in a new folder and will not see yesterday's completed city outputs.
   - The optimized aggregator keeps exact results by default and exports directly from DuckDB instead of materializing the full city result in pandas
   - By default the optimized aggregator writes a parquet dataset directory per city, split into smaller `part_*.parquet` files to avoid getting stuck on a single massive final write
   - Set `--parquet-file-size 0` if you explicitly want one single parquet file instead of chunked output
   - If the progress JSON has no pending pairs but still says `in_progress`, the optimized aggregator proceeds and logs a warning

## DINOv3 visual similarity pipeline

The DINOv3 pipeline runs beside the classifier-probability B5 pipeline and writes to separate `c_city_dinov3_*` folders. Do not overwrite classifier outputs when testing DINOv3.

Pipeline orchestrator:

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was

MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"
CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv

python dinov3_pipeline.py \
  --stage smoke \
  --city "Saidpur" \
  --city-meta "${CITY_META}" \
  --model-name "${MODEL_NAME}" \
  --execute
```

Use `--stage embed`, `--stage aggregate`, `--stage pairwise`, `--stage b5c`, `--stage summary`, or `--stage all` to run other stages. Without `--execute`, the orchestrator prints the exact commands instead of running them.

SLURM production run for 127 cities and 30M+ images:

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was

export MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"
export CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv
export REPO_DIR=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was

bash slurm/submit_dinov3_pipeline.sh
```

The SLURM scripts use city-level arrays for the two per-city stages:

- `slurm/dinov3_01_embed_array.sbatch`: `#SBATCH --array=1-127%8`, one GPU job per city, capped at 8 concurrent cities.
- `slurm/dinov3_02_h3_array.sbatch`: `#SBATCH --array=1-127%24`, one CPU job per city, capped at 24 concurrent cities.
- `slurm/dinov3_03_pairwise.sbatch`, `slurm/dinov3_04_b5c_aggregate.sbatch`, and `slurm/dinov3_05_summary.sbatch` run after the city arrays complete.

Tune `--partition`, `--gres`, `--mem`, `--time`, and the `%` array concurrency caps in the `.sbatch` files for the actual cluster limits. The embedding stage is resumable because each city writes chunked parquet shards and skips already written image names.

Recommended server order:
0. Use the verified Hugging Face model card for the default DINOv3 backbone:
   - Model ID: `facebook/dinov3-vitb16-pretrain-lvd1689m`
   - Model card: `https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`
   - Direct weights URL: `https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/resolve/main/model.safetensors`

   The card is manually gated. Accept the model license on Hugging Face first, then authenticate on the server with `huggingface-cli login` or set `HF_TOKEN`. To prefill the cache on a connected node:

```bash
MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"

huggingface-cli download "${MODEL_NAME}" \
  --include config.json preprocessor_config.json model.safetensors
```

1. Verify the model/checkpoint on the server with a two-image smoke test before any all-city run. If the checkpoint is already staged on disk, set `MODEL_NAME` to that local path and pass `--local-files-only`. If it is not found on disk and the node has internet access, set `MODEL_NAME` to the verified Hugging Face or timm model ID and omit `--local-files-only` so the backend downloads it into the model cache. After that first download, rerun with `--local-files-only` for production jobs.

```bash
MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"

python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5d_dinov3_embed_city.py \
  --city "Saidpur" \
  --valfolder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed_smoke \
  --model-name "${MODEL_NAME}" \
  --backend transformers \
  --batch-size 2 \
  --device cuda \
  --local-files-only \
  --limit 2
```

Download-on-miss smoke test, only for a connected node:

```bash
MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"

python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5d_dinov3_embed_city.py \
  --city "Hong Kong" \
  --valfolder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed_smoke \
  --model-name "${MODEL_NAME}" \
  --backend transformers \
  --batch-size 2 \
  --device cuda \
  --limit 2
```

2. Run one-city image embedding smoke test, for example Hong Kong with `--limit 256`, and validate row count, one `embedding_dim`, finite `e_*` columns, near-unit vector norms, and no duplicate `name`.

```bash
MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"

python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5d_dinov3_embed_city.py \
  --city "Hong Kong" \
  --valfolder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed \
  --model-name "${MODEL_NAME}" \
  --backend transformers \
  --batch-size 32 \
  --device cuda \
  --local-files-only \
  --limit 256
```

3. Run all-city image embeddings with `B5d_dinov3_embed_city.py`.

```bash
MODEL_NAME="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/model/checkpoint"
CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv

CITY_META="${CITY_META}" python - <<'PY' | while IFS= read -r CITY; do
import os
import pandas as pd
city_meta = pd.read_csv(os.environ["CITY_META"])
for city in city_meta["City"].dropna().drop_duplicates():
    print(city)
PY
  python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5d_dinov3_embed_city.py \
    --city "${CITY}" \
    --valfolder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir \
    --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed \
    --model-name "${MODEL_NAME}" \
    --backend transformers \
    --batch-size 64 \
    --device cuda \
    --local-files-only
done
```

4. Aggregate embeddings to H3 with `B5e_dinov3_vector_summary.py`; confirm every city has nonzero `res=8` rows and approximately unit-norm H3 vectors.

```bash
CITY_META=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv

CITY_META="${CITY_META}" python - <<'PY' | while IFS= read -r CITY; do
import os
import pandas as pd
city_meta = pd.read_csv(os.environ["CITY_META"])
for city in city_meta["City"].dropna().drop_duplicates():
    print(city)
PY
  python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5e_dinov3_vector_summary.py \
    --city "${CITY}" \
    --rootfolder /lustre1/g/geog_pyloo/05_timemachine \
    --input-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed \
    --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary \
    --train-test-folder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8 \
    --res-exclude 11 \
    --log-level INFO
done
```

5. Run pairwise cosine with the optimized B5b script against DINOv3 H3 vectors. For the first production DINOv3 run, use `--threshold -1.0` so the city-pair averages are not biased by dropping low, zero, or negative cosine similarities.

```bash
python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py \
  --resolution 8 \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --source-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair \
  --input-template 'dinov3_city={city}_res_exclude={res_exclude}.parquet' \
  --feature-prefix e_ \
  --threshold -1.0 \
  --row-block-size 1000 \
  --memory-limit 16GB \
  --log-dir logs/dinov3_similarity
```

6. Aggregate optimized pairwise outputs with B5c. This works after B5b has removed temp shards because B5c can fall back to merged `optimized/similarity_city=<City>_res=8_optimized.parquet` files.

```bash
python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py \
  --resolution 8 \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair/optimized/_progress_res=8_optimized.json \
  --resume \
  --agg-progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_agg_progress_res=8.json \
  --duckdb-memory-limit 32GB \
  --duckdb-temp-dir /lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_dinov3_similarity \
  --duckdb-threads 8 \
  --parquet-file-size 512MB
```

7. Build the city-pair model table:

```bash
python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B5h_summarize_dinov3_citypair_similarity.py \
  --input-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --output /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_summary_res=8.parquet \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv
```

Validation checklist:
- One embedding dimension across all city input files before pairwise computation starts.
- No null or non-finite image/H3 embedding values.
- H3 vectors are approximately unit norm after mean pooling.
- Every city has nonzero `res=8` H3 rows.
- Pairwise progress JSON reaches a completed state, or merged optimized outputs exist for every expected city.
- Final B5c aggregated outputs contain no rows where `city_1 == city_2`.
- B5h summary row count equals `n * (n - 1) / 2` unordered city pairs when `--city-meta` or `--expected-city-count` is provided.

4. process the distance between hexagons and their associated CBD
```python B6a_h3_distance_processor.py```

5. summarize inter-city similarity by landuse bucket from optimized pairwise shards.
```bash
python /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was/B7_similarity_by_landuse.py \
  --resolution 8 \
  --landuse-source stage3 \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair \
  --stage3-landuse-root /lustre1/g/geog_pyloo/05_timemachine/_transformed/landuse_poi_res=8 \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_by_landuse \
  --duckdb-memory-limit 32GB \
  --duckdb-temp-dir /lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_city_similarity_by_landuse
```

   Notes:
   - `B7_similarity_by_landuse.py` depends on the detailed optimized pairwise temp shards under `optimized/temp/city1=*/city2=*/part_res=<resolution>.parquet`
   - It also supports legacy B5c-style per-city parquet outputs in both forms: a single parquet file like `.../similarity_intracity_city=<city>_res=8.parquet` and a parquet dataset directory like `.../similarity_intracity_city=<city>_res=8.parquet/part_0.parquet`
   - If `--pairwise-root` is omitted, B7 now defaults to the pre-aggregated source `.../_curated/c_city_similarity_optimized_res=<resolution>`
   - Legacy CLI aliases are supported via the `similarity_by_landuse.py` wrapper: `--res` maps to `--resolution`, and `--use-two-phase` streams one parquet dataset at a time to reduce memory pressure
   - `B5c_pairwise_agg_optimized.py` finishing is useful context, but its city-level outputs are not enough for landuse filtering because B7 still needs `hex_id1` and `hex_id2`
   - The migrated B7 script defaults to remote-server paths instead of the older local Dropbox project paths
   - Use `--check-only` to inspect landuse resolution coverage before a full run
   - The current stage3 POI-tier data appears sparse at `res=8`; if coverage is present for fewer than two cities the script stops unless you add `--allow-sparse-landuse`

# Use ochestrator. To be updated to include all steps later.
```
# Run all pipelines
python run_pipelines.py --pipeline all --city-meta ../city_meta.csv

# Run only distance pipeline for specific resolutions
python run_pipelines.py --pipeline distance --resolutions 6 7 8 --log-level DEBUG

# Run similarity pipeline with resume capability
python run_pipelines.py --pipeline similarity --resume

# Use custom configuration
python run_pipelines.py --config my_config.json --pipeline all```
