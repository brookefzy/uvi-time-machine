# Usage
* TO-DO: add previous steps
1.  summarizes the all probablity vector at resolution 6,7,8 for each city
```python B5a_prob_vector_summary.py --city all```
2. summarizes the pairwire similarity index, saved by each city.
```python B5b_compute_similarity_pairwise.py --res 6``` 
   Optimized resumable version:
```bash
python /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py \
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
   - Default log folder: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/logs`
   - Optional override: add `--log-dir /path/to/logs`

3. summarize similarity index for each city.
```python B5c_pairwise_agg.py``` 
   Pairing:
   - Use `B5c_pairwise_agg.py` with `B5b_compute_similarity_pairwise.py`
   - Use `B5c_pairwise_agg_optimized.py` with `B5b_compute_similarity_pairwise-optimized.py`

   Optimized aggregation example:
```bash
python /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --resolution 8 \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_optimized_res=8 \
  --progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair/optimized/_progress_res=8_optimized.json \
  --resume \
  --agg-progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_optimized_progress_res=8.json \
  --duckdb-memory-limit 32GB \
  --duckdb-temp-dir /lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_city_similarity
```

   Notes:
   - `B5c_pairwise_agg_optimized.py` reads temp shards from `optimized/temp/city1=<city>/city2=*/part_res=<resolution>.parquet`
   - It does not require merged per-city optimized parquet files to exist first
   - `--resume` skips cities that already have aggregated output files, and if `--agg-progress-file` is provided it also resumes from that city-level progress JSON
   - Use an explicit `--export-folder` when resuming. The script's default export folder is date-based, so a bare `python B5c_pairwise_agg_optimized.py --resume` on a later day will look in a new folder and will not see yesterday's completed city outputs.
   - The optimized aggregator keeps exact results by default and exports directly from DuckDB instead of materializing the full city result in pandas
   - If the progress JSON has no pending pairs but still says `in_progress`, the optimized aggregator proceeds and logs a warning

4. process the distance between hexagons and their associated CBD
```python B6a_h3_distance_processor.py```

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
