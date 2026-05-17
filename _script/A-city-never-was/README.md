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
  --group-size 10
```

   Resume behavior:
   - If `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair/optimized/_progress_res=6_optimized.json` exists, the script resumes from its pending city list.
   - If that JSON file does not exist, the script scans existing `similarity_city=*_optimized.parquet` files in the `optimized/` output folder and skips cities that are already finished.
   - Use `--fresh-start` to ignore previous progress and recompute from scratch.

   Local debug logs:
   - Default log folder: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/logs`
   - Optional override: add `--log-dir /path/to/logs`

3. summarize similarity index for each city.
```python B5c_pairwise_agg.py``` 
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
