# Usage
* TO-DO: add previous steps
1.  summarizes the all probablity vector at resolution 6,7,8 for each city
```python B5a_prob_vector_summary.py --city all```
2. summarizes the pairwire similarity index, saved by each city.
```python B5b_compute_similarity_pairwise.py --res 6``` 
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