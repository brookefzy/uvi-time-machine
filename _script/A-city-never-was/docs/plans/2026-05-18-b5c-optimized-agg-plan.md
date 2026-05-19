# Optimized Pairwise Aggregation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `B5c_pairwise_agg_optimized.py` to aggregate the optimized temp shard layout from `B5b_compute_similarity_pairwise-optimized.py` into the same downstream-friendly inter-city outputs as the original `B5c` pipeline.

**Architecture:** Keep the original `B5c_pairwise_agg.py` untouched. Create a new optimized companion script that reads `optimized/temp/city1=*/city2=*/part_res=<res>.parquet`, deduplicates unordered hex pairs in DuckDB, splits inner-city vs inter-city rows using shard-provided city labels, and writes per-`city1` aggregated parquet outputs.

**Tech Stack:** Python, DuckDB, pandas, pathlib, unittest

### Task 1: Add failing tests for the optimized temp-shard contract

**Files:**
- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5c_pairwise_agg_optimized.py`
- Reference: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg.py`

**Step 1: Write the failing tests**

Cover:
- reading `temp/city1=<city1>/city2=*/part_res=<res>.parquet`
- deduplicating unordered hex pairs with max similarity
- splitting `city1 == city2` vs `city1 != city2`
- skipping cleanly when a city has no temp shards

**Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5c_pairwise_agg_optimized.py
```

Expected: FAIL because the optimized aggregator does not exist yet.

### Task 2: Implement `B5c_pairwise_agg_optimized.py`

**Files:**
- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5c_pairwise_agg_optimized.py`

**Step 1: Implement the minimal processor**

Build a new processor that:
- locates shard files under `optimized/temp/city1=<city>/city2=*/part_res=<res>.parquet`
- unions them with DuckDB
- deduplicates by unordered hex pair
- returns `city_1` / `city_2` columns for downstream compatibility

**Step 2: Run the tests to verify they pass**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5c_pairwise_agg_optimized.py
```

Expected: PASS.

### Task 3: Add CLI/docs and final verification

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/README.md`
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/requirements.txt` only if needed
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5c_pairwise_agg_optimized.py`

**Step 1: Document the new pairing**

Explain that:
- `B5b_compute_similarity_pairwise.py` pairs with `B5c_pairwise_agg.py`
- `B5b_compute_similarity_pairwise-optimized.py` pairs with `B5c_pairwise_agg_optimized.py`

**Step 2: Run full verification**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5c_pairwise_agg_optimized.py
python3 -m py_compile /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py
```

Expected: all tests PASS, script compiles cleanly.
