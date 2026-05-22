# B7 Mixed Optimized City Inputs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `B7_similarity_by_landuse.py` read optimized-city inputs when the pairwise root contains a mix of standalone parquet files and parquet dataset directories.

**Architecture:** Keep source detection unchanged at the root level, but replace the single broad `read_parquet('*res=<n>.parquet')` usage for optimized-city inputs with an explicit file expansion step. Expand standalone `.parquet` files directly and expand dataset directories to their inner `*.parquet` parts, then feed the resulting concrete paths to DuckDB.

**Tech Stack:** Python, DuckDB, unittest

### Task 1: Add failing regression

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py`

**Step 1: Write the failing test**

Add a test that creates:
- one file named `similarity_intracity_city=Sydney_res=8.parquet`
- one directory named `similarity_intracity_city=Astrakhan_res=8.parquet` containing `part_0.parquet`

Assert that the optimized-city path expansion returns both concrete parquet paths.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py`

Expected: fail because the helper does not exist yet.

### Task 2: Implement explicit parquet path expansion

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B7_similarity_by_landuse.py`

**Step 1: Add helper**

Add a helper that:
- takes a `PairwiseSourceConfig`
- returns explicit parquet file paths
- for `optimized_city`:
  - keeps standalone `.parquet` files
  - expands `.parquet` directories to their inner `*.parquet` parts
- for temp shards:
  - returns the existing matched shard files

**Step 2: Route all `read_parquet(...)` calls through the helper**

Use the expanded concrete paths everywhere pairwise inputs are read.

### Task 3: Verify

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/README.md` only if behavior description needs clarification

**Step 1: Run tests**

Run: `python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py`

Expected: PASS

**Step 2: Compile**

Run: `python3 -m py_compile /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B7_similarity_by_landuse.py /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/similarity_by_landuse.py`

Expected: PASS
