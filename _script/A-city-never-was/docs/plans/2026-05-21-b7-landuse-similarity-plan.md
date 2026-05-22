# B7 Landuse Similarity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a B7 script in `A-city-never-was` that summarizes inter-city similarity by landuse bucket from the optimized pairwise temp shards, using remote-server path defaults and a res-8 landuse coverage preflight.

**Architecture:** Keep the old Dropbox project script as the semantic reference, but migrate only the logic needed for this repo. The new script should read optimized pairwise temp shards from the same tree consumed by `B5c_pairwise_agg_optimized.py`, derive similarity-covered hex IDs directly from those shards, load landuse hex definitions from configurable parquet roots, aggregate filtered similarities in DuckDB, and export one CSV per landuse bucket.

**Tech Stack:** Python, DuckDB, pandas, pathlib, dataclasses, unittest

### Task 1: Capture the migration findings in tests

**Files:**
- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py`
- Reference: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py`

**Step 1: Write the failing tests**

Cover:
- remote default path wiring for pairwise and landuse sources
- city normalization behavior needed for cross-file joins
- resolution coverage preflight for stage-3 landuse files
- summary output naming

**Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py
```

Expected: FAIL because the B7 script does not exist yet.

### Task 2: Implement the B7 processor

**Files:**
- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B7_similarity_by_landuse.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py`

**Step 1: Implement the minimal processor**

Build a script that:
- reads `optimized/temp/city1=*/city2=*/part_res=<res>.parquet`
- derives `(city_lower, hex_id)` coverage from those detailed shards
- loads stage-2 or stage-3 landuse hex IDs from configurable parquet roots
- aggregates filtered inter-city similarities into CSV summaries
- emits a clear warning when requested landuse coverage is sparse, especially for stage-3 `res=8`

**Step 2: Run the tests to verify they pass**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py
```

Expected: PASS.

### Task 3: Wire docs and verify the script

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/README.md`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py`

**Step 1: Document the new B7 step**

Explain:
- B7 depends on detailed pairwise shards, not only B5c aggregate outputs
- default inputs target remote-server paths
- stage-3 `res=8` currently appears sparse and should be checked before full runs

**Step 2: Run verification**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b7_similarity_by_landuse.py
python3 -m py_compile /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B7_similarity_by_landuse.py
```

Expected: tests PASS and the script compiles cleanly.
