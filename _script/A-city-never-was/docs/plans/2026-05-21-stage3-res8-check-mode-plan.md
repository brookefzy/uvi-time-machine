# Stage3 Res8 Check Mode Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight dry-run/check mode to `stage3_poi_density_diversity_tiers.py` so it reports which cities are eligible to generate `res=8` outputs before a full run.

**Architecture:** Reuse the script's existing prerequisite gate, `load_city_hex_ids(city, res)`, to build a cheap preflight helper that only inspects POI filenames and similarity-hex availability. Expose it through a `--check-only` CLI flag that prints a readiness summary and exits before any geospatial aggregation.

**Tech Stack:** Python, unittest, importlib, stubbed dependency modules

### Task 1: Add a failing test for the preflight helper

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py`
- Reference: `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py`

**Step 1: Write the failing test**

Cover:
- helper returns `ready` for cities with `res=8` hex IDs
- helper returns `missing_hex_ids` otherwise

**Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py
```

Expected: FAIL because the helper does not exist yet.

### Task 2: Implement the check mode

**Files:**
- Modify: `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py`

**Step 1: Add helper and CLI flag**

Implement:
- a city-status helper based on `load_city_hex_ids`
- a `--check-only` flag that prints a readiness report and exits

**Step 2: Run test to verify it passes**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py
```

Expected: PASS.

### Task 3: Verify syntax

**Files:**
- Modify: `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py`

**Step 1: Compile-check**

Run:

```bash
python3 -m py_compile '/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py'
```

Expected: compile succeeds cleanly.
