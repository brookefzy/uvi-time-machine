# Stage3 Res8-Only Output Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `stage3_poi_density_diversity_tiers.py` so it only processes `res=8` and writes outputs to a separate `landuse_poi_res=8` folder.

**Architecture:** Keep the existing stage3 pipeline intact and make the smallest possible config-level change. Verify the behavior with an import-level regression test that checks the script now targets only `res=8` and the new output folder path.

**Tech Stack:** Python, unittest, importlib, stubbed modules for heavy geospatial dependencies

### Task 1: Add a failing regression test

**Files:**
- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py`
- Reference: `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py`

**Step 1: Write the failing test**

Check:
- `RESOLUTIONS == [8]`
- `OUTPUT_FOLDER` ends with `landuse_poi_res=8`

**Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py
```

Expected: FAIL against the current script because it still targets `[6, 7]` and `landuse_poi`.

### Task 2: Update the external stage3 script

**Files:**
- Modify: `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py`

**Step 1: Change the configuration**

Update:
- `OUTPUT_FOLDER` to `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/landuse_poi_res=8`
- `RESOLUTIONS` to `[8]`
- sample plot resolutions to `[8]`

**Step 2: Run the test to verify it passes**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_stage3_poi_density_diversity_tiers_config.py
```

Expected: PASS.

### Task 3: Compile-check the updated script

**Files:**
- Modify: `/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py`

**Step 1: Run compile verification**

Run:

```bash
python3 -m py_compile '/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_scripts/03_landuse/stage3_poi_density_diversity_tiers.py'
```

Expected: compile succeeds cleanly.
