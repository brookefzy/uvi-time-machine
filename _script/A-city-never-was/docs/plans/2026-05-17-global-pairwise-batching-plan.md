# Exact Global Pairwise Batching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the original script's exact global city-pair similarity behavior while making resolution-8 runs feasible by batching within each city pair and writing intermediate results to disk.

**Architecture:** Restore the original outer loop over all unique `(city1, city2)` pairs, then replace whole-pair in-memory cosine computation with blockwise exact cosine over `city1` and `city2` feature chunks. Persist per-pair temp shards and checkpoint at the city-pair level so interrupted runs can resume without changing output semantics.

**Tech Stack:** Python, pandas, NumPy, DuckDB, parquet/pyarrow, `unittest`, `tqdm`

### Task 1: Lock in exact global semantics with failing tests

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py`
- Reference: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise.py`
- Reference: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py`

**Step 1: Write the failing test for global pair generation**

Add a test that seeds `city_meta.csv` with `["Alpha", "Beta", "Gamma"]`, calls a new helper on the optimized processor, and asserts the exact ordered pairs are:

```python
[
    ("Alpha", "Beta"),
    ("Alpha", "Gamma"),
    ("Beta", "Gamma"),
]
```

**Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: FAIL because the optimized script still batches cities instead of enumerating global city pairs.

**Step 3: Write the failing test for resume over completed city pairs**

Add a test that writes a checkpoint containing one completed pair, runs the optimized processor with three cities, and asserts only the remaining global city pairs are processed.

**Step 4: Run the tests to verify they fail for the expected reason**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: FAIL on missing pair-level semantics or wrong processed pair sequence.

**Step 5: Commit**

```bash
git add /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
git commit -m "test: add global pair batching regressions"
```

### Task 2: Restore global pair enumeration and pair-level resume

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py`

**Step 1: Implement minimal global pair helpers**

Add helpers modeled on the original script:

```python
def create_city_pairs(self, city_meta_path: str) -> List[Tuple[str, str]]:
    ...

def pair_to_key(self, city1: str, city2: str, resolution: int) -> str:
    return f"{city1}__{city2}__res={resolution}"
```

**Step 2: Replace city-list progress with pair-list progress**

Change checkpoint payloads from:

```python
{"completed_cities": [...], "pending_cities": [...]}
```

to exact pair-based state such as:

```python
{
    "completed_pair_keys": [...],
    "pending_pairs": [["Alpha", "Beta"], ["Alpha", "Gamma"]],
}
```

**Step 3: Rewrite `run()` to iterate global city pairs**

Replace the current `pending_cities[i : i + group_size]` loop with:

```python
for city1, city2 in pending_pairs:
    process_one_city_pair(...)
```

Keep resume default behavior and the existing `--fresh-start` escape hatch.

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: PASS for the new global-pair and pair-resume tests.

**Step 5: Commit**

```bash
git add /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
git commit -m "feat: restore exact global city pair iteration"
```

### Task 3: Add exact blocked similarity inside one city pair

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py`

**Step 1: Write the failing test for blocked pair processing**

Add a unit test that monkeypatches feature loading for two tiny cities and asserts:
- blocked processing emits the same similarity rows as direct full-matrix processing
- same-city comparisons still keep only the upper triangle

**Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: FAIL because no exact pairwise blocked method exists yet.

**Step 3: Implement the minimal blocked exact processor**

Add a new method similar to:

```python
def compute_city_pair_similarity_blocked(
    self,
    city1: str,
    city2: str,
    resolution: int,
    row_block_size: int,
) -> pd.DataFrame:
    ...
```

Implementation rules:
- load `df1` and `df2` separately
- iterate row blocks over `df1` and `df2`
- compute exact cosine for each block pair
- for same-city comparisons, keep only the upper triangle and skip duplicate/self matches
- emit result rows without materializing the full dense pair matrix

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: PASS for the blocked exact-equivalence test.

**Step 5: Commit**

```bash
git add /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
git commit -m "feat: add exact blocked city pair similarity"
```

### Task 4: Write per-pair temp shards and merge into final city outputs

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py`
- Reference: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py`

**Step 1: Write the failing test for pair temp shard layout**

Add a test asserting that processing one pair writes shards under a dedicated temp directory such as:

```text
optimized/temp/city1=Alpha/city2=Beta/part-00000.parquet
```

and that incomplete pairs are not marked complete before all shards are written.

**Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: FAIL because the current optimized script writes directly to final per-city outputs.

**Step 3: Implement temp shard writing and merge phase**

Add helpers like:

```python
def save_temp_similarity_shard(...): ...
def merge_city_results(...): ...
def cleanup_temp_pair(...): ...
```

Implementation rules:
- write temp shard parquet files atomically
- only mark the pair complete after all its shards are written
- merge all `city2=*` temp data for each `city1` into final `similarity_city=<city1>_optimized.parquet`

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: PASS for shard layout and completion-order checks.

**Step 5: Commit**

```bash
git add /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
git commit -m "feat: stream exact pair results to temp shards"
```

### Task 5: Add CLI tuning knobs, docs, and final verification

**Files:**
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py`
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/README.md`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py`

**Step 1: Write the failing test for CLI-backed block size configuration**

Add a test that initializes the processor with a configured row block size and asserts the blocked pair method reads that value from config/CLI.

**Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
```

Expected: FAIL because the optimized script does not yet expose the new block-size control or updated exact semantics in docs.

**Step 3: Implement the minimal CLI/doc changes**

Add a CLI flag such as:

```bash
--row-block-size 1000
```

and update the README to explain:
- exact global pair behavior
- pair-level resume
- temp shard disk usage
- local logs and checkpoint file locations

**Step 4: Run full verification**

Run:

```bash
python3 -m unittest /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py
python3 -m py_compile /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py
```

Expected: all tests PASS, script compiles without syntax errors.

**Step 5: Commit**

```bash
git add /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/test_b5b_compute_similarity_pairwise_optimized.py /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/README.md
git commit -m "docs: document exact resumable pair batching"
```

