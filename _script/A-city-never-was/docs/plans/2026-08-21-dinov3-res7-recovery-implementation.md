# DINOv3 Resolution-7 Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and verify an idempotent pipeline that audits and recovers the seven affected DINOv3 resolution-7 cities without imputing missing similarities.

**Architecture:** Add a Python audit/manifest/validation CLI and a thin Bash Slurm orchestrator that reuse B5d, B5e, B5b, and B5c. Keep all regenerated post-embedding artifacts in a timestamped recovery root until a strict validator writes a READY marker.

**Tech Stack:** Python 3, pandas/pyarrow, H3, pytest, Bash, Slurm (`sbatch`, `squeue`, `sacct`).

### Task 1: City-source discovery and boundary audit

**Files:**
- Create: `dinov3_res7_recovery.py`
- Create: `test_dinov3_res7_recovery.py`

1. Write failing fixtures for exact stems, decomposed Unicode, state/country suffixes, known historical aliases, ambiguous candidates, and absent imagery evidence.
2. Run `python3 -m pytest -q test_dinov3_res7_recovery.py` and confirm failures are caused by the missing module/API.
3. Implement conservative candidate discovery and per-boundary counts.
4. Re-run the focused tests and retain JSON-serializable evidence for every checked path.

### Task 2: Current H3 and Sitapur diagnosis

**Files:**
- Modify: `dinov3_res7_recovery.py`
- Modify: `test_dinov3_res7_recovery.py`

1. Add failing tests where embedded panoids join current metadata but occupy stale/non-core H3 cells, and where core cells have source panoids but missing embeddings.
2. Implement current-coordinate H3 recomputation and overlap partitions against required/core membership.
3. Assert that “no source imagery” is distinct from “source exists but embedding/H3 is missing.”

### Task 3: Affected-only pair manifest

**Files:**
- Modify: `dinov3_res7_recovery.py`
- Modify: `test_dinov3_res7_recovery.py`

1. Add failing tests proving every pair has at least one recovered endpoint, both H3 inputs are usable at res=7, ordering is deterministic, and unaffected pairs are excluded.
2. Implement manifest generation with expected-shard status fields.
3. Verify focused tests pass.

### Task 4: Final export validator and orientation regression

**Files:**
- Modify: `dinov3_res7_recovery.py`
- Modify: `test_dinov3_res7_recovery.py`
- Modify only if a reproduced defect requires it: `B5c_pairwise_agg_optimized.py`

1. Add failing tests for reversed endpoints, nulls, exact-zero sentinels, duplicates, invalid ranges, and missing required-city pairs.
2. Implement strict five-column validation and membership-backed endpoint checks.
3. If B5c fails the fixture, fix the source orientation logic minimally and run its existing membership tests.

### Task 5: Slurm orchestration

**Files:**
- Create: `pipeline/run_dinov3_res7_recovery.bash`
- Create: `slurm/dinov3_res7_embed.cmd`
- Create: `slurm/dinov3_res7_h3.cmd`
- Modify: `test_dinov3_res7_recovery.py`

1. Add failing contract tests for recovery-root isolation, explicit city manifests, completed-shard preservation, `sacct` terminal-state checks, job-ID logging, and READY-after-validation ordering.
2. Implement dry-run/preflight and execute modes; require Stage-3 membership paths for execution.
3. Run `bash -n` on every new shell file and focused pytest tests.

### Task 6: Verification and operator documentation

**Files:**
- Modify: `pipeline/INDEX.md`
- Modify: `README.md`

1. Document preflight, execution, resume, output layout, and exact remote command.
2. Run all DINOv3-related tests plus shell syntax checks.
3. Run the CLI against local fixtures and inspect emitted JSON/CSV/parquet validation reports.
4. Search for temporary debug markers and inspect `git diff --check`.

Remote execution remains a separate evidence phase: synchronize the reviewed
changes to Lustre, run preflight, resolve any ambiguous aliases from its report,
execute, monitor to terminal states, and attach the generated root-cause table,
commands, job IDs, counts, logs, and READY artifacts.
