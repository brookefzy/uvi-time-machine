# DINOv3 Visual Similarity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a parallel DINOv3 visual-similarity pipeline that produces city-pair similarity measures comparable to the existing classifier-probability B5 pipeline without overwriting baseline artifacts.

**Architecture:** Keep `stage1_classifier/`, `B5a_prob_vector_summary.py`, `B5b_compute_similarity_pairwise-optimized.py`, and `B5c_pairwise_agg_optimized.py` behavior stable for classifier-probability outputs. Add DINOv3-specific embedding and H3-summary scripts, then make the optimized pairwise/aggregation code accept configurable vector columns and filename conventions where the change is low-risk. This avoids duplicating the large B5b/B5c checkpointing logic while keeping DINOv3 outputs in separate folders.

**Tech Stack:** Python, pandas, numpy, pyarrow/parquet, DuckDB, h3, PyTorch, transformers or timm, Pillow, scikit-learn cosine similarity, unittest/pytest-compatible tests.

## Current Baseline Contracts

- `stage1_classifier/B3_inference_city_prob.py` reads image-list parquet files from `/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir/<cityabbr>.parquet` and writes chunked classifier probability parquet files under `_curated/c_city_classifiier_prob/<cityabbr>/`.
- `B5a_prob_vector_summary.py` joins image predictions to panorama metadata by `panoid`, excludes train/test leakage by H3 cells at resolutions 11, 12, and 13, aggregates fixed classifier columns `"0"` through `"126"` to H3 resolutions 6, 7, and 8, and writes `prob_city=<City>_res_exclude=<res>.parquet`.
- `B5b_compute_similarity_pairwise-optimized.py` currently assumes `prob_city=...` input files and fixed columns `"0"` through `"126"`. Its checkpointing, temp shard layout, and pairwise blocking should be reused.
- `B5b_compute_similarity_pairwise-optimized.py` deletes `optimized/temp/` after a fully successful merge. `B5c_pairwise_agg_optimized.py` already handles this by reading temp shards first and falling back to merged `optimized/similarity_city=<City>_res=<res>_optimized.parquet` files.
- `B5c_pairwise_agg_optimized.py` already accepts `--pairwise-root` and `--export-folder`; it mostly depends on `hex_id1`, `hex_id2`, `city1`, `city2`, and `similarity`, so it should be reusable with only small metadata pass-through changes if needed. Its inherited output filename says `similarity_intracity...`, but the current query exports `city_1 != city_2` inter-city rows.
- The existing tests live at repo root as `test_*.py`, not under `tests/`. New tests should follow that convention unless the repo is reorganized separately.

## Engineering Review Updates

1. Add a checkpoint/model verification gate before any all-city embedding run. Do not hard-code a speculative Hugging Face model ID in production commands; require `--model-name` or `--model-path` to be verified on the server with a two-image smoke test first. If the model is not present on disk and the node has internet access, omit `--local-files-only` for the smoke test so the backend downloads the verified model ID into cache, then use `--local-files-only` for repeat production runs.
2. When parameterizing `B5b_compute_similarity_pairwise-optimized.py`, update both loaders, not only the CLI:
   - `__init__` must select either fixed classifier columns or discovered `e_` columns.
   - `load_features_optimized()` must use the configurable source root and input filename template.
   - `load_city_features()` must use the same template and column discovery path.
   - `save_pair_results()` and `merge_city_results()` may preserve `metric` and `model_name` only if doing so is constant per shard and does not change classifier defaults.
3. Pairwise output has two useful layers: intermediate temp shards and merged optimized files. Tests may assert temp shard writes, but runbooks should treat merged files as the stable input to B5c after a completed B5b run unless a new `--keep-temp` flag is added.
4. DINOv3 H3 aggregation should copy the exclusion mechanics from `B5a_prob_vector_summary.py`, but not its classifier-specific `compute_statistics()` logic (`max_class`, `max_prob`, `second_class`).
5. `atomic_write_parquet()` must write the temporary file in the destination directory, create parent directories first, and use `Path.replace()` so cross-filesystem renames do not break on Lustre.

## What Already Exists

- `B3_inference_city_prob.py` already provides the per-city chunk/resume shape; Task 2 should reuse that behavior but avoid importing the YOLO model at module import time.
- `B5a_prob_vector_summary.py` already contains the pano metadata join, H3 conversion compatibility handling, and train/test H3 exclusion logic; Task 3 should adapt those parts and replace only vector columns, input roots, output roots, and stats.
- `B5b_compute_similarity_pairwise-optimized.py` already contains city-pair ordering, progress checkpointing, bounded row blocks, atomic shard writes, merged optimized outputs, and purge/resume controls; Task 4 should parameterize those paths instead of forking the script.
- `B5c_pairwise_agg_optimized.py` already reads temp shards or merged optimized files from `--pairwise-root` and writes filtered inter-city outputs to `--export-folder`; Task 5 should begin with a reuse test before changing code.
- `requirements.txt` already includes the core data stack (`pandas`, `numpy`, `duckdb`, `h3`, `scipy`, `scikit-learn`, `pyarrow`, `pytest`) and optional `cupy`; Task 1 only needs to add image/model dependencies.

## Key Decisions

1. Create new DINOv3 image and H3 scripts instead of modifying `stage1_classifier/`; the classifier stage is a baseline and should remain reproducible.
2. Add a small shared utility module for vector-column discovery, L2 normalization, model loading, and atomic parquet writes.
3. Parameterize `B5b_compute_similarity_pairwise-optimized.py` rather than fork it if the edits stay narrow: source root, output root, filename template, vector prefix, metric label, and optional metadata columns.
4. Reuse `B5c_pairwise_agg_optimized.py` with its existing CLI roots. Modify only if DINOv3 metadata columns such as `metric` or `model_name` must be preserved in aggregated outputs.
5. Treat the DINOv3 model identifier as an environment-specific configuration. The DINOv3 paper is real, but the exact local/Hugging Face checkpoint name must be verified on the server before production.

## Output Contract

Use separate roots:

- Image embeddings: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed/<cityabbr>/<cityabbr>_<timestamp>.parquet`
- H3 embeddings: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary/dinov3_city=<City>_res_exclude=<res>.parquet`
- Pairwise shards: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair/optimized/temp/city1=<City>/city2=<City>/part_res=<res>.parquet`
- Merged pairwise outputs: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair/optimized/similarity_city=<City>_res=<res>_optimized.parquet`
- Aggregated city outputs: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_<YYYYMMDD>/similarity_intracity_city=<City>_res=<res>.parquet`
- City-pair model table: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_summary_res=<res>.parquet`

Required columns:

- Image embedding rows: `name`, `panoid`, `model_name`, `embedding_dim`, `e_0000` ... `e_<dim-1>`.
- H3 embedding rows: `hex_id`, `res`, `img_count`, `model_name`, `embedding_dim`, `e_0000` ... `e_<dim-1>`.
- Pairwise rows: `hex_id1`, `hex_id2`, `city1`, `city2`, `similarity`, plus `metric="cosine"` and `model_name` if preserved cheaply.
- City-pair summary rows: `city_1`, `city_2`, `dinov3_cosine_avg`, `dinov3_cosine_p50`, `dinov3_cosine_p90`, `dinov3_cosine_p95`, `dinov3_cosine_max`, `dinov3_pair_count_observed`.

## Task 1: Add Shared DINOv3 Vector Utilities

**Files:**

- Create: `dinov3_utils.py`
- Create: `test_dinov3_utils.py`
- Modify: `requirements.txt`

**Steps:**

1. Write tests for `build_embedding_columns(dim)`, `discover_embedding_columns(df)`, `l2_normalize_rows(values)`, and `atomic_write_parquet(df, path)`.
2. Run `python -m pytest test_dinov3_utils.py -q`; expected failure because the module does not exist.
3. Implement helpers:
   - `build_embedding_columns(dim: int) -> list[str]`
   - `discover_embedding_columns(df: pd.DataFrame, prefix: str = "e_") -> list[str]`
   - `l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray`
   - `atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> None`
   - `load_rgb_image(path: str) -> PIL.Image.Image`
   - `resolve_city_file_stem(city: str, override: str | None = None) -> str`
4. Make `discover_embedding_columns()` deterministic and defensive:
   - Sort columns by numeric suffix, not plain string sort.
   - Reject missing or duplicate numeric suffixes unless the caller passes `strict=False`.
   - Ignore metadata columns such as `embedding_dim`.
5. Add model helpers with dependency injection so tests do not download weights:
   - `load_embedding_backend(model_name, device, backend, local_files_only)`
   - `embed_image_batch(paths, model, processor, device) -> np.ndarray`
6. Add a no-network checkpoint smoke helper:
   - `verify_embedding_backend(model_name_or_path, backend, device, local_files_only, sample_paths) -> dict`
   - Return `model_name`, `embedding_dim`, `batch_size`, `device`, and a finite/unit-norm check.
   - When `local_files_only=False`, allow the backend to download a verified model ID into the server cache if the checkpoint is not already present on disk.
7. Add runtime dependencies to `requirements.txt`: `torch`, `torchvision`, `transformers`, `Pillow`, and optionally `timm` if the server checkpoint is exposed through timm.
8. Re-run `python -m pytest test_dinov3_utils.py -q`; expected pass.

## Task 2: Add Per-City DINOv3 Image Embedding

**Files:**

- Create: `B5d_dinov3_embed_city.py`
- Create: `test_dinov3_embed_city.py`

**Steps:**

1. Write tests with temporary parquet inputs and a fake embedder. Cover city-name-to-file-stem conversion, resume behavior from existing chunk files, `panoid = name[:22]`, deterministic embedding columns, and L2-normalized output.
2. Run `python -m pytest test_dinov3_embed_city.py -q`; expected failure.
3. Implement CLI:

```bash
python B5d_dinov3_embed_city.py \
  --city "Hong Kong" \
  --valfolder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed \
  --model-name <verified-dinov3-checkpoint-or-local-path> \
  --backend transformers \
  --batch-size 64 \
  --device cuda \
  --local-files-only
```

If the checkpoint is not present on disk and the server node is allowed to reach the model registry, run the same smoke test once without `--local-files-only` and with `--model-name <verified-huggingface-or-timm-dinov3-model-id>` so the backend downloads the model into cache. After the cache is populated, keep `--local-files-only` for production all-city runs.

4. Add `--city-file-stem` for names with punctuation or non-standard server file stems. Default to `resolve_city_file_stem(city)`.
5. Mirror `B3_inference_city_prob.py` resume semantics, but use atomic writes and expose `--limit` for smoke tests.
6. Store chunks under `<output-root>/<cityabbr>/` and skip any `name` already present in previous chunks.
7. Validate each written shard: all embeddings finite, one `embedding_dim`, no duplicate `name`, and L2 norm within tolerance for nonzero rows.
8. Re-run `python -m pytest test_dinov3_embed_city.py -q`; expected pass.
9. Server smoke test: run Hong Kong with `--limit 256`, then verify row count, constant `embedding_dim`, finite embeddings, near-unit norms, and no duplicate `name`.

## Task 3: Add DINOv3 H3 Aggregation

**Files:**

- Create: `B5e_dinov3_vector_summary.py`
- Create: `test_dinov3_vector_summary.py`

**Steps:**

1. Write fixture tests for aggregation and train/test exclusion. Use fake image embeddings, fake panorama metadata, and one training panoid whose exclusion-resolution H3 cell overlaps an embedding row.
2. Run `python -m pytest test_dinov3_vector_summary.py -q`; expected failure.
3. Adapt the logic from `H3HexagonAggregator` in `B5a_prob_vector_summary.py`:
   - Input root defaults to `_curated/c_city_dinov3_embed`.
   - Output root defaults to `_curated/c_city_dinov3_hex_summary`.
   - Discover vector columns by `e_` prefix.
   - Keep summary resolutions `[6, 7, 8]`.
   - Keep exclusion resolutions `[11, 12, 13]`.
   - Keep `PANO_PATH`, `PATH_PATH`, and train/test folder conventions.
   - Add `img_count` in each H3 group.
   - L2-normalize averaged H3 vectors before writing.
4. Do not reuse classifier-only stats from `compute_statistics()` in `B5a_prob_vector_summary.py`; those assume class probabilities and `max_class`.
5. Handle empty joins or total exclusion explicitly: write no parquet, write a sidecar JSON with `status="empty"`, and return a nonzero CLI exit unless `--allow-empty` is passed.
6. Write sidecar JSON stats for each output with city, model name, embedding dimension, image counts before exclusion, image counts after exclusion, excluded image count, res8 H3 count, and res8 mean image count.
7. Re-run `python -m pytest test_dinov3_vector_summary.py -q`; expected pass.
8. Server smoke test:

```bash
python B5e_dinov3_vector_summary.py --city "Hong Kong" --res-exclude 11 --log-level INFO
```

Expected: `dinov3_city=Hong Kong_res_exclude=11.parquet` contains nonzero `res=8` rows.

## Task 4: Parameterize Optimized Pairwise Similarity for DINOv3

**Files:**

- Modify: `B5b_compute_similarity_pairwise-optimized.py`
- Create: `test_b5b_dinov3_vector_contract.py`

**Steps:**

1. Write tests that instantiate `OptimizedSimilarityProcessor` against fake DINOv3 H3 files named `dinov3_city=Alpha_res_exclude=11.parquet`.
2. Test that `--feature-prefix e_` discovers embedding columns, filters `res=8`, computes cosine similarity, writes temp shards to the DINOv3 output root, and preserves existing classifier defaults when no new flags are passed.
3. Run `python -m pytest test_b5b_dinov3_vector_contract.py test_b5b_compute_similarity_pairwise_optimized.py -q`; expected failure only for the new DINOv3 test.
4. Add config/CLI options:
   - `--source-root`
   - `--output-root`
   - `--input-template`, default `prob_city={city}_res_exclude={res_exclude}.parquet`
   - `--feature-prefix`, default empty or `None` for classifier fixed columns
   - `--metric-label`, default `cosine`
   - `--res-exclude`, default `11`
   - `--keep-temp`, default false, only if server debugging needs temp shards after merge.
5. Refactor exact internal points:
   - Add `build_input_path(city)` and use it from both `load_features_optimized()` and `load_city_features()`.
   - Add `resolve_vector_columns(df_or_path)` and call it once per input contract, with a consistency check across cities.
   - Add optional metadata constants to pairwise rows only after similarity is computed.
6. Keep default behavior byte-compatible for classifier runs: fixed columns `"0"` through `"126"`, classifier source root, classifier output root, threshold `0.01`, existing progress naming, and temp cleanup.
7. Re-run:

```bash
python -m pytest test_b5b_dinov3_vector_contract.py test_b5b_compute_similarity_pairwise_optimized.py -q
```

Expected: pass.

8. Server command for DINOv3:

```bash
python B5b_compute_similarity_pairwise-optimized.py \
  --resolution 8 \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --source-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair \
  --input-template 'dinov3_city={city}_res_exclude={res_exclude}.parquet' \
  --feature-prefix e_ \
  --threshold -1.0 \
  --row-block-size 1000 \
  --memory-limit 16GB \
  --log-dir logs/dinov3_similarity
```

Expected: either temp shards exist during the run, or merged files exist after completion under `optimized/similarity_city=<City>_res=8_optimized.parquet`. Do not require temp shards after a successful run unless `--keep-temp` is implemented and passed.

## Task 5: Reuse or Lightly Extend Optimized Pairwise Aggregation

**Files:**

- Prefer Modify: `B5c_pairwise_agg_optimized.py`
- Create only if necessary: `B5g_pairwise_agg_dinov3.py`
- Modify/Create: `test_b5c_pairwise_agg_optimized.py` or `test_dinov3_pairwise_agg.py`

**Steps:**

1. Add a test with a DINOv3 temp shard tree under `c_city_dinov3_similarity_by_pair/optimized/temp/`.
2. Add a second test with only merged optimized files under `c_city_dinov3_similarity_by_pair/optimized/similarity_city=<City>_res=8_optimized.parquet`, because B5b deletes temp shards after a successful run.
3. Assert that the existing `--pairwise-root` and `--export-folder` options produce `similarity_intracity_city=<City>_res=8.parquet`, with inter-city rows only (`city_1 != city_2`).
4. If `metric` and `model_name` columns are present in shards, decide whether to preserve them. If the existing aggregation drops them and downstream summaries can add constants, do not modify B5c.
5. If preservation is needed, add optional pass-through aggregation for constant metadata columns while keeping current classifier outputs unchanged.
6. Run:

```bash
python -m pytest test_b5c_pairwise_agg_optimized.py -q
```

Expected: pass.

7. Server command:

```bash
python B5c_pairwise_agg_optimized.py \
  --resolution 8 \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair/optimized/_progress_res=8_optimized.json \
  --resume \
  --agg-progress-file /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_agg_progress_res=8.json \
  --duckdb-memory-limit 32GB \
  --duckdb-temp-dir /lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_dinov3_similarity \
  --duckdb-threads 8 \
  --parquet-file-size 512MB
```

## Task 6: Add City-Pair Summary Table

**Files:**

- Create: `B5h_summarize_dinov3_citypair_similarity.py`
- Create: `test_dinov3_citypair_summary.py`

**Steps:**

1. Write tests that read tiny per-city aggregated parquet outputs and produce one unordered row per city pair.
2. Run `python -m pytest test_dinov3_citypair_summary.py -q`; expected failure.
3. Implement DuckDB-based summary:
   - Read both single-file parquet and parquet dataset directories.
   - Normalize city-pair ordering with `LEAST(city_1, city_2)` and `GREATEST(city_1, city_2)`.
   - Compute average, p50, p90, p95, max, and observed pair count.
   - Use only inter-city rows and fail loudly if any `city_1 == city_2` rows remain.
4. Add `--expected-city-count` or infer it from `--city-meta` so the script can validate expected unordered pair count `n * (n - 1) / 2`.
5. Add `--output-format parquet|csv` if downstream modeling needs CSV.
6. Re-run `python -m pytest test_dinov3_citypair_summary.py -q`; expected pass.
7. Server command:

```bash
python B5h_summarize_dinov3_citypair_similarity.py \
  --input-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --output /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_summary_res=8.parquet
```

## Task 7: Documentation and Runbook

**Files:**

- Modify: `README.md`
- Optional Create: `docs/plans/2026-06-29-dinov3-server-runbook.md`

**Steps:**

1. Add a DINOv3 section after the existing B5 classifier pipeline notes.
2. Document the server run order: model/checkpoint verification, one-city embedding smoke test, all-city embeddings, H3 aggregation, pairwise cosine, optimized aggregation, city-pair summary.
3. Document that `--threshold -1.0` is recommended for the first DINOv3 production run if averages should not be threshold-biased.
4. Document validation checks:
   - one embedding dimension across cities,
   - no null or non-finite embeddings,
   - H3 vectors approximately unit norm,
   - nonzero res8 rows for every city,
   - completed pairwise progress JSON,
   - no same-city rows in final aggregated outputs,
   - expected unordered city-pair count in summary.

## Task 8: Full Verification Sequence

Run local tests first:

```bash
python -m pytest \
  test_dinov3_utils.py \
  test_dinov3_embed_city.py \
  test_dinov3_vector_summary.py \
  test_b5b_dinov3_vector_contract.py \
  test_b5b_compute_similarity_pairwise_optimized.py \
  test_b5c_pairwise_agg_optimized.py \
  test_dinov3_citypair_summary.py \
  -q
```

Then run server smoke tests in this order:

1. Verify DINOv3 checkpoint loading with a two-image batch.
2. Embed `Hong Kong` with `--limit 256`.
3. Aggregate Hong Kong embeddings to H3 with `--res-exclude 11`.
4. Build fake/tiny two-city H3 inputs or use two already embedded cities and run pairwise at `--resolution 8`.
5. Aggregate pairwise shards with `B5c_pairwise_agg_optimized.py`.
6. Create the city-pair summary.

## Data Flow Diagram

```text
image-list parquet
  path, name?
      |
      v
B5d_dinov3_embed_city.py
  load image -> DINOv3 backend -> L2-normalized e_* vectors
      |
      v
c_city_dinov3_embed/<cityabbr>/*.parquet
  name, panoid, model_name, embedding_dim, e_*
      |
      v
B5e_dinov3_vector_summary.py
  join pano metadata by panoid
  remove train/test leakage cells
  mean-pool by H3 cell
  L2-normalize H3 vectors
      |
      v
c_city_dinov3_hex_summary/dinov3_city=<City>_res_exclude=<res>.parquet
  hex_id, res, img_count, model_name, embedding_dim, e_*
      |
      v
B5b_compute_similarity_pairwise-optimized.py
  configurable input template + configurable vector columns
  blocked cosine pairwise
      |
      +--> optimized/temp/.../part_res=<res>.parquet      (during run / debugging)
      |
      v
optimized/similarity_city=<City>_res=<res>_optimized.parquet
      |
      v
B5c_pairwise_agg_optimized.py
  read temp shards if present, otherwise merged optimized files
  export inter-city rows only
      |
      v
c_city_dinov3_similarity_<date>/similarity_intracity_city=<City>_res=<res>.parquet
      |
      v
B5h_summarize_dinov3_citypair_similarity.py
  unordered city-pair summary statistics
      |
      v
c_city_dinov3_similarity_summary_res=<res>.parquet
```

## Acceptance Criteria

- Existing classifier B5 tests still pass with default CLI/config behavior.
- New DINOv3 unit tests pass without downloading model weights.
- One-city DINOv3 embedding smoke test writes valid image embeddings.
- One-city H3 aggregation smoke test writes valid res8 H3 vectors.
- Two-city pairwise smoke test writes nonempty DINOv3 temp shards during the run or nonempty merged optimized files after completion.
- Optimized aggregation writes per-city inter-city outputs under the DINOv3 root.
- City-pair summary table has one row per unordered city pair expected by downstream modeling.
- No existing classifier artifacts are overwritten.

## Failure Modes to Cover

| Codepath | Realistic failure | Test coverage required | Error handling required | User-visible result |
|----------|-------------------|------------------------|-------------------------|---------------------|
| DINOv3 backend loading | Server cannot resolve checkpoint, compute node has no internet, cached path is incomplete, or a connected node needs to download the verified model ID. | Unit test fake backend plus server two-image smoke command in local-only and download-on-miss modes. | CLI exits before processing any city; message names backend, model/path, and `local_files_only`. If download is intended, omit `--local-files-only` only for the smoke/cache-fill run. | Clear failure before expensive work; connected nodes can populate the cache deliberately. |
| Image embedding | One corrupt or missing image crashes a whole city batch. | Test with one bad path and one valid fake path. | Record skipped image count in sidecar/log; fail only if all rows fail unless `--strict-images` is passed. | Clear warning with row counts. |
| Resume logic | Existing shard has duplicate names or old embedding dimension. | Test finished-name collection and mixed-dimension shard rejection. | Refuse to resume from incompatible shards unless `--fresh-start` or explicit purge is used. | Clear error instead of mixed embeddings. |
| H3 aggregation | Pano metadata join drops every embedding because `panoid` parsing or city stem is wrong. | Fixture test for empty join. | Nonzero exit unless `--allow-empty`; sidecar JSON records `status="empty"`. | Clear failure before pairwise stage. |
| Train/test exclusion | Exclusion removes all rows for a city/resolution. | Fixture test where all rows share train/test H3 cell. | Same empty-output handling as above. | Clear failure or explicit allowed empty output. |
| B5b vector discovery | Cities have different embedding dimensions or missing `e_*` columns. | DINOv3 contract test with mismatched fake inputs. | Abort before pairwise compute and name offending city/file. | Clear failure before long run. |
| B5b thresholding | `--threshold 0.01` silently biases DINOv3 averages by dropping low, zero, or negative similarities. | CLI/default test confirms classifier default remains `0.01`; DINOv3 command uses `-1.0`. | Runbook documents `--threshold -1.0` for first DINOv3 production run. | Avoids biased summary by default in DINOv3 command. |
| B5c aggregation | Successful B5b run removed temp shards, so a temp-only B5c test gives false confidence. | Test both temp-shard and merged-file inputs. | Existing fallback should remain; do not require temp shards after completion. | Aggregation works after normal B5b completion. |
| City-pair summary | Duplicate per-city files double-count unordered city pairs. | Test two files containing the same city pair in opposite order. | Normalize ordering and group before statistics; validate expected pair count when `--city-meta` is provided. | One row per unordered city pair. |

Critical gap to close before implementation is considered done: B5b must validate embedding dimensions across all DINOv3 input files before starting pairwise computation. Without that, a late DuckDB/query or numpy failure could waste hours and produce partial progress artifacts.

## NOT in Scope

- Distributional metrics such as MMD or sliced Wasserstein.
- Approximate nearest-neighbor search.
- Land-use-stratified DINOv3 similarity via `B7_similarity_by_landuse.py`.
- Joint model table that merges classifier similarity and DINOv3 similarity with passenger-flow outcomes.
- Replacing the existing classifier-probability B5 pipeline.
- Renaming inherited `similarity_intracity...` B5c output files; the name is confusing, but changing it would ripple into existing downstream tooling.
- Proving that DINOv3 is the best possible foundation model; this plan only creates a frozen-feature robustness pipeline.

## Parallelization Strategy

Sequential implementation is safest for the shared core changes, but there are useful work lanes after Task 1 lands.

| Step | Modules touched | Depends on |
|------|-----------------|------------|
| Shared utilities | repo root utility/test files, requirements | - |
| Image embedding | repo root DINOv3 script/test | Shared utilities |
| H3 aggregation | repo root DINOv3 script/test, B5a-derived logic | Shared utilities |
| B5b parameterization | existing B5b script/test | Shared utilities |
| B5c reuse tests | existing B5c test/script if needed | B5b output contract |
| City-pair summary | repo root DINOv3 summary script/test | B5c output contract |
| Documentation/runbook | README, optional runbook | All command contracts |

Parallel lanes:

- Lane A: Shared utilities -> image embedding -> server checkpoint smoke.
- Lane B: Shared utilities -> H3 aggregation.
- Lane C: Shared utilities -> B5b parameterization -> B5c reuse tests.
- Lane D: City-pair summary, after the B5c output contract is fixed.
- Lane E: Documentation/runbook, after CLI names settle.

Execution order: implement Task 1 first in the main workspace. Then Lane A, Lane B, and Lane C can proceed in separate worktrees. Merge those before Lane D and Lane E.

Conflict flags: Lane B and Lane C should avoid editing shared helpers at the same time after Task 1. Lane C touches the existing optimized B5b script, so it should not run in parallel with unrelated B5b cleanup.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | - | Not run |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | - | Not run |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | UPDATED | 4 plan issues clarified, 1 critical implementation gap flagged |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | - | Not applicable, backend/data pipeline only |

- **UNRESOLVED:** DINOv3 checkpoint/backend name must be verified on the server before production embedding.
- **VERDICT:** ENG UPDATED, ready to implement after the checkpoint smoke gate is satisfied.
