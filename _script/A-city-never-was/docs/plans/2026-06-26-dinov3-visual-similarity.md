# DINOv3 Visual Similarity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a DINOv3-based, label-free visual similarity pipeline that can be run on the remote server data and compared against the existing classifier-probability cosine similarity pipeline.

**Architecture:** Keep the current classifier pipeline intact as the baseline. Add a parallel DINOv3 branch that extracts frozen image embeddings by city, aggregates embeddings to H3 cells using the existing metadata and train/test H3 exclusion logic, then computes city-pair similarity from H3 embedding distributions. The first implementation should produce cosine nearest-neighbor and summary distribution metrics; Wasserstein/MMD can be added after the embedding and H3 outputs are validated.

**Tech Stack:** Python, PyTorch, Hugging Face `transformers` or `timm` depending on DINOv3 availability in the server environment, pandas, pyarrow/parquet, DuckDB, h3, scipy, scikit-learn, pytest.

## Reviewer Rationale and Sources

The current method uses 127-dimensional probabilities from a city classifier and cosine similarity between averaged H3 probability vectors. This is a useful baseline, but reviewers may argue that it measures resemblance in a city-label classifier space rather than a task-independent visual morphology space. The new DINOv3 branch should be framed as a robustness and construct-validity test: if the passenger-flow association holds with frozen, label-free visual foundation embeddings, the paper is less dependent on the original classifier design.

Primary source to cite in the method note/manuscript: DINOv3, Siméoni et al. 2025, `https://arxiv.org/abs/2508.10104`. The paper reports strong off-the-shelf visual features and dense representations across natural, aerial, and satellite imagery, which is closer to this task than older supervised city-label features. Use SigLIP 2, Tschannen et al. 2025, `https://arxiv.org/abs/2502.14786`, only as a secondary semantic robustness check after the DINOv3 pipeline is working.

## Existing Code to Reuse

Reference folder:

- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/stage1_classifier/B3_inference_city_prob.py`
- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/stage1_classifier/B3_inference_all-prod.py`
- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/stage1_classifier/classifier_hexagon_agg.py`
- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5a_prob_vector_summary.py`
- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5b_compute_similarity_pairwise-optimized.py`
- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py`
- `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/utils/urban_utils.py`

Important existing conventions:

- Remote data root is `/lustre1/g/geog_pyloo/05_timemachine`.
- Image-list parquet input is currently `/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir/<city>.parquet` with a `path` column.
- Classifier probability output is chunked by city under `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob/<city>/`.
- H3 summary outputs currently use `prob_city=<City>_res_exclude=<res>.parquet` under `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary`.
- Existing H3 aggregation joins prediction rows to panorama metadata by `panoid`, computes H3 resolutions `[6, 7, 8]`, and excludes samples that fall in train/test H3 cells at exclusion resolutions `[11, 12, 13]`.
- Existing pairwise similarity uses checkpointed temp shards under an `optimized/temp/city1=<city>/city2=<city>/part_res=<res>.parquet` tree and merges them to per-city parquet outputs.

## New Output Contract

Use separate folders so no old classifier artifacts are overwritten:

- Image-level DINOv3 embeddings:
  `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed/<city>/<city>_<timestamp>.parquet`
- H3-level DINOv3 embeddings:
  `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary/dinov3_city=<City>_res_exclude=<res>.parquet`
- Pairwise DINOv3 H3 similarity shards:
  `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair/optimized/temp/city1=<City>/city2=<City>/part_res=<res>.parquet`
- Aggregated inter-city outputs:
  `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_<YYYYMMDD>/similarity_intracity_city=<City>_res=<res>.parquet`

Recommended columns:

- Image-level output: `name`, `panoid`, `model_name`, `embedding_dim`, `e_0000` ... `e_<dim-1>`.
- H3 output: `hex_id`, `res`, `img_count`, `model_name`, `embedding_dim`, `e_0000` ... `e_<dim-1>`.
- Pairwise output: `hex_id1`, `hex_id2`, `city1`, `city2`, `similarity`, `metric`, `model_name`.

## Assumptions to Confirm Before Running on Server

1. The remote server has GPU access and can install or already has compatible PyTorch, torchvision, transformers/timm, and pyarrow.
2. DINOv3 weights can be downloaded once to a server cache, or pre-downloaded and referenced with `--model-path` if the compute node has no internet. Default verified Hugging Face card: `facebook/dinov3-vitb16-pretrain-lvd1689m` at `https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`. The card is manually gated, so accept the license and authenticate with `huggingface-cli login` or `HF_TOKEN` before downloading. If the model is not found on disk and the node is connected, run the smoke test without `--local-files-only` so the verified model ID downloads directly into cache; then rerun production jobs with `--local-files-only`.
3. Existing city names in `city_meta.csv` should remain title-cased, while image-list files use lowercase no-space city abbreviations, matching the current scripts.
4. H3 resolution 8 and exclusion resolution 11 should be the first production run, because that is the current downstream comparison setting.
5. The first deliverable is cosine-based DINOv3 similarity. Distributional metrics should be implemented only after image-level and H3-level embedding files pass validation.

## Task 1: Add Lightweight Model and Image Utilities

**Files:**

- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/dinov3_utils.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_utils.py`
- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/requirements.txt`

**Step 1: Write failing tests for feature-column naming and L2 normalization**

Create tests that do not download DINOv3. Test only local helper behavior:

```python
import numpy as np

from dinov3_utils import build_embedding_columns, l2_normalize_rows


def test_build_embedding_columns_zero_padded():
    assert build_embedding_columns(3) == ["e_0000", "e_0001", "e_0002"]


def test_l2_normalize_rows_handles_zero_rows():
    arr = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = l2_normalize_rows(arr)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])
```

**Step 2: Run tests and verify failure**

Run from `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was`:

```bash
python -m pytest tests/test_dinov3_utils.py -q
```

Expected: fail because `dinov3_utils.py` does not exist.

**Step 3: Implement minimal utility functions**

Implement:

```python
def build_embedding_columns(dim: int) -> list[str]:
    return [f"e_{idx:04d}" for idx in range(dim)]


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)
```

Add image loading/model-loading helpers in the same file, but keep them thin and injectable so tests can use fake processors/models without downloading weights:

- `load_rgb_image(path: str) -> PIL.Image.Image`
- `load_dinov3_model(model_name: str, device: str, local_files_only: bool = False)`
- `embed_image_batch(paths, model, processor, device) -> np.ndarray`

**Step 4: Add dependencies**

Add only the required runtime dependencies:

```text
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.45.0
Pillow>=10.0.0
```

If the selected DINOv3 release is easier through `timm`, add `timm>=1.0.0` and document which backend is active in the script CLI help.

**Step 5: Re-run tests**

```bash
python -m pytest tests/test_dinov3_utils.py -q
```

Expected: pass.

## Task 2: Add Per-City DINOv3 Image Embedding Script

**Files:**

- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5d_dinov3_embed_city.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_embed_city.py`

**Step 1: Write a fixture-based resume test**

Use a temporary input parquet with columns `path` and optional `name`. Use a fake embedder returning deterministic vectors. Verify that when one output shard already contains `name=a.jpg`, the task list skips `a.jpg` and embeds only the remaining rows.

Expected public functions:

- `load_city_image_index(valfolder: str, city: str) -> pd.DataFrame`
- `collect_finished_names(curated_city_folder: str) -> set[str]`
- `build_pending_batches(df: pd.DataFrame, finished_names: set[str], chunk_size: int) -> list[pd.DataFrame]`

**Step 2: Implement the script by mirroring `B3_inference_city_prob.py`**

Required CLI:

```bash
python B5d_dinov3_embed_city.py \
  --city "Hong Kong" \
  --valfolder /lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir \
  --output-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --batch-size 64 \
  --device cuda \
  --local-files-only
```

Implementation requirements:

- Convert city display name to file stem with `city.lower().replace(" ", "")` unless `--city-file-stem` is provided.
- Read `<valfolder>/<city_file_stem>.parquet`.
- Preserve current chunk/resume behavior from `B3_inference_city_prob.py`.
- Save each chunk atomically using a temporary parquet path and `Path.replace`.
- Include `name`, `panoid`, `model_name`, `embedding_dim`, and embedding columns.
- Derive `panoid` as `name[:22]`, matching `B5a_prob_vector_summary.py`.
- L2-normalize image embeddings before writing so cosine similarity is dot-product compatible downstream.

**Step 3: Test without loading the real model**

```bash
python -m pytest tests/test_dinov3_embed_city.py -q
```

Expected: pass with fake embedder and tiny parquet fixture.

**Step 4: Server smoke command**

On the remote server, run one small city first:

```bash
python B5d_dinov3_embed_city.py \
  --city "Hong Kong" \
  --batch-size 32 \
  --device cuda \
  --limit 256 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m
```

Expected: one parquet shard appears under `c_city_dinov3_embed/hongkong/`, with 256 or fewer rows and no null embeddings.

## Task 3: Add DINOv3 H3 Aggregation Script

**Files:**

- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5e_dinov3_vector_summary.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_vector_summary.py`

**Step 1: Write fixture test for aggregation and exclusion**

Create small fake inputs:

- Image embeddings with `name`, `panoid`, `e_0000`, `e_0001`.
- Panorama metadata with `panoid`, `lat`, `lon`.
- Training panoid set that shares an exclusion-resolution H3 cell with one prediction row.

Expected behavior:

- Aggregated output has columns `hex_id`, `res`, `img_count`, `e_0000`, `e_0001`.
- The excluded row is absent when `res_exclude=11`.
- H3 vectors are L2-normalized after averaging.

**Step 2: Implement by adapting `H3HexagonAggregator` from `B5a_prob_vector_summary.py`**

Rename/generalize only what is needed:

- Replace fixed `self.vector_columns = [str(x) for x in range(127)]` with discovered columns matching `e_` prefix.
- Replace `CURATED_FOLDER` with `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed`.
- Replace export folder with `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary`.
- Keep `summary_resolutions = [6, 7, 8]`.
- Keep `exclude_resolutions = [11, 12, 13]`.
- Keep metadata loading through `PANO_PATH` and `PATH_PATH`.
- Add `img_count` to each grouped H3 row.

**Step 3: Add validation stats**

Write a sidecar JSON for each city/exclusion output:

```json
{
  "city": "Hong Kong",
  "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
  "embedding_dim": 1024,
  "total_images_after_exclusion": 12345,
  "total_hexagons_res8": 678,
  "mean_img_count_res8": 18.2
}
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_dinov3_vector_summary.py -q
```

Expected: pass.

**Step 5: Server smoke command**

```bash
python B5e_dinov3_vector_summary.py \
  --city "Hong Kong" \
  --res-exclude 11 \
  --log-level INFO
```

Expected: `dinov3_city=Hong Kong_res_exclude=11.parquet` exists and contains rows for `res in [6, 7, 8]`.

## Task 4: Add DINOv3 Pairwise Cosine Similarity

**Files:**

- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5f_compute_similarity_dinov3_pairwise.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_pairwise_similarity.py`

**Step 1: Write tiny two-city similarity test**

Create two fake H3 summary parquet files:

- City A: two H3 rows with simple normalized vectors.
- City B: two H3 rows with simple normalized vectors.

Assert that:

- The script reads `dinov3_city=<City>_res_exclude=11.parquet`.
- It filters to `res=8`.
- It outputs cross-city rows only.
- Similarity equals cosine dot product for normalized vectors.
- `metric == "cosine"` and `model_name` is preserved.

**Step 2: Implement by adapting `B5b_compute_similarity_pairwise-optimized.py`**

Keep the current checkpointing and temp shard pattern. Change only the feature contract:

- Source folder: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary`.
- Export folder: `/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair`.
- Feature columns: all columns matching `e_` prefix.
- Source filename: `dinov3_city={city}_res_exclude={RES_EXCLUDE}.parquet`.
- Output rows include `metric="cosine"` and `model_name`.

Do not add approximate nearest-neighbor search in this first version. Use the same blocked exact cosine approach first so results are directly comparable with the classifier baseline.

**Step 3: Add CLI**

```bash
python B5f_compute_similarity_dinov3_pairwise.py \
  --resolution 8 \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --row-block-size 1000 \
  --threshold -1.0 \
  --memory-limit 16GB \
  --log-dir logs/dinov3_similarity
```

Use `--threshold -1.0` for the first production run to avoid changing the meaning of averages. Add thresholding only after zero-fill aggregation is verified.

**Step 4: Run tests**

```bash
python -m pytest tests/test_dinov3_pairwise_similarity.py -q
```

Expected: pass.

## Task 5: Reuse Aggregation or Add Thin DINOv3 Aggregator Wrapper

**Files:**

- Prefer modifying only if needed: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5c_pairwise_agg_optimized.py`
- Otherwise create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5g_pairwise_agg_dinov3.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_pairwise_agg.py`

**Step 1: Decide whether the existing aggregator already works**

Run with a tiny temp shard tree from Task 4. If `B5c_pairwise_agg_optimized.py` passes through extra columns poorly, create a DINOv3 wrapper. If it ignores extra columns but preserves required fields, reuse it with CLI parameters only.

**Step 2: Test inter-city aggregation**

Assert that output:

- Removes same-city pairs.
- Deduplicates unordered H3 pairs.
- Writes `similarity_intracity_city=<City>_res=8.parquet` for compatibility.
- Preserves or can reconstruct `metric="cosine"` and `model_name` in downstream summaries.

**Step 3: Server aggregation command**

```bash
python B5c_pairwise_agg_optimized.py \
  --resolution 8 \
  --city-meta /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --resume \
  --duckdb-memory-limit 16GB \
  --duckdb-temp-dir /lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_dinov3_similarity \
  --duckdb-threads 8
```

Expected: per-city inter-city parquet outputs in the DINOv3 export folder.

## Task 6: Add City-Pair Summary Metrics for Modeling

**Files:**

- Create: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5h_summarize_dinov3_citypair_similarity.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_citypair_summary.py`

**Step 1: Write summary test**

Given small inter-city parquet rows, assert summary columns:

- `city_1`
- `city_2`
- `dinov3_cosine_avg`
- `dinov3_cosine_p50`
- `dinov3_cosine_p90`
- `dinov3_cosine_p95`
- `dinov3_cosine_max`
- `dinov3_pair_count_observed`

Do not make `max` the primary statistic. For the manuscript, prioritize `avg`, `p90/p95`, and observed/expected coverage.

**Step 2: Implement with DuckDB**

Read all per-city outputs from the DINOv3 aggregation folder and summarize by unordered city pair. Use DuckDB quantile functions where possible.

**Step 3: Add expected pair counts later**

After the first cosine run is validated, add expected cross-city H3-pair counts from the H3 summary files so the model table can include zero-fill-normalized averages. Keep this as a second pass to avoid coupling too many changes to the first implementation.

**Step 4: Run tests**

```bash
python -m pytest tests/test_dinov3_citypair_summary.py -q
```

Expected: pass.

## Task 7: Production Run Order on Remote Server

**Files:** no new files.

**Step 1: Install/update environment on server**

From the script directory on the server:

```bash
python -m pip install -r requirements.txt
```

If model download is blocked on compute nodes, download DINOv3 weights on a login node or connected environment, then run with `--local-files-only` and the correct cache/model path. If model download is allowed and the checkpoint is not on disk, omit `--local-files-only` for the first two-image smoke test so the backend downloads `facebook/dinov3-vitb16-pretrain-lvd1689m` directly.

```bash
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"

huggingface-cli download "${MODEL_NAME}" \
  --include config.json preprocessor_config.json model.safetensors
```

**Step 2: Run one-city embedding smoke test**

```bash
python B5d_dinov3_embed_city.py --city "Hong Kong" --limit 256 --batch-size 32 --device cuda
```

Verify output row count, embedding dimension, and null count.

**Step 3: Run all-city embeddings**

Use the scheduler style appropriate for the server. One city per job is safest because the script is resumable by city folder.

```bash
python B5d_dinov3_embed_city.py --city "<City>" --batch-size 64 --device cuda --local-files-only
```

**Step 4: Run H3 aggregation for all cities**

```bash
python B5e_dinov3_vector_summary.py --city all --res-exclude 11 --resume
```

**Step 5: Run exact pairwise cosine similarity at H3 resolution 8**

```bash
python B5f_compute_similarity_dinov3_pairwise.py --resolution 8 --threshold -1.0 --row-block-size 1000 --memory-limit 16GB
```

**Step 6: Aggregate pairwise shards**

```bash
python B5c_pairwise_agg_optimized.py \
  --resolution 8 \
  --pairwise-root /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_by_pair \
  --export-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --resume
```

**Step 7: Summarize for downstream modeling**

```bash
python B5h_summarize_dinov3_citypair_similarity.py \
  --input-folder /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_$(date +%Y%m%d) \
  --output /lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_similarity_summary_res=8.parquet
```

## Task 8: Validation Checklist Before Using in Paper Models

**Files:** no new files unless validation failures require fixes.

Run these checks and save results in the run log:

1. Image-level embeddings exist for every city in `city_meta.csv`.
2. Each city has nonzero H3 rows at `res=8` after `res_exclude=11`.
3. Embedding dimensions are constant across all files.
4. Image and H3 embeddings have finite values only.
5. H3 vectors are approximately L2-normalized after aggregation.
6. Pairwise progress JSON reports `status = completed`.
7. Aggregated outputs contain no same-city rows.
8. Summary table has one row per unordered city pair expected by the downstream passenger-flow model.
9. Correlation between classifier-based similarity and DINOv3 similarity is reported but not assumed to be high.
10. Passenger-flow model is re-run with classifier similarity, DINOv3 similarity, and both together.

## Task 9: Optional Second-Pass Distributional Metrics

Only start this after Tasks 1-8 pass.

**Files:**

- Modify: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/B5h_summarize_dinov3_citypair_similarity.py`
- Test: `/Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/tests/test_dinov3_distributional_metrics.py`

Add city-pair distribution distances from H3 embedding sets:

- MMD with an RBF kernel over H3 embeddings.
- Sliced Wasserstein distance over random projections.
- Optional energy distance if runtime is acceptable.

Do not compute full optimal transport first; it may be too expensive at all-city H3 scale. Sliced Wasserstein is a more practical first distributional metric.

## Final Acceptance Criteria

The implementation is complete when:

- Unit tests pass for utility, embedding resume behavior, H3 aggregation, pairwise similarity, aggregation, and city-pair summary.
- A one-city server smoke test produces valid DINOv3 image embeddings.
- A one-city H3 aggregation smoke test produces valid `res=8` H3 vectors.
- A two-city server smoke test produces nonempty pairwise cosine shards.
- Full production run has completed progress files and city-pair summaries.
- Downstream modeling can compare the existing classifier-based similarity to DINOv3-based similarity without changing old classifier outputs.
