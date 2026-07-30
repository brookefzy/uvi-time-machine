# Global DINOv3 Mode Distribution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Build one globally comparable DINOv3 visual-mode vocabulary, represent every eligible resolution-8 H3 cell as a sparse distribution over those modes, and compute Jensen–Shannon H3 similarity across city pairs.

**Architecture:** Reuse the existing DINOv3 image embeddings and eligibility/H3 logic to make a deterministic up-to-50-image sample for every occupied resolution-8 cell. Fit several global spherical FAISS k-means codebooks on a balanced subset of that pool, evaluate `K ∈ {64, 128, 256, 512}`, render representative-image galleries for review, then select one model. The selected model assigns the same sampled images to globally shared mode IDs; assignments are grouped into sparse H3 histograms and compared in bounded blocks using Jensen–Shannon distance/similarity.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyArrow/Parquet, FAISS CPU, scikit-learn metrics, existing `sample_similar_pairs.common`, existing `dinov3_utils`, Bash/Slurm.

## Decisions fixed by this plan

- **Comparison population:** eligible 2012–2022 images at resolution 8, using the same metadata join and optional exclusion semantics as `sample_similar_pairs.common.attach_image_geography`.
- **Sampling:** deterministic lexical selection of up to 50 images per H3 cell. The selected images are the only images assigned to modes, so H3 distributions are comparable rather than driven by capture density.
- **Mode space:** one codebook shared by all cities. Mode IDs are meaningful only together with a saved `model_id`; they must never be compared across codebook versions. Compute the artifact checksum from canonical centroid payload bytes *before* adding `model_id`: little-endian contiguous float32 centroid matrix ordered by `mode_id`, plus UTF-8 canonical JSON (`sort_keys=True`, compact separators) for the immutable training/sampling configuration and ordered embedding-column names. `model_id` is then the hash of that checksum and the same canonical configuration; it is written into the centroid table only after construction, avoiding a circular checksum.
- **K selection:** calculate held-out assignment cohesion, seed stability, mode-support diagnostics, and galleries for all four K values. Write an automatic recommendation, but allow `--selected-k` to override it after visual review. Purely numerical cohesion rises with K, so it is not sufficient by itself.
- **Jensen–Shannon outputs:** store both `js_distance = sqrt(JSD_base2)` and `js_similarity = 1 - js_distance` in `[0, 1]`. The latter is the requested similarity; keeping the distance prevents ambiguity.
- **Output root:** `${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50/model=<model_id>/`. `model_id` incorporates K, source embedding identity, sampling/year/exclusion settings, and the codebook artifact checksum.
- **Scalability:** city arrays submit in batches and wait between batches. Global fit, K selection, and gallery construction are single jobs. Exact H3×H3 work follows `B5b_compute_similarity_pairwise-optimized.py`: one unordered city pair per independent array task, bounded computation blocks, explicit threshold filtering, and atomic pair shards. There is no artificial output-pair cap; operators choose a threshold or `--threshold -1` to retain all similarities.

## Output contracts

```text
.../res=8/sample=50/
  sampled_images/city=<city>.parquet
  image_index/city=<city>.parquet
  codebook_candidates/k=<K>/centroids.parquet
  codebook_candidates/k=<K>/metrics.json
  codebook_candidates/scorecard.parquet
  codebook_candidates/recommended_model.json
  mode_gallery/k=<K>/index.html
  model=<model_id>/assignments/city=<city>.parquet
  model=<model_id>/h3_histograms/city=<city>.parquet
  model=<model_id>/h3_similarity/city_1=<city>/city_2=<city>.parquet
  model=<model_id>/city_pair_summary.parquet
```

`sampled_images` columns: `city`, `hex_id`, `res`, `name`, `panoid`, `lat`, `lon`, `embedding_dim`, `model_name`, and the existing `e_####` columns.

`image_index` is an explicit gallery input root, passed as `--image-index-root`; each `city=<city>.parquet` contains `path` and optionally `name` (derived from `path` when absent). Gallery construction joins it to `sampled_images` on `(city, name)` and treats absent images as an auditable skip.

`centroids` columns: `model_id`, `k`, `mode_id`, `embedding_dim`, `training_image_count`, `e_####` columns. Centroids are L2-normalized.

`assignments` columns: `city`, `hex_id`, `res`, `name`, `panoid`, `mode_id`, `assignment_cosine`, `model_id`.

`h3_histograms` is long/sparse: `city`, `hex_id`, `res`, `mode_id`, `mode_image_count`, `sampled_image_count`, `mode_fraction`, `model_id`. Fractions sum to one for every `(city, hex_id, model_id)`.

## Task 1: Create the stage package and shared data-contract helpers

**Files:**
- Create: `stage2_dino_modality/__init__.py`
- Create: `stage2_dino_modality/common.py`
- Create: `stage2_dino_modality/test_common.py`

**Step 1: Write failing tests**

Test deterministic per-H3 selection, vector schema validation, L2 normalization, model-ID construction, sparse-histogram validation, and `js_similarity = 1 - sqrt(JSD_base2)` for known two-mode distributions.

```python
def test_js_similarity_is_one_for_identical_histograms():
    assert js_similarity(np.array([0.4, 0.6]), np.array([0.4, 0.6])) == pytest.approx(1.0)

def test_sample_limits_each_hex_to_fifty_in_name_order():
    sampled = sample_per_hex(frame, max_images_per_hex=50)
    assert sampled.groupby("hex_id").size().max() == 50
```

**Step 2: Run the focused test and verify failure**

Run: `python3 -m pytest stage2_dino_modality/test_common.py -q`

Expected: import/function failure before implementation.

**Step 3: Implement minimal helpers**

Import and reuse `CityVectors`, `load_city_embeddings`, `attach_image_geography`, and `spatially_sample_city` from `sample_similar_pairs.common`; do not duplicate their source-file, year, exclusion, or H3 API handling. Add only stage-specific helpers: atomic JSON/Parquet audit writing, contiguous float32 validation, canonical centroid-payload and model-ID hashing (with the byte ordering and JSON serialization defined above), `require_faiss()`, Jensen–Shannon block calculation, and sparse-histogram checks.

**Step 4: Verify green**

Run: `python3 -m pytest stage2_dino_modality/test_common.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add stage2_dino_modality
git commit -m "feat: add global DINO mode shared helpers"
```

## Task 2: Sample up to 50 images per resolution-8 H3 cell

**Files:**
- Create: `stage2_dino_modality/01_sample_h3_images.py`
- Create: `stage2_dino_modality/test_sample_h3_images.py`

**Step 1: Write failing tests**

Use tiny sharded city embeddings and panorama metadata. Assert that only eligible images remain, each H3 has at most 50 rows, selection is repeatable after shuffled input, and an audit reports before/joined/sampled row counts and undersupplied H3 count.

**Step 2: Run red test**

Run: `python3 -m pytest stage2_dino_modality/test_sample_h3_images.py -q`

**Step 3: Implement CLI**

Implement `--city`, `--embedding-root`, `--rootfolder`, `--train-test-folder`, `--res-exclude`, `--min-year`, `--max-year`, `--h3-resolution` (default `8`), `--max-images-per-h3` (default `50`), and `--output`. Call `load_city_embeddings`, `attach_image_geography`, and `spatially_sample_city`; write one city Parquet plus JSON audit using `write_parquet_with_json_audit` semantics. Preserve vectors because the next stage needs them.

**Step 4: Run green test**

Run: `python3 -m pytest stage2_dino_modality/test_sample_h3_images.py -q`

**Step 5: Commit**

```bash
git add stage2_dino_modality
git commit -m "feat: sample balanced H3 images for global modes"
```

## Task 3: Fit and evaluate candidate global FAISS codebooks

**Files:**
- Create: `stage2_dino_modality/02_fit_evaluate_codebooks.py`
- Create: `stage2_dino_modality/test_fit_evaluate_codebooks.py`

**Step 1: Write failing tests**

Test that normalized two-dimensional synthetic clusters yield normalized centroids, every requested valid K emits a centroid table and metric record, an invalid K greater than training rows emits an `invalid` scorecard record while valid candidates continue, and seed-label stability uses adjusted Rand score (which is invariant to cluster-label permutations). The command exits nonzero only when no requested K is valid.

**Step 2: Run red test**

Run: `python3 -m pytest stage2_dino_modality/test_fit_evaluate_codebooks.py -q`

**Step 3: Implement fit/evaluation**

- Read the sampled-image Parquet dataset with PyArrow batches; validate one embedding schema and deduplicate `(city, name)`.
- Form a deterministic, **city-balanced** FAISS training subset from the up-to-50-per-H3 pool. Expose `--max-training-images-per-city`; default it conservatively (for example `100000`) so London/Tokyo cannot define the global vocabulary. This is a second-stage cap, not a replacement for the required 50-per-H3 sampling output.
- Hold out a deterministic 20% of that balanced subset for metrics; train each K on the remaining 80% with at least two fixed seeds, CPU FAISS `Kmeans(..., spherical=True)`, and configurable `--niter`, `--max-points-per-centroid`, and `--seed`.
- For each K, save the primary-seed centroids; calculate held-out mean/p05 assignment cosine (cohesion), mean ARI between two seed assignments (stability), min/median mode share, number of near-empty modes, training/holdout counts, and runtime.
- Write a scorecard with one row per K, including `status` and `error` fields. Keep all K artifacts and unsuccessful diagnostics; do not silently discard a candidate. A K greater than available training rows is a recorded invalid candidate, not a reason to discard valid requested K values.

**Step 4: Run green tests**

Run: `python3 -m pytest stage2_dino_modality/test_fit_evaluate_codebooks.py -q`

**Step 5: Commit**

```bash
git add stage2_dino_modality
git commit -m "feat: fit and evaluate global FAISS mode codebooks"
```

## Task 4: Build mode galleries and select a model version

**Files:**
- Create: `stage2_dino_modality/03_build_mode_gallery.py`
- Create: `stage2_dino_modality/04_select_mode_model.py`
- Create: `stage2_dino_modality/test_mode_gallery.py`
- Create: `stage2_dino_modality/test_select_mode_model.py`

**Step 1: Write failing tests**

Test that each mode gallery contains the requested number of distinct nearest assigned images, shows mode ID/size/cohesion/city/H3/assignment cosine, and copies image files into a portable `images/` directory. Test that selection rejects unsupported K and recommends the smallest K meeting configurable stability/support thresholds and a held-out-cohesion elbow rule.

**Step 2: Run red tests**

Run: `python3 -m pytest stage2_dino_modality/test_mode_gallery.py stage2_dino_modality/test_select_mode_model.py -q`

**Step 3: Implement gallery and selection**

- Require `--image-index-root` and reuse the image-index convention from `sample_similar_pairs/build_image_pair_gallery.py`: it contains `city=<city>.parquet` files with `path` (and optionally `name`; derive name from path when absent). Join on `(city, name)`, report unmatched sampled representatives in the gallery audit, and do not copy all sampled images—copy only the top `--images-per-mode` representatives, default 20.
- For each K, assign a streaming representative pool to its centroids and retain the highest assignment-cosine images per mode. Render a self-contained HTML page grouped by mode with thumbnails, image/city/H3 metadata, assignment cosine, mode share, and a link to the image location map. Follow the existing gallery’s Leaflet/OpenStreetMap layout.
- `04_select_mode_model.py` reads the scorecard and writes `recommended_model.json`. Default rule: reject K below `--min-stability` (default 0.90) or with too many modes below `--min-mode-share`; among valid K, select the smallest K for which the held-out cohesion gain over the previous K is below `--cohesion-gain-epsilon` (default 0.005). If no K reaches an elbow, choose the valid K maximizing `stability * held_out_mean_cohesion`; record the rule and all metrics.
- Add `--selected-k` to explicitly override the recommendation after visual gallery review; write `selected_model.json` and copy/reference the immutable centroid artifact. Assignment jobs must require this selected-model file rather than infer a K.

**Step 4: Run green tests**

Run: `python3 -m pytest stage2_dino_modality/test_mode_gallery.py stage2_dino_modality/test_select_mode_model.py -q`

**Step 5: Commit**

```bash
git add stage2_dino_modality
git commit -m "feat: add global mode galleries and model selection"
```

## Task 5: Assign sampled images and build sparse H3 histograms

**Files:**
- Create: `stage2_dino_modality/05_assign_images_to_modes.py`
- Create: `stage2_dino_modality/06_build_h3_mode_histograms.py`
- Create: `stage2_dino_modality/test_assign_images_to_modes.py`
- Create: `stage2_dino_modality/test_build_h3_mode_histograms.py`

**Step 1: Write failing tests**

Test exact nearest-centroid assignments on unit vectors; verify one assignment per sampled image; verify histogram counts and fractions; reject an assignment model ID different from the selected model; reject H3 groups with zero/incorrectly normalized fractions.

**Step 2: Run red tests**

Run: `python3 -m pytest stage2_dino_modality/test_assign_images_to_modes.py stage2_dino_modality/test_build_h3_mode_histograms.py -q`

**Step 3: Implement both CLIs**

- Assignment reads one city’s sampled-image vectors and `selected_model.json`, builds `faiss.IndexFlatIP` from normalized selected centroids, and searches with batches. Output only metadata, assigned `mode_id`, and `assignment_cosine`; never duplicate full image vectors.
- Histogram construction reads one assignment file, groups by `(city, hex_id, res, mode_id)`, calculates counts and fractions, and writes a long sparse Parquet file with an audit. It validates that every H3 has 1–50 assigned sampled images and fractions sum to one.

**Step 4: Run green tests**

Run: `python3 -m pytest stage2_dino_modality/test_assign_images_to_modes.py stage2_dino_modality/test_build_h3_mode_histograms.py -q`

**Step 5: Commit**

```bash
git add stage2_dino_modality
git commit -m "feat: assign global modes and build sparse H3 distributions"
```

## Task 6: Compute blocked cross-city H3 Jensen–Shannon similarity

**Files:**
- Create: `stage2_dino_modality/07_compute_h3_mode_js_similarity.py`
- Create: `stage2_dino_modality/08_summarize_mode_citypairs.py`
- Create: `stage2_dino_modality/test_compute_h3_mode_js_similarity.py`
- Create: `stage2_dino_modality/test_summarize_mode_citypairs.py`

**Step 1: Write failing tests**

Test identical, disjoint, and partial-overlap histograms against hand-calculated base-2 Jensen–Shannon values. Test blocked execution matches a simple pairwise reference, no same-city pair is accepted, output includes both distance and similarity, and the summary has one unordered row per city pair.

**Step 2: Run red tests**

Run: `python3 -m pytest stage2_dino_modality/test_compute_h3_mode_js_similarity.py stage2_dino_modality/test_summarize_mode_citypairs.py -q`

**Step 3: Implement pairwise and summary CLIs**

- Input two city sparse-histogram files for one selected model. Densify only the two city matrices to `float32` shape `(n_hex, K)`; reject mismatched model IDs or incomplete mode ranges.
- Calculate JSD in source/target blocks, masking zero terms in `p * log2(p / m)`. Default `--row-block-size 64` and `--target-block-size 2048`; expose both so memory can be tuned without changing results.
- Follow `B5b_compute_similarity_pairwise-optimized.py`: accept exactly one `--city-pair CITY1|CITY2` per independent Slurm array task, process exact cross-city results in bounded source/target blocks, and atomically write a single pair shard beneath `h3_similarity/city_1=<city_1>/city_2=<city_2>/part_res=8.parquet`. Support `--threshold` as an explicit filtering option and `--threshold -1` to retain every result; never silently truncate or introduce a separate output-pair cap. Retain all source and target H3 rows even when their `hex_id` values coincide.
- Write `city_1`, `hex_id_1`, `city_2`, `hex_id_2`, `model_id`, `js_distance`, `js_similarity`, and a generic `similarity` alias for compatibility with existing downstream conventions.
- Summarize to `js_similarity_avg`, `p50`, `p90`, `p95`, `max`, and `pair_count_observed`; validate expected unordered city-pair count from `city_meta.csv` just as `B5h_summarize_dinov3_citypair_similarity.py` does.

**Step 4: Run green tests**

Run: `python3 -m pytest stage2_dino_modality/test_compute_h3_mode_js_similarity.py stage2_dino_modality/test_summarize_mode_citypairs.py -q`

**Step 5: Commit**

```bash
git add stage2_dino_modality
git commit -m "feat: compare H3 global-mode distributions with Jensen-Shannon similarity"
```

## Task 7: Add Slurm jobs and bounded stage submitters

**Files:**
- Create: `slurm/dinov3_mode_sample_array.cmd`
- Create: `slurm/dinov3_mode_fit_codebooks.cmd`
- Create: `slurm/dinov3_mode_gallery.cmd`
- Create: `slurm/dinov3_mode_select.cmd`
- Create: `slurm/dinov3_mode_assign_array.cmd`
- Create: `slurm/dinov3_mode_histogram_array.cmd`
- Create: `slurm/dinov3_mode_similarity_array.cmd`
- Create: `slurm/dinov3_mode_city_summary.cmd`
- Create: `slurm/submit_dinov3_mode_city_batches.bash`
- Create: `slurm/submit_dinov3_mode_similarity_batches.bash`
- Create: `slurm/generate_dinov3_mode_pair_manifest.py`
- Create: `slurm/run_dinov3_mode_pipeline.bash`
- Modify: `test_dinov3_slurm_paths.py`
- Create: `stage2_dino_modality/test_slurm_contracts.py`

**Step 1: Write failing tests**

Assert every mode job resolves the project’s `${REPO_ROOT}/.venv/bin/python` through `VENV_PYTHON`, uses the fixed stage script path, validates executability, uses the same `city_meta.csv` location convention, and writes only beneath configured mode-output roots. Assert submitters use bounded arrays and wait for each submitted batch before the next, with conservative defaults (`BATCH_SIZE=20`, `ARRAY_CONCURRENCY=2` for assignment/similarity; no array for global jobs).

**Step 2: Run red tests**

Run: `python3 -m pytest test_dinov3_slurm_paths.py stage2_dino_modality/test_slurm_contracts.py -q`

**Step 3: Implement Slurm layer**

- Follow `slurm/dinov3_sample_image_pairs.cmd` for interpreter resolution and `slurm/submit_dinov3_h3_batches.bash` / `slurm/submit_dinov3_pairwise_batches.bash` for capped submission-and-wait behavior.
- Sample, assignment, and histogram use city-index arrays over `city_meta.csv`. The sample/assignment/histogram submitter may accept `STAGE=sample|assign|histogram`, but must keep each task’s `.cmd` directly runnable for debugging.
- Fit/evaluate codebooks, gallery, selection, and summary are one global job each, avoiding avoidable submitted-job slots.
- Generate the mode pair manifest only after all selected-model histograms exist; validate `res=8`, `model_id`, and nonempty sparse histogram rows before adding a city.
- The similarity submitter sends bounded pair arrays and waits between batches, default `ARRAY_CONCURRENCY=2`, following the account-cap pattern already in the repository.
- `run_dinov3_mode_pipeline.bash` orchestrates sample → fit → gallery → selection → assignment → histogram → pair manifest → similarity → summary. It must stop after gallery/automatic recommendation unless `SELECTED_K` is supplied, so visual confirmation is possible before expensive downstream work.

**Step 4: Run green tests and syntax checks**

Run:

```bash
python3 -m pytest test_dinov3_slurm_paths.py stage2_dino_modality/test_slurm_contracts.py -q
for f in slurm/dinov3_mode_*.cmd slurm/submit_dinov3_mode_*.bash slurm/run_dinov3_mode_pipeline.bash; do bash -n "$f"; done
```

Expected: PASS and no Bash syntax errors.

**Step 5: Commit**

```bash
git add slurm stage2_dino_modality test_dinov3_slurm_paths.py
git commit -m "feat: add bounded Slurm workflow for global DINO modes"
```

## Task 8: End-to-end synthetic smoke test and operator documentation

**Files:**
- Create: `stage2_dino_modality/test_end_to_end_smoke.py`
- Create: `stage2_dino_modality/README.md`

**Step 1: Write failing smoke test**

Build two tiny cities with two H3 cells and obvious normalized two-dimensional modes. Exercise sampling, codebook fit, selection, assignment, sparse histograms, exact pairwise JSD, summary, and gallery manifest generation in a temporary directory.

**Step 2: Run red test**

Run: `python3 -m pytest stage2_dino_modality/test_end_to_end_smoke.py -q`

**Step 3: Implement README and any minimal wiring**

Document remote defaults, output layout, K-selection diagnostics, how to inspect the HTML gallery, how to override `SELECTED_K`, how to resume each stage, the exact Slurm commands, and why `MAX_OUTPUT_PAIRS` must be set consciously for full H3 cross-products.

**Step 4: Run final verification**

Run:

```bash
python3 -m pytest stage2_dino_modality test_dinov3_slurm_paths.py -q
git diff --check
```

Expected: all tests pass and no whitespace errors.

**Step 5: Commit**

```bash
git add stage2_dino_modality slurm test_dinov3_slurm_paths.py
git commit -m "docs: document global DINO mode workflow"
```

## Operator sequence after implementation

```bash
# 1. Bounded sample arrays across cities.
bash slurm/run_dinov3_mode_pipeline.bash

# 2. Review codebook-candidate galleries and scorecard, then choose a K.
SELECTED_K=128 bash slurm/run_dinov3_mode_pipeline.bash
```

The coordinator resumes completed immutable artifacts and must never overwrite a different `model_id`. All commands accept `ROOTFOLDER`, `UVI_SAMPLE_REPO_DIR`, `VENV_PYTHON`, `CITY_META`, and output-root overrides for the remote cluster.
