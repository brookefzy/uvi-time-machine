from pathlib import Path


ROOT = Path(__file__).parents[1] / "slurm"
JOBS = ["dinov3_mode_sample_array.cmd", "dinov3_mode_fit_codebooks.cmd", "dinov3_mode_gallery.cmd", "dinov3_mode_select.cmd", "dinov3_mode_assign_array.cmd", "dinov3_mode_histogram_array.cmd", "dinov3_mode_similarity_array.cmd", "dinov3_mode_city_summary.cmd"]


def test_mode_jobs_use_self_contained_remote_defaults_and_fixed_stage_scripts():
    for name in JOBS:
        text = (ROOT / name).read_text()
        assert '#SBATCH --partition=amd' in text
        assert '#SBATCH --export=ALL' in text
        assert '/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python' in text
        assert '/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was' in text
        assert 'if [[ ! -x "${PYTHON}" ]]' in text


def test_submitters_are_bounded_and_wait_between_batches():
    for name in ("submit_dinov3_mode_city_batches.bash", "submit_dinov3_mode_similarity_batches.bash"):
        text = (ROOT / name).read_text()
        assert 'BATCH_SIZE="${BATCH_SIZE:-20}"' in text
        assert 'while squeue -h -j' in text


def test_city_arrays_cover_zero_based_csv_data_rows():
    coordinator = (ROOT / "run_dinov3_mode_pipeline.bash").read_text()
    submitter = (ROOT / "submit_dinov3_mode_city_batches.bash").read_text()

    assert 'FIRST_CITY="${FIRST_CITY:-0}"' in submitter
    assert '$(wc -l < "${CITY_META}") - 2' in coordinator


def test_similarity_batches_use_local_array_indices_with_manifest_offsets():
    submitter = (ROOT / "submit_dinov3_mode_similarity_batches.bash").read_text()
    worker = (ROOT / "dinov3_mode_similarity_array.cmd").read_text()

    assert 'batch_count=$((end-start+1))' in submitter
    assert 'pair_index_offset=$((start-1))' in submitter
    assert '--array="1-${batch_count}%${ARRAY_CONCURRENCY}"' in submitter
    assert '--export="ALL,PAIR_INDEX_OFFSET=${pair_index_offset}"' in submitter
    assert 'PAIR_INDEX_OFFSET="${PAIR_INDEX_OFFSET:-0}"' in worker
    assert 'PAIR_INDEX=$((PAIR_INDEX_OFFSET + SLURM_ARRAY_TASK_ID))' in worker
    assert '"${PAIR_MANIFEST}" "${PAIR_INDEX}"' in worker


def test_similarity_worker_reuses_complete_shards_when_resuming():
    worker = (ROOT / "dinov3_mode_similarity_array.cmd").read_text()

    assert 'OUTPUT=' in worker
    assert '[[ "${RESUME:-1}" == "1" && -f "${OUTPUT}" ]]' in worker
    assert '--output "${OUTPUT}"' in worker


def test_downstream_arrays_resolve_inputs_from_manifest_and_selected_model():
    assign = (ROOT / "dinov3_mode_assign_array.cmd").read_text()
    histogram = (ROOT / "dinov3_mode_histogram_array.cmd").read_text()
    similarity = (ROOT / "dinov3_mode_similarity_array.cmd").read_text()
    for text in (assign, histogram):
        assert 'CITY_META' in text
        assert 'SLURM_ARRAY_TASK_ID' in text
        assert 'SELECTED_MODEL' in text
    assert 'PAIR_MANIFEST' in similarity
    assert 'SLURM_ARRAY_TASK_ID' in similarity
    assert 'h3_similarity' in similarity


def test_coordinator_waits_for_each_downstream_stage_and_passes_cli_paths():
    text = (ROOT / "run_dinov3_mode_pipeline.bash").read_text()
    assert 'REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was}"' in text
    assert 'MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT:-${ROOTFOLDER}/_curated/c_city_dinov3_global_modes/res=8/sample=50}"' in text
    assert 'export SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:--1}"' in text
    assert 'LANDUSE_TIERS_PATH="${LANDUSE_TIERS_PATH:-${ROOTFOLDER}/_curated/03_similarity_grid/landuse_tiers/h3_landuse_tiers_pct10_res=8.csv}"' in text
    assert 'MODE_MAX_TRAINING_IMAGES_PER_CITY="${MODE_MAX_TRAINING_IMAGES_PER_CITY:-2000}"' in text
    assert 'MODE_MAX_TRAINING_IMAGES_PER_H3="${MODE_MAX_TRAINING_IMAGES_PER_H3:-5}"' in text
    assert 'MODE_STRATUM_WEIGHTS="${MODE_STRATUM_WEIGHTS:-core=.4,suburban=.3,occupied_rural=.2,no_poi=.1}"' in text
    assert 'MODE_K_VALUES="${MODE_K_VALUES:-32 64 128}"' in text
    assert 'MODE_GALLERY_MEMORY="${MODE_GALLERY_MEMORY:-128G}"' in text
    assert 'sbatch --wait --mem="${MODE_GALLERY_MEMORY}" slurm/dinov3_mode_gallery.cmd' in text
    assert '--k "${MODE_K_ARGS[@]}"' in text
    assert '--landuse-tiers "${LANDUSE_TIERS_PATH}"' in text
    assert '--max-training-images-per-city "${MODE_MAX_TRAINING_IMAGES_PER_CITY}"' in text
    assert '--max-training-images-per-h3 "${MODE_MAX_TRAINING_IMAGES_PER_H3}"' in text
    assert '--stratum-weights "${MODE_STRATUM_WEIGHTS}"' in text
    assert 'training_pool_audit.json' in text
    assert 'Use a new MODE_OUTPUT_ROOT' in text
    assert 'sbatch --wait' in text
    assert 'dinov3_mode_assign_array.cmd' in text
    assert 'dinov3_mode_histogram_array.cmd' in text
    assert 'generate_dinov3_mode_pair_manifest.py' in text
    assert 'dinov3_mode_similarity_array.cmd' in text
    assert 'dinov3_mode_city_summary.cmd' in text
    assert 'ALLOW_MISSING_CITIES' in text
    assert '--allow-missing' in text
    assert '--expected-model-id "${MODEL_ID}"' in text
    assert '--pair-manifest' in text
    assert 'Accepting existing histogram subset because ALLOW_MISSING_CITIES=1' in text
    assert '"${ALLOW_MISSING_CITIES:-0}" == "1" && "${RESUME:-1}" == "1"' in text
    assert 'if [[ "${RESUME:-1}" != "1" || ! -f "${MODE_OUTPUT_ROOT}/model=${MODEL_ID}/city_pair_summary.parquet" ]]' not in text
