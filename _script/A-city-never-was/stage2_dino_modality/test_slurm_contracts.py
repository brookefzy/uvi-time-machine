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
    assert 'sbatch --wait' in text
    assert 'dinov3_mode_assign_array.cmd' in text
    assert 'dinov3_mode_histogram_array.cmd' in text
    assert 'generate_dinov3_mode_pair_manifest.py' in text
    assert 'dinov3_mode_similarity_array.cmd' in text
    assert 'dinov3_mode_city_summary.cmd' in text
