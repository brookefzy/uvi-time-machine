from pathlib import Path


ROOT = Path(__file__).parents[1] / "slurm"
JOBS = ["dinov3_mode_sample_array.cmd", "dinov3_mode_fit_codebooks.cmd", "dinov3_mode_gallery.cmd", "dinov3_mode_select.cmd", "dinov3_mode_assign_array.cmd", "dinov3_mode_histogram_array.cmd", "dinov3_mode_similarity_array.cmd", "dinov3_mode_city_summary.cmd"]


def test_mode_jobs_use_configured_venv_and_fixed_stage_scripts():
    for name in JOBS:
        text = (ROOT / name).read_text()
        assert 'VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python' in text
        assert 'if [[ ! -x "${PYTHON}" ]]' in text


def test_submitters_are_bounded_and_wait_between_batches():
    for name in ("submit_dinov3_mode_city_batches.bash", "submit_dinov3_mode_similarity_batches.bash"):
        text = (ROOT / name).read_text()
        assert 'BATCH_SIZE="${BATCH_SIZE:-20}"' in text
        assert 'while squeue -h -j' in text
