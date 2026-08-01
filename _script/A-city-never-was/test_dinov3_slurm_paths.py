from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parent
H3_ARRAY_SCRIPT = REPO_ROOT / "slurm" / "dinov3_02_h3_array.cmd"
SAMPLE_SLURM_SCRIPTS = [
    REPO_ROOT / "slurm" / "dinov3_sample_image_pairs.cmd",
    REPO_ROOT / "slurm" / "dinov3_sample_h3_pairs.cmd",
    REPO_ROOT / "slurm" / "dinov3_build_sample_gallery.cmd",
]
CITY_TEMPLATE_SCRIPTS = [
    REPO_ROOT / "slurm" / "dinov3_sample_h3_pairs.cmd",
    REPO_ROOT / "slurm" / "dinov3_03_pairwise_array.cmd",
    REPO_ROOT / "slurm" / "submit_dinov3_pairwise_batches.bash",
]
PAIRWISE_DIRECT_RUNNER = REPO_ROOT / "pipeline" / "run_dinov3_pairwise_direct.bash"
PAIRWISE_RESUME_RUNNER = REPO_ROOT / "pipeline" / "run_dinov3_pairwise_resume.bash"
PAIRWISE_RESUME_SUBMITTER = REPO_ROOT / "slurm" / "submit_dinov3_pairwise_resume_batches.bash"
PIPELINE_INDEX = REPO_ROOT / "pipeline" / "INDEX.md"


def test_h3_array_rewrites_legacy_city_meta_path(tmp_path: Path) -> None:
    repo_dir = tmp_path / "_script" / "A-city-never-was"
    repo_dir.mkdir(parents=True)
    captured_args = tmp_path / "python-args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${CAPTURED_ARGS}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURED_ARGS": str(captured_args),
            "CITY_META": str(repo_dir / "city_meta.csv"),
            "PYTHON": str(fake_python),
            "REPO_DIR": str(repo_dir),
            "SLURM_ARRAY_TASK_ID": "1",
        }
    )

    subprocess.run(["bash", str(H3_ARRAY_SCRIPT)], env=env, check=True)

    args = captured_args.read_text(encoding="utf-8").splitlines()
    city_meta_index = args.index("--city-meta") + 1
    assert args[city_meta_index] == str(repo_dir.parent / "city_meta.csv")


def test_sample_jobs_prioritize_explicit_venv_over_inherited_python() -> None:
    for script in SAMPLE_SLURM_SCRIPTS:
        contents = script.read_text(encoding="utf-8")
        assert "#SBATCH --export=ALL" in contents
        assert 'REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"' in contents
        assert 'VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"' in contents
        assert 'PYTHON="${VENV_PYTHON}"' in contents
        assert '-x "${PYTHON}"' in contents


def test_sample_jobs_ignore_unrelated_repo_dir_environment_variable() -> None:
    for script in SAMPLE_SLURM_SCRIPTS:
        contents = script.read_text(encoding="utf-8")
        assert (
            'REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/'
            'uvi-time-machine/_script/A-city-never-was}"'
        ) in contents
        assert 'REPO_DIR="${REPO_DIR:-' not in contents


def test_image_sample_job_enables_mmr_scene_diversity_by_default() -> None:
    contents = (REPO_ROOT / "slurm" / "dinov3_sample_image_pairs.cmd").read_text(encoding="utf-8")
    assert '--top-k "${TOP_K:-30}"' in contents
    assert '--max-pairs-per-source-image "${MAX_PAIRS_PER_SOURCE_IMAGE:-1}"' in contents
    assert '--max-pairs-per-hex-pair "${MAX_PAIRS_PER_HEX_PAIR:-1}"' in contents
    assert '--mmr-candidate-pool "${MMR_CANDIDATE_POOL:-200}"' in contents
    assert '--mmr-relevance-weight "${MMR_RELEVANCE_WEIGHT:-0.7}"' in contents


def test_pairwise_submitter_uses_local_array_indices_after_slurm_limit() -> None:
    submitter = (REPO_ROOT / "slurm" / "submit_dinov3_pairwise_batches.bash").read_text(encoding="utf-8")
    worker = (REPO_ROOT / "slurm" / "dinov3_03_pairwise_array.cmd").read_text(encoding="utf-8")

    assert 'task_count=$((end - start + 1))' in submitter
    assert 'PAIR_OFFSET=$((start - 1))' in submitter
    assert '--array="1-${task_count}%${ARRAY_CONCURRENCY}"' in submitter
    assert 'PAIR_LINE_NUMBER=$((PAIR_OFFSET + SLURM_ARRAY_TASK_ID))' in worker
    assert 'sed -n "${PAIR_LINE_NUMBER}p" "${PAIR_MANIFEST}"' in worker


def test_city_input_template_defaults_do_not_break_bash_braces() -> None:
    for script in CITY_TEMPLATE_SCRIPTS:
        contents = script.read_text(encoding="utf-8")
        assert 'INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"' in contents or 'H3_INPUT_TEMPLATE="${H3_INPUT_TEMPLATE:-}"' in contents
        assert "dinov3_city={city}_res_exclude=None.parquet" in contents


def test_direct_pairwise_runners_provide_fresh_and_resume_safe_modes() -> None:
    direct = PAIRWISE_DIRECT_RUNNER.read_text(encoding="utf-8")
    resume = PAIRWISE_RESUME_RUNNER.read_text(encoding="utf-8")
    index = PIPELINE_INDEX.read_text(encoding="utf-8")

    assert "generate_dinov3_pair_manifest.py" in direct
    assert "B5b_compute_similarity_pairwise-optimized.py" in direct
    assert '--resolution "${RESOLUTION}"' in direct
    assert '"${OUTPUT_ROOT}"' in direct
    assert 'INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"' in direct
    assert "INPUT_TEMPLATE='dinov3_city={city}_res_exclude=None.parquet'" in direct

    assert 'INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"' in resume
    assert "INPUT_TEMPLATE='dinov3_city={city}_res_exclude=None.parquet'" in resume
    assert 'SHARD="${OUTPUT_ROOT}/optimized/temp/city1=${CITY1}/city2=${CITY2}/part_res=${RESOLUTION}.parquet"' in resume
    assert '[[ -s "${SHARD}" ]]' in resume
    assert "DONE — skipping" in resume

    assert "run_dinov3_pairwise_direct.bash" in index
    assert "run_dinov3_pairwise_resume.bash" in index


def test_pairwise_resume_submitter_skips_nonempty_shards_and_batches_pending_pairs() -> None:
    submitter = PAIRWISE_RESUME_SUBMITTER.read_text(encoding="utf-8")

    assert "generate_dinov3_pair_manifest.py" in submitter
    assert 'PENDING_MANIFEST="${PAIR_MANIFEST%.txt}_pending.txt"' in submitter
    assert 'SHARD="${OUTPUT_ROOT}/optimized/temp/city1=${CITY1}/city2=${CITY2}/part_res=${RESOLUTION}.parquet"' in submitter
    assert '[[ -s "${SHARD}" ]]' in submitter
    assert 'printf \'Previously complete: %s; pending: %s\\n\'' in submitter
    assert 'PAIR_OFFSET=$((start - 1))' in submitter
    assert '--array="1-${task_count}%${ARRAY_CONCURRENCY}"' in submitter
