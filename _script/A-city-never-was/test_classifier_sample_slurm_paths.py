"""Contract tests for classifier-probability sampling SLURM jobs."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_JOB = REPO_ROOT / "slurm" / "classifier_sample_image_pairs.cmd"
GALLERY_JOB = REPO_ROOT / "slurm" / "classifier_build_sample_gallery.cmd"


def test_classifier_jobs_use_isolated_repo_and_python_contracts() -> None:
    for script in (SAMPLE_JOB, GALLERY_JOB):
        contents = script.read_text(encoding="utf-8")
        assert "#SBATCH --export=ALL" in contents
        assert (
            'REPO_DIR="${UVI_SAMPLE_REPO_DIR:-/lustre1/g/geog_pyloo/05_timemachine/'
            'uvi-time-machine/_script/A-city-never-was}"'
        ) in contents
        assert 'REPO_DIR="${REPO_DIR:-' not in contents
        assert 'REPO_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"' in contents
        assert 'VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"' in contents
        assert 'PYTHON="${VENV_PYTHON}"' in contents
        assert '-x "${PYTHON}"' in contents
        assert "mkdir -p logs/slurm sample_similar_pairs/output" in contents


def test_classifier_sample_job_passes_probability_and_selection_options() -> None:
    contents = SAMPLE_JOB.read_text(encoding="utf-8")

    assert "sample_classifier_image_pairs_faiss.py" in contents
    assert '"Paris|London"' in contents and '"Hong Kong|Singapore"' in contents
    assert '--probability-root "${CLASSIFIER_PROB_ROOT:-${ROOTFOLDER}/_curated/c_city_classifiier_prob}"' in contents
    assert '--image-index-root "${IMAGE_INDEX_ROOT:-${ROOTFOLDER}/_transformed/t_classifier_img_yolo8_inf_dir}"' in contents
    assert '--expected-dim "${CLASSIFIER_EXPECTED_DIM:-127}"' in contents
    assert '--vector-schema-id "${CLASSIFIER_SCHEMA_ID:-city-classifier-train4-probabilities-v1}"' in contents
    assert '--threshold "${CLASSIFIER_THRESHOLD:--1.0}"' in contents
    for variable in (
        "MAX_IMAGES_PER_H3",
        "MAX_IMAGES_PER_CITY",
        "TOP_K",
        "QUERY_BATCH_SIZE",
        "MAX_PAIRS_PER_SOURCE_IMAGE",
        "MAX_PAIRS_PER_HEX_PAIR",
        "MMR_CANDIDATE_POOL",
        "MMR_RELEVANCE_WEIGHT",
        "PAIRS_PER_CITY_PAIR",
    ):
        assert variable in contents
    assert "sample_similar_pairs/output/classifier_image_pairs.parquet" in contents


def test_classifier_gallery_job_uses_classifier_artifacts_and_labels() -> None:
    contents = GALLERY_JOB.read_text(encoding="utf-8")

    assert "sample_similar_pairs/output/classifier_image_pairs.parquet" in contents
    assert "sample_similar_pairs/output/classifier_image_gallery" in contents
    assert 'Classifier-probability cross-city similar-image samples' in contents
    assert 'Classifier-profile cosine similarity' in contents
