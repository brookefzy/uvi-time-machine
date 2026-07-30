import numpy as np
import pandas as pd
import pytest


def test_sample_limits_each_hex_to_fifty_in_name_order():
    from stage2_dino_modality.common import sample_per_hex

    frame = pd.DataFrame(
        {
            "hex_id": ["h1"] * 52 + ["h2"],
            "name": [f"image-{index:03d}" for index in range(52)] + ["only-image"],
        }
    ).sample(frac=1, random_state=4)

    sampled = sample_per_hex(frame, max_images_per_hex=50)

    assert sampled.groupby("hex_id").size().max() == 50
    assert sampled.loc[sampled["hex_id"] == "h1", "name"].tolist() == [
        f"image-{index:03d}" for index in range(50)
    ]


def test_validate_vectors_requires_contiguous_float32_finite_matrix():
    from stage2_dino_modality.common import validate_vectors

    vectors = validate_vectors(np.array([[1.0, 0.0]], dtype=np.float64), embedding_dim=2)
    assert vectors.dtype == np.float32
    assert vectors.flags.c_contiguous
    with pytest.raises(ValueError, match="embedding dimension"):
        validate_vectors(np.array([[1.0, 0.0]], dtype=np.float32), embedding_dim=3)
    with pytest.raises(ValueError, match="non-finite"):
        validate_vectors(np.array([[np.nan, 0.0]], dtype=np.float32), embedding_dim=2)


def test_l2_normalize_rows_rejects_zero_vectors_and_returns_unit_float32():
    from stage2_dino_modality.common import normalize_rows

    normalized = normalize_rows(np.array([[3.0, 4.0]], dtype=np.float32))
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[0.6, 0.8]])
    with pytest.raises(ValueError, match="zero"):
        normalize_rows(np.array([[0.0, 0.0]], dtype=np.float32))


def test_model_id_uses_canonical_centroid_payload_not_embedded_model_id():
    from stage2_dino_modality.common import build_model_id, centroid_checksum

    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    config = {"k": 2, "sample": 50, "columns": ["e_0000", "e_0001"]}

    checksum = centroid_checksum(centroids, config)
    assert checksum == centroid_checksum(centroids.copy(), {"columns": ["e_0000", "e_0001"], "sample": 50, "k": 2})
    assert build_model_id(checksum, config).startswith("k=2-")


def test_validate_sparse_histogram_requires_per_hex_unit_fractions():
    from stage2_dino_modality.common import validate_sparse_histogram

    valid = pd.DataFrame(
        {"city": ["Paris", "Paris"], "hex_id": ["h", "h"], "res": [8, 8], "mode_id": [0, 1], "mode_image_count": [2, 3], "sampled_image_count": [5, 5], "mode_fraction": [0.4, 0.6], "model_id": ["m", "m"]}
    )
    validate_sparse_histogram(valid)
    invalid = valid.copy()
    invalid.loc[1, "mode_fraction"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        validate_sparse_histogram(invalid)


def test_js_similarity_is_one_for_identical_histograms():
    from stage2_dino_modality.common import js_similarity

    assert js_similarity(np.array([0.4, 0.6]), np.array([0.4, 0.6])) == pytest.approx(1.0)


def test_js_similarity_is_zero_for_disjoint_histograms():
    from stage2_dino_modality.common import js_similarity

    assert js_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
