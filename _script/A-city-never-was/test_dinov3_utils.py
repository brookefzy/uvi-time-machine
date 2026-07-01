from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dinov3_utils import (
    atomic_write_parquet,
    build_embedding_columns,
    discover_embedding_columns,
    embed_image_batch,
    l2_normalize_rows,
    resolve_city_file_stem,
    verify_embedding_backend,
)


def test_build_embedding_columns_uses_zero_padded_suffixes():
    assert build_embedding_columns(4) == ["e_0000", "e_0001", "e_0002", "e_0003"]


def test_discover_embedding_columns_sorts_numeric_suffixes_and_ignores_metadata():
    df = pd.DataFrame(
        columns=["name", "embedding_dim", "e_0010", "e_0002", "e_0001", "e_0000"]
    )

    assert discover_embedding_columns(df, strict=False) == [
        "e_0000",
        "e_0001",
        "e_0002",
        "e_0010",
    ]


def test_discover_embedding_columns_rejects_gaps_in_strict_mode():
    df = pd.DataFrame(columns=["e_0000", "e_0002"])

    with pytest.raises(ValueError, match="missing embedding columns"):
        discover_embedding_columns(df)


def test_l2_normalize_rows_preserves_zero_rows_and_normalizes_nonzero_rows():
    values = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=float)

    normalized = l2_normalize_rows(values)

    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(normalized[[0, 2]], axis=1), [1.0, 1.0])


def test_atomic_write_parquet_creates_parent_and_replaces_existing_file(tmp_path):
    output_path = tmp_path / "nested" / "embeddings.parquet"
    atomic_write_parquet(pd.DataFrame({"value": [1]}), output_path)
    atomic_write_parquet(pd.DataFrame({"value": [2]}), output_path)

    result = pd.read_parquet(output_path)

    assert result["value"].tolist() == [2]
    assert not list(output_path.parent.glob("*.tmp"))


def test_resolve_city_file_stem_normalizes_city_or_uses_override():
    assert resolve_city_file_stem("Hong Kong") == "hongkong"
    assert resolve_city_file_stem("São Paulo") == "saopaulo"
    assert resolve_city_file_stem("Hong Kong", override="hk_server") == "hk_server"


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModel:
    def __call__(self, **_kwargs):
        return type(
            "FakeOutput",
            (),
            {
                "pooler_output": FakeTensor(
                    [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]]
                )
            },
        )()


class FakeProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": images}


def test_embed_image_batch_uses_injected_model_and_returns_numpy(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"image_{i}.jpg"
        path.write_bytes(b"fake")
        paths.append(path)

    embeddings = embed_image_batch(
        paths,
        FakeModel(),
        FakeProcessor(),
        "cpu",
        image_loader=lambda _path: object(),
    )

    np.testing.assert_allclose(embeddings, [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]])


def test_verify_embedding_backend_reports_finite_unit_norm_embeddings(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"image_{i}.jpg"
        path.write_bytes(b"fake")
        paths.append(path)

    result = verify_embedding_backend(
        "fake-model",
        backend="fake",
        device="cpu",
        local_files_only=True,
        sample_paths=paths,
        backend_loader=lambda **_kwargs: (FakeModel(), FakeProcessor()),
        embedder=lambda sample_paths, _model, _processor, _device: np.array(
            [[1.0, 0.0], [0.0, 1.0]]
        ),
    )

    assert result == {
        "model_name": "fake-model",
        "embedding_dim": 2,
        "batch_size": 2,
        "device": "cpu",
        "finite": True,
        "unit_norm": True,
    }
