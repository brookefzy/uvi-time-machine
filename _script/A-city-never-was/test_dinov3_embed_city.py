from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from B5d_dinov3_embed_city import collect_finished_names, embed_city


class FakeBackend:
    model = object()
    processor = object()

    def embed(self, paths, _model, _processor, _device):
        embeddings = []
        for path in paths:
            index = int(Path(path).stem.rsplit("_", 1)[-1])
            embeddings.append([float(index + 1), float(index + 2)])
        return np.asarray(embeddings, dtype=float)


def make_input(tmp_path, city_stem="hongkong", count=5):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for i in range(count):
        name = f"panoid_{i:02d}_abcdefghijkl_{i}.jpg"
        path = image_root / name
        path.write_bytes(b"fake")
        rows.append({"path": str(path)})
    valfolder = tmp_path / "val"
    valfolder.mkdir()
    pd.DataFrame(rows).to_parquet(valfolder / f"{city_stem}.parquet", index=False)
    return valfolder, rows


def test_embed_city_writes_l2_normalized_chunks_with_expected_columns(tmp_path):
    valfolder, rows = make_input(tmp_path, count=3)
    output_root = tmp_path / "out"
    backend = FakeBackend()

    written = embed_city(
        city="Hong Kong",
        city_file_stem=None,
        valfolder=valfolder,
        output_root=output_root,
        model_name="fake-dinov3",
        backend_name="fake",
        batch_size=2,
        device="cpu",
        local_files_only=True,
        limit=None,
        backend_loader=lambda **_kwargs: (backend.model, backend.processor),
        embedder=backend.embed,
    )

    assert len(written) == 2
    result = pd.concat(pd.read_parquet(path) for path in written).sort_values("name")
    embedding_cols = ["e_0000", "e_0001"]

    assert result.columns.tolist() == [
        "name",
        "panoid",
        "model_name",
        "embedding_dim",
        *embedding_cols,
    ]
    assert result["name"].tolist() == [Path(row["path"]).name for row in rows]
    assert result["panoid"].tolist() == [Path(row["path"]).name[:22] for row in rows]
    assert result["model_name"].unique().tolist() == ["fake-dinov3"]
    assert result["embedding_dim"].unique().tolist() == [2]
    np.testing.assert_allclose(
        np.linalg.norm(result[embedding_cols].to_numpy(dtype=float), axis=1),
        np.ones(3),
    )


def test_embed_city_resumes_by_skipping_names_in_existing_chunks(tmp_path):
    valfolder, rows = make_input(tmp_path, count=4)
    output_root = tmp_path / "out"
    city_output = output_root / "hongkong"
    city_output.mkdir(parents=True)
    existing_name = Path(rows[0]["path"]).name
    pd.DataFrame(
        {
            "name": [existing_name],
            "panoid": [existing_name[:22]],
            "model_name": ["fake-dinov3"],
            "embedding_dim": [2],
            "e_0000": [1.0],
            "e_0001": [0.0],
        }
    ).to_parquet(city_output / "hongkong_existing.parquet", index=False)
    backend = FakeBackend()

    written = embed_city(
        city="Hong Kong",
        city_file_stem="hongkong",
        valfolder=valfolder,
        output_root=output_root,
        model_name="fake-dinov3",
        backend_name="fake",
        batch_size=10,
        device="cpu",
        local_files_only=True,
        limit=None,
        backend_loader=lambda **_kwargs: (backend.model, backend.processor),
        embedder=backend.embed,
    )

    new_rows = pd.concat(pd.read_parquet(path) for path in written)

    assert existing_name not in new_rows["name"].tolist()
    assert sorted(new_rows["name"].tolist()) == sorted(
        Path(row["path"]).name for row in rows[1:]
    )


def test_resume_rejects_existing_shards_with_mixed_embedding_dimensions(tmp_path):
    output_dir = tmp_path / "out" / "hongkong"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "name": ["panoid_a.jpg", "panoid_b.jpg"],
            "panoid": ["panoid_a", "panoid_b"],
            "model_name": ["fake-dinov3", "fake-dinov3"],
            "embedding_dim": [2, 3],
            "e_0000": [1.0, 1.0],
            "e_0001": [0.0, 0.0],
        }
    ).to_parquet(output_dir / "bad_existing.parquet", index=False)

    with pytest.raises(ValueError, match="embedding_dim"):
        collect_finished_names(output_dir, expected_model_name="fake-dinov3")


def test_resume_rejects_duplicate_names_across_existing_shards(tmp_path):
    output_dir = tmp_path / "out" / "hongkong"
    output_dir.mkdir(parents=True)
    duplicate_name = "panoid_a.jpg"
    existing = {
        "name": [duplicate_name],
        "panoid": ["panoid_a"],
        "model_name": ["fake-dinov3"],
        "embedding_dim": [2],
        "e_0000": [1.0],
        "e_0001": [0.0],
    }
    pd.DataFrame(existing).to_parquet(output_dir / "existing_a.parquet", index=False)
    pd.DataFrame(existing).to_parquet(output_dir / "existing_b.parquet", index=False)

    with pytest.raises(ValueError, match="duplicate names"):
        collect_finished_names(output_dir, expected_model_name="fake-dinov3")


def test_embed_city_limit_applies_after_resume_filter(tmp_path):
    valfolder, _rows = make_input(tmp_path, count=5)
    output_root = tmp_path / "out"
    backend = FakeBackend()

    written = embed_city(
        city="Hong Kong",
        city_file_stem=None,
        valfolder=valfolder,
        output_root=output_root,
        model_name="fake-dinov3",
        backend_name="fake",
        batch_size=10,
        device="cpu",
        local_files_only=True,
        limit=2,
        backend_loader=lambda **_kwargs: (backend.model, backend.processor),
        embedder=backend.embed,
    )

    result = pd.concat(pd.read_parquet(path) for path in written)

    assert len(result) == 2
