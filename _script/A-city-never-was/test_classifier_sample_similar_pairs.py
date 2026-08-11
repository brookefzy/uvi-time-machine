"""Tests for classifier-probability similar-image sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class _FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


class _FakeIndexIDMap2:
    def __init__(self, _base: _FakeIndexFlatIP) -> None:
        self.vectors: np.ndarray | None = None
        self.ids: np.ndarray | None = None

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.vectors is not None and self.ids is not None
        scores = np.asarray(queries) @ self.vectors.T
        positions = np.argsort(-scores, axis=1)[:, :top_k]
        return np.take_along_axis(scores, positions, axis=1), self.ids[positions]


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP
    IndexIDMap2 = _FakeIndexIDMap2


def _probability_rows(names: list[str], vectors: list[list[float]]) -> list[dict[str, object]]:
    return [
        {
            **{str(index): value for index, value in enumerate(vector)},
            "name": name,
        }
        for name, vector in zip(names, vectors)
    ]


def _write_city_shard(
    root: Path,
    city_stem: str,
    filename: str,
    rows: list[dict[str, object]],
) -> Path:
    city_dir = root / city_stem
    city_dir.mkdir(parents=True, exist_ok=True)
    path = city_dir / filename
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_pano_metadata(root: Path, city_stem: str, panoid: str, lat: float, lon: float) -> None:
    metadata_dir = root / "GSV" / "gsv_rgb" / city_stem / "gsvmeta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"panoid": panoid, "year": 2020, "lat": lat, "lon": lon}]
    ).to_csv(metadata_dir / "gsv_pano.csv", index=False)
    pd.DataFrame([{"panoid": panoid}]).to_csv(metadata_dir / "gsv_path.csv", index=False)


def test_load_city_classifier_probabilities_discovers_normalizes_and_audits(tmp_path: Path) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    names = [
        "abcdefghijklmnopqrstuv_0.jpg",
        "zyxwvutsrqponmlkjihgfe_90.jpg",
    ]
    root = tmp_path / "probabilities"
    _write_city_shard(root, "hongkong", "hongkong_002.parquet", _probability_rows(names[1:], [[0.0, 3.0, 4.0]]))
    _write_city_shard(root, "hongkong", "hongkong_001.parquet", _probability_rows(names[:1], [[2.0, 0.0, 0.0]]))

    loaded, stats = load_city_classifier_probabilities(
        root,
        "Hong Kong",
        expected_dim=3,
        return_stats=True,
    )

    assert loaded.city == "Hong Kong"
    assert loaded.vector_columns == ["prob_000", "prob_001", "prob_002"]
    assert loaded.metadata["name"].tolist() == names
    assert loaded.metadata["panoid"].tolist() == [name[:22] for name in names]
    assert loaded.vectors.dtype == np.float32
    assert loaded.vectors.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0)
    assert stats == {
        "input_shards": 2,
        "input_rows": 2,
        "probability_sum_min": 2.0,
        "probability_sum_mean": 4.5,
        "probability_sum_max": 7.0,
    }


def test_classifier_probability_loader_requires_city_shards(tmp_path: Path) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    with pytest.raises(FileNotFoundError, match="Paris.*paris"):
        load_city_classifier_probabilities(tmp_path, "Paris", expected_dim=3)


def test_classifier_probability_loader_rejects_missing_name(tmp_path: Path) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    _write_city_shard(tmp_path, "paris", "part.parquet", [{"0": 1.0, "1": 0.0, "2": 0.0}])
    with pytest.raises(ValueError, match="Paris.*name"):
        load_city_classifier_probabilities(tmp_path, "Paris", expected_dim=3)


def test_classifier_probability_loader_rejects_null_and_duplicate_names(tmp_path: Path) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    _write_city_shard(tmp_path, "paris", "null.parquet", _probability_rows([None], [[1.0, 0.0, 0.0]]))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Paris.*null names"):
        load_city_classifier_probabilities(tmp_path, "Paris", expected_dim=3)

    duplicate_root = tmp_path / "duplicate"
    rows = _probability_rows(["abcdefghijklmnopqrstuv_0.jpg"], [[1.0, 0.0, 0.0]])
    _write_city_shard(duplicate_root, "paris", "a.parquet", rows)
    _write_city_shard(duplicate_root, "paris", "b.parquet", rows)
    with pytest.raises(ValueError, match="Paris.*duplicate names"):
        load_city_classifier_probabilities(duplicate_root, "Paris", expected_dim=3)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"0": 1.0, "1": 0.0, "name": "abcdefghijklmnopqrstuv_0.jpg"}, "missing.*2"),
        (
            {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0, "name": "abcdefghijklmnopqrstuv_0.jpg"},
            "unexpected.*3",
        ),
    ],
)
def test_classifier_probability_loader_rejects_wrong_dimensions(
    tmp_path: Path,
    columns: dict[str, object],
    message: str,
) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    _write_city_shard(tmp_path, "paris", "part.parquet", [columns])
    with pytest.raises(ValueError, match=rf"Paris.*{message}"):
        load_city_classifier_probabilities(tmp_path, "Paris", expected_dim=3)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([np.nan, 0.0, 1.0], "non-finite"),
        ([-0.1, 0.1, 1.0], "negative"),
        ([0.0, 0.0, 0.0], "zero probability mass"),
    ],
)
def test_classifier_probability_loader_rejects_invalid_values(
    tmp_path: Path,
    vector: list[float],
    message: str,
) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    _write_city_shard(
        tmp_path,
        "paris",
        "part.parquet",
        _probability_rows(["abcdefghijklmnopqrstuv_0.jpg"], [vector]),
    )
    with pytest.raises(ValueError, match=rf"Paris.*{message}"):
        load_city_classifier_probabilities(tmp_path, "Paris", expected_dim=3)


def test_classifier_probability_loader_rejects_inconsistent_shard_schemas(tmp_path: Path) -> None:
    from sample_similar_pairs.common import load_city_classifier_probabilities

    _write_city_shard(
        tmp_path,
        "paris",
        "a.parquet",
        _probability_rows(["abcdefghijklmnopqrstuv_0.jpg"], [[1.0, 0.0, 0.0]]),
    )
    _write_city_shard(
        tmp_path,
        "paris",
        "b.parquet",
        [{"0": 1.0, "1": 0.0, "3": 0.0, "name": "zyxwvutsrqponmlkjihgfe_0.jpg"}],
    )

    with pytest.raises(ValueError, match="Paris.*inconsistent probability columns"):
        load_city_classifier_probabilities(tmp_path, "Paris", expected_dim=3)


def test_classifier_image_sampler_parser_defaults() -> None:
    from sample_similar_pairs.sample_classifier_image_pairs_faiss import build_parser

    args = build_parser().parse_args(["--output", "pairs.parquet"])

    assert str(args.probability_root).endswith("/_curated/c_city_classifiier_prob")
    assert args.expected_dim == 127
    assert args.vector_schema_id == "city-classifier-train4-probabilities-v1"
    assert args.threshold == -1.0
    assert args.max_images_per_h3 == 100
    assert args.max_images_per_city == 0
    assert args.top_k == 30
    assert args.max_pairs_per_source_image == 1
    assert args.max_pairs_per_hex_pair == 1
    assert args.mmr_candidate_pool == 200
    assert args.mmr_relevance_weight == 0.7
    assert args.pairs_per_city_pair == 10
    assert args.city_pairs == [
        "Paris|London",
        "London|Hong Kong",
        "Hong Kong|Singapore",
        "London|Sydney",
        "New York|London",
    ]


def test_classifier_image_sampler_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from sample_similar_pairs import image_pair_pipeline
    from sample_similar_pairs.sample_classifier_image_pairs_faiss import main

    probability_root = tmp_path / "probabilities"
    city_inputs = {
        "paris": ("abcdefghijklmnopqrstuv", [1.0, 0.0], 48.8566, 2.3522),
        "london": ("zyxwvutsrqponmlkjihgfe", [1.0, 0.0], 51.5072, -0.1276),
    }
    for city_stem, (panoid, vector, lat, lon) in city_inputs.items():
        _write_city_shard(
            probability_root,
            city_stem,
            f"{city_stem}_001.parquet",
            _probability_rows([f"{panoid}_0.jpg"], [vector]),
        )
        _write_pano_metadata(tmp_path, city_stem, panoid, lat, lon)
    monkeypatch.setattr(image_pair_pipeline, "_faiss", lambda: _FakeFaiss)
    output = tmp_path / "classifier_pairs.parquet"

    main(
        [
            "--city-pairs", "Paris|London",
            "--probability-root", str(probability_root),
            "--expected-dim", "2",
            "--vector-schema-id", "test-classifier-v1",
            "--rootfolder", str(tmp_path),
            "--train-test-folder", str(tmp_path / "train"),
            "--core-h3-pool-root", "none",
            "--top-k", "1",
            "--query-batch-size", "1",
            "--mmr-candidate-pool", "1",
            "--pairs-per-city-pair", "1",
            "--output", str(output),
        ]
    )

    result = pd.read_parquet(output)
    assert result.columns.tolist() == [
        "city_1", "name_1", "panoid_1", "hex_id_1", "lat_1", "lon_1",
        "city_2", "name_2", "panoid_2", "hex_id_2", "lat_2", "lon_2",
        "cosine_similarity", "city_pair_key",
    ]
    assert result[["city_1", "city_2", "cosine_similarity"]].to_dict("records") == [
        {"city_1": "Paris", "city_2": "London", "cosine_similarity": 1.0}
    ]
    audit = json.loads(output.with_suffix(".json").read_text())
    assert audit["modality"] == "classifier_probability"
    assert audit["vector_schema_id"] == "test-classifier-v1"
    assert audit["expected_dim"] == 2
    assert audit["normalization"] == "row-wise L2 normalization before FAISS inner product"
    assert audit["vector_root"] == str(probability_root)
    assert audit["input_stats"]["Paris"]["input_rows"] == 1
    assert audit["city_pairs"]["Paris|London"]["accepted_pairs"] == 1
    assert "scene diversity" not in audit["method"].lower()


def test_classifier_gallery_uses_modality_specific_labels(tmp_path: Path) -> None:
    from sample_similar_pairs.build_image_pair_gallery import build_gallery

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paris_image = source_dir / "paris.jpg"
    london_image = source_dir / "london.jpg"
    paris_image.write_bytes(b"paris")
    london_image.write_bytes(b"london")
    index_root = tmp_path / "image-index"
    index_root.mkdir()
    pd.DataFrame({"name": ["paris.jpg"], "path": [str(paris_image)]}).to_parquet(
        index_root / "paris.parquet", index=False
    )
    pd.DataFrame({"name": ["london.jpg"], "path": [str(london_image)]}).to_parquet(
        index_root / "london.parquet", index=False
    )
    pairs_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "city_1": "Paris", "name_1": "paris.jpg", "panoid_1": "paris",
                "lat_1": 48.8566, "lon_1": 2.3522,
                "city_2": "London", "name_2": "london.jpg", "panoid_2": "london",
                "lat_2": 51.5072, "lon_2": -0.1276,
                "cosine_similarity": 0.99,
            }
        ]
    ).to_parquet(pairs_path, index=False)
    output = tmp_path / "gallery"

    build_gallery(
        pairs_path,
        index_root,
        output,
        title="Classifier-probability cross-city similar-image samples",
        similarity_label="Classifier-profile cosine similarity",
    )

    rendered = (output / "index.html").read_text()
    assert "Classifier-probability cross-city similar-image samples" in rendered
    assert "Classifier-profile cosine similarity: 0.9900" in rendered
    assert sorted(path.read_bytes() for path in (output / "images").iterdir()) == [b"london", b"paris"]
