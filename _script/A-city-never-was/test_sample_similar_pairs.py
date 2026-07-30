"""Tests for exact cross-city DINOv3 similar-pair sampling."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def test_sampling_entry_points_expose_parsers():
    from sample_similar_pairs.sample_h3_pairs_faiss import build_parser as h3_parser
    from sample_similar_pairs.sample_image_pairs_faiss import build_parser as image_parser

    assert image_parser().prog
    assert h3_parser().prog
    assert "--core-h3-pool-root" in image_parser()._option_string_actions
    assert "--core-h3-pool-root" in h3_parser()._option_string_actions


def _embedding_rows(names, vectors):
    rows = []
    for name, vector in zip(names, vectors):
        rows.append(
            {
                "name": name,
                "panoid": name[:22],
                "model_name": "test-dino",
                "embedding_dim": len(vector),
                **{f"e_{index:04d}": value for index, value in enumerate(vector)},
            }
        )
    return rows


def test_load_city_embeddings_validates_and_samples_deterministically(tmp_path):
    from sample_similar_pairs.common import load_city_embeddings, spatially_sample_images

    root = tmp_path / "embed"
    city_dir = root / "paris"
    city_dir.mkdir(parents=True)
    rows = _embedding_rows(
        ["panoid-0000000000000001_a.jpg", "panoid-0000000000000002_b.jpg", "panoid-0000000000000003_c.jpg"],
        [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
    )
    pd.DataFrame(rows[:2]).to_parquet(city_dir / "paris_000.parquet", index=False)
    pd.DataFrame(rows[2:]).to_parquet(city_dir / "paris_001.parquet", index=False)

    loaded = load_city_embeddings(root, "Paris")
    assert loaded.city == "Paris"
    assert loaded.vector_columns == ["e_0000", "e_0001"]
    assert loaded.vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0)

    sampled_input = loaded.metadata.assign(hex_id=["a", "a", "b"])
    first = spatially_sample_images(sampled_input.sample(frac=1, random_state=1), 1, 2)
    second = spatially_sample_images(sampled_input.sample(frac=1, random_state=2), 1, 2)
    assert first["name"].tolist() == second["name"].tolist() == [rows[0]["name"], rows[2]["name"]]


def test_load_city_embeddings_rejects_non_unit_vectors(tmp_path):
    from sample_similar_pairs.common import load_city_embeddings

    city_dir = tmp_path / "embed" / "paris"
    city_dir.mkdir(parents=True)
    pd.DataFrame(_embedding_rows(["bad.jpg"], [[2.0, 0.0]])).to_parquet(
        city_dir / "paris_000.parquet", index=False
    )

    with pytest.raises(ValueError, match="L2-normalized"):
        load_city_embeddings(tmp_path / "embed", "Paris")


def test_parse_city_pairs_rejects_duplicate_and_self_pairs():
    from sample_similar_pairs.common import parse_city_pairs

    assert parse_city_pairs(["Paris|London"]) == [("Paris", "London")]
    with pytest.raises(ValueError, match="same city"):
        parse_city_pairs(["Paris|Paris"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_city_pairs(["Paris|London", "Paris|London"])


class _FakeIndexFlatIP:
    def __init__(self, dimension):
        self.dimension = dimension


class _FakeIndexIDMap2:
    def __init__(self, _base):
        self.vectors = None
        self.ids = None

    def add_with_ids(self, vectors, ids):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)

    def search(self, queries, top_k):
        scores = np.asarray(queries) @ self.vectors.T
        positions = np.argsort(-scores, axis=1)[:, :top_k]
        return np.take_along_axis(scores, positions, axis=1), self.ids[positions]


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP
    IndexIDMap2 = _FakeIndexIDMap2


def test_image_search_returns_exact_cross_city_matches():
    from sample_similar_pairs.common import CityVectors
    from sample_similar_pairs.sample_image_pairs_faiss import search_image_pair

    paris = CityVectors(
        "Paris",
        pd.DataFrame({"name": ["paris.jpg"], "panoid": ["paris"], "hex_id": ["phex"], "lat": [48.8566], "lon": [2.3522]}),
        ["e_0000", "e_0001"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    london = CityVectors(
        "London",
        pd.DataFrame({"name": ["london.jpg", "other.jpg"], "panoid": ["london", "other"], "hex_id": ["lhex", "ohex"], "lat": [51.5072, 51.5], "lon": [-0.1276, -0.1]}),
        ["e_0000", "e_0001"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    result, audit = search_image_pair(paris, london, top_k=2, threshold=0.9, query_batch_size=1, faiss_module=_FakeFaiss)

    assert result[["city_1", "name_1", "city_2", "name_2"]].to_dict("records") == [
        {"city_1": "Paris", "name_1": "paris.jpg", "city_2": "London", "name_2": "london.jpg"}
    ]
    assert result["cosine_similarity"].tolist() == [1.0]
    assert result[["lat_1", "lon_1", "lat_2", "lon_2"]].to_dict("records") == [
        {"lat_1": 48.8566, "lon_1": 2.3522, "lat_2": 51.5072, "lon_2": -0.1276}
    ]
    assert audit["threshold_hits"] == 1


def test_diversity_caps_repeated_source_and_hex_pairs():
    from sample_similar_pairs.sample_image_pairs_faiss import apply_image_diversity_caps

    candidates = pd.DataFrame(
        [
            {"city_1": "Paris", "name_1": "a", "hex_id_1": "x", "city_2": "London", "name_2": "b", "hex_id_2": "y", "cosine_similarity": 0.99},
            {"city_1": "Paris", "name_1": "a", "hex_id_1": "x", "city_2": "London", "name_2": "c", "hex_id_2": "z", "cosine_similarity": 0.98},
            {"city_1": "Paris", "name_1": "d", "hex_id_1": "x", "city_2": "London", "name_2": "e", "hex_id_2": "y", "cosine_similarity": 0.97},
        ]
    )
    accepted = apply_image_diversity_caps(candidates, max_pairs_per_source_image=1, max_pairs_per_hex_pair=1, pairs_per_city_pair=10)
    assert accepted[["name_1", "name_2"]].to_dict("records") == [{"name_1": "a", "name_2": "b"}]


def test_mmr_reranking_selects_a_less_redundant_scene():
    from sample_similar_pairs.common import CityVectors
    from sample_similar_pairs.sample_image_pairs_faiss import select_mmr_image_pairs

    source = CityVectors(
        "Paris",
        pd.DataFrame({"name": ["tunnel-a", "tunnel-b", "street"], "panoid": ["a", "b", "c"]}),
        ["e_0000", "e_0001"],
        np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]], dtype=np.float32),
    )
    target = CityVectors(
        "London",
        pd.DataFrame({"name": ["tunnel-a", "tunnel-b", "street"], "panoid": ["d", "e", "f"]}),
        ["e_0000", "e_0001"],
        np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]], dtype=np.float32),
    )
    candidates = pd.DataFrame(
        [
            {"name_1": "tunnel-a", "name_2": "tunnel-a", "cosine_similarity": 0.99, "_source_index": 0, "_target_index": 0},
            {"name_1": "tunnel-b", "name_2": "tunnel-b", "cosine_similarity": 0.98, "_source_index": 1, "_target_index": 1},
            {"name_1": "street", "name_2": "street", "cosine_similarity": 0.90, "_source_index": 2, "_target_index": 2},
        ]
    )

    selected = select_mmr_image_pairs(
        candidates,
        source,
        target,
        candidate_pool_size=3,
        relevance_weight=0.8,
        pairs_per_city_pair=2,
    )

    assert selected[["name_1", "name_2"]].to_dict("records") == [
        {"name_1": "tunnel-a", "name_2": "tunnel-a"},
        {"name_1": "street", "name_2": "street"},
    ]
    assert "_source_index" not in selected and "_target_index" not in selected


def test_image_sampler_defaults_rank_top_ten_without_a_cosine_cutoff():
    from sample_similar_pairs.sample_image_pairs_faiss import (
        apply_image_diversity_caps,
        build_parser,
    )

    args = build_parser().parse_args(["--output", "pairs.parquet"])
    assert args.max_images_per_h3 == 100
    assert args.max_images_per_city == 0
    assert args.top_k == 30
    assert args.threshold == -1.0
    assert args.pairs_per_city_pair == 10
    assert args.mmr_candidate_pool == 200
    assert args.mmr_relevance_weight == 0.7
    assert args.max_pairs_per_source_image == 1
    assert args.max_pairs_per_hex_pair == 1
    candidates = pd.DataFrame(
        [
            {"city_1": "Paris", "name_1": f"a{index}", "hex_id_1": f"x{index}", "city_2": "London", "name_2": f"b{index}", "hex_id_2": f"y{index}", "cosine_similarity": 1 - index / 100}
            for index in range(12)
        ]
    )
    accepted = apply_image_diversity_caps(
        candidates,
        max_pairs_per_source_image=0,
        max_pairs_per_hex_pair=0,
        pairs_per_city_pair=10,
    )
    assert accepted["cosine_similarity"].tolist() == [1 - index / 100 for index in range(10)]


def test_h3_search_filters_resolution_and_returns_exact_matches(tmp_path):
    from sample_similar_pairs.sample_h3_pairs_faiss import load_h3_vectors, search_h3_pair

    root = tmp_path / "h3"
    root.mkdir()
    for city, hex_id in [("Paris", "phex"), ("London", "lhex")]:
        pd.DataFrame(
            [
                {"hex_id": hex_id, "res": 8, "img_count": 3, "embedding_dim": 2, "e_0000": 1.0, "e_0001": 0.0},
                {"hex_id": f"{hex_id}-other", "res": 7, "img_count": 3, "embedding_dim": 2, "e_0000": 0.0, "e_0001": 1.0},
            ]
        ).to_parquet(root / f"dinov3_city={city}_res_exclude=None.parquet", index=False)

    paris = load_h3_vectors(root, "Paris", "dinov3_city={city}_res_exclude=None.parquet", 8)
    london = load_h3_vectors(root, "London", "dinov3_city={city}_res_exclude=None.parquet", 8)
    result, audit = search_h3_pair(paris, london, top_k=1, threshold=0.9, query_batch_size=1, faiss_module=_FakeFaiss)

    assert result[["hex_id_1", "hex_id_2", "cosine_similarity"]].to_dict("records") == [
        {"hex_id_1": "phex", "hex_id_2": "lhex", "cosine_similarity": 1.0}
    ]
    assert audit["threshold_hits"] == 1


def test_gallery_builder_copies_images_and_writes_side_by_side_html(tmp_path):
    from sample_similar_pairs.build_image_pair_gallery import build_gallery

    source_a = tmp_path / "source" / "paris.jpg"
    source_b = tmp_path / "source" / "london.jpg"
    source_a.parent.mkdir()
    source_a.write_bytes(b"paris image")
    source_b.write_bytes(b"london image")
    index_root = tmp_path / "image-index"
    index_root.mkdir()
    pd.DataFrame({"path": [str(source_a)]}).to_parquet(index_root / "paris.parquet", index=False)
    pd.DataFrame({"name": ["london.jpg"], "path": [str(source_b)]}).to_parquet(index_root / "london.parquet", index=False)
    pairs = pd.DataFrame([{
        "city_1": "Paris", "name_1": "paris.jpg", "panoid_1": "paris", "lat_1": 48.8566, "lon_1": 2.3522,
        "city_2": "London", "name_2": "london.jpg", "panoid_2": "london", "lat_2": 51.5072, "lon_2": -0.1276,
        "cosine_similarity": 0.99, "city_pair_key": "London|Paris",
    }])
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    output = tmp_path / "preview"
    manifest = build_gallery(pairs_path, index_root, output)

    assert len(manifest) == 1
    assert (output / "index.html").exists()
    assert sorted(path.read_bytes() for path in (output / "images").iterdir()) == [b"london image", b"paris image"]
    html = (output / "index.html").read_text()
    assert "Paris ↔ London" in html
    assert "Cosine similarity: 0.9900" in html
    assert "images/" in html
    assert "48.8566" in html and "51.5072" in html
    assert "leaflet" in html.lower()


def test_export_core_h3_pools_selects_only_requested_resolution_and_tier(tmp_path):
    from sample_similar_pairs.export_core_h3_pools import export_core_h3_pools

    source_root = tmp_path / "tiers"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"h3_index": "core-res8", "resolution": 8, "tier_pct": "core"},
            {"h3_index": "suburban-res8", "resolution": 8, "tier_pct": "suburban"},
            {"h3_index": "core-res7", "resolution": 7, "tier_pct": "core"},
        ]
    ).to_parquet(source_root / "paris_h3_poi_tiers.parquet", index=False)

    output_root = tmp_path / "pools"
    audit = export_core_h3_pools(
        source_root=source_root,
        output_root=output_root,
        cities=["Paris"],
        resolution=8,
        profile_id="pct5_sub30_z1_m05",
    )

    pool_path = output_root / "res=8" / "profile=pct5_sub30_z1_m05" / "paris.parquet"
    assert pd.read_parquet(pool_path).to_dict("records") == [{"hex_id": "core-res8"}]
    assert audit["cities"]["Paris"] == {"input_rows": 3, "core_hex_count": 1}
    assert (pool_path.parent / "core_h3_pool_audit.json").exists()


def test_export_core_h3_pools_cli_runs_as_a_standalone_script(tmp_path):
    source_root = tmp_path / "tiers"
    source_root.mkdir()
    pd.DataFrame([{"h3_index": "core-res8", "resolution": 8, "tier_pct": "core"}]).to_parquet(
        source_root / "paris_h3_poi_tiers.parquet", index=False
    )
    script = Path(__file__).parent / "sample_similar_pairs" / "export_core_h3_pools.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--source-root", str(source_root),
            "--output-root", str(tmp_path / "pools"), "--cities", "Paris",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_filter_city_to_core_h3_pool_removes_non_core_rows(tmp_path):
    from sample_similar_pairs.common import CityVectors, filter_city_to_core_h3_pool

    pool_dir = tmp_path / "pools" / "res=8" / "profile=pct5_sub30_z1_m05"
    pool_dir.mkdir(parents=True)
    pd.DataFrame({"hex_id": ["core"]}).to_parquet(pool_dir / "paris.parquet", index=False)
    vectors = CityVectors(
        "Paris",
        pd.DataFrame({"name": ["core.jpg", "rural.jpg"], "hex_id": ["core", "rural"]}),
        ["e_0000"],
        np.array([[1.0], [1.0]], dtype=np.float32),
    )

    filtered, stats = filter_city_to_core_h3_pool(
        vectors, pool_root=tmp_path / "pools", resolution=8, profile_id="pct5_sub30_z1_m05"
    )

    assert filtered.metadata["hex_id"].tolist() == ["core"]
    assert stats == {"before_rows": 2, "core_hex_count": 1, "after_rows": 1}
