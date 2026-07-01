import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

from B5e_dinov3_vector_summary import (
    DINOv3H3HexagonAggregator,
    build_default_config,
    main,
)


def _h3_cell(lat, lon, res):
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, res)
    return h3.latlng_to_cell(lat, lon, res)


def _panoid(label):
    return f"{label:<22}"[:22].replace(" ", "0")


def _write_city_metadata(root, train_test_root, city, rows, train_panoids):
    city_abbr = city.lower().replace(" ", "")
    meta_dir = root / "GSV" / "gsv_rgb" / city_abbr / "gsvmeta"
    meta_dir.mkdir(parents=True)

    pd.DataFrame(rows).to_csv(meta_dir / "gsv_pano.csv", index=False)
    pd.DataFrame({"panoid": [row["panoid"] for row in rows]}).to_csv(
        meta_dir / "gsv_path.csv", index=False
    )

    train_dir = train_test_root / "train" / city
    train_dir.mkdir(parents=True)
    for panoid in train_panoids:
        (train_dir / f"{panoid}_000.jpg").write_text("fake image")


def test_aggregates_dinov3_embeddings_with_h3_exclusion_and_unit_vectors(tmp_path):
    city = "Test City"
    input_root = tmp_path / "embed"
    output_root = tmp_path / "hex"
    root = tmp_path / "root"
    train_test_root = tmp_path / "train_test"

    train_panoid = _panoid("train")
    excluded_panoid = _panoid("excluded")
    keep_a = _panoid("keep_a")
    keep_b = _panoid("keep_b")

    overlap_lat, overlap_lon = 40.0, -73.0
    keep_lat, keep_lon = 41.0, -72.0
    assert _h3_cell(overlap_lat, overlap_lon, 11) == _h3_cell(
        overlap_lat, overlap_lon, 11
    )

    _write_city_metadata(
        root,
        train_test_root,
        city,
        [
            {"id": 1, "panoid": train_panoid, "lat": overlap_lat, "lon": overlap_lon},
            {
                "id": 2,
                "panoid": excluded_panoid,
                "lat": overlap_lat,
                "lon": overlap_lon,
            },
            {"id": 3, "panoid": keep_a, "lat": keep_lat, "lon": keep_lon},
            {"id": 4, "panoid": keep_b, "lat": keep_lat, "lon": keep_lon},
        ],
        [train_panoid],
    )

    city_abbr = city.lower().replace(" ", "")
    embed_dir = input_root / city_abbr
    embed_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "name": f"{excluded_panoid}_excluded.jpg",
                "panoid": excluded_panoid,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 10.0,
                "e_0001": 0.0,
            },
            {
                "name": f"{keep_a}_a.jpg",
                "panoid": keep_a,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 3.0,
                "e_0001": 4.0,
            },
            {
                "name": f"{keep_b}_b.jpg",
                "panoid": keep_b,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 0.0,
                "e_0001": 4.0,
            },
        ]
    ).to_parquet(embed_dir / "testcity_000.parquet", index=False)

    config = build_default_config()
    config.update(
        {
            "ROOTFOLDER": str(root),
            "CURATED_FOLDER": str(input_root),
            "CURATE_FOLDER_EXPORT": str(output_root),
            "TRAIN_TEST_FOLDER": str(train_test_root),
        }
    )

    aggregator = DINOv3H3HexagonAggregator(config, log_level="ERROR")

    assert aggregator.process_city(city, res_exclude=11, allow_empty=False)

    output_file = output_root / "dinov3_city=Test City_res_exclude=11.parquet"
    stats_file = output_file.with_suffix(".json")
    assert output_file.exists()
    assert stats_file.exists()

    result = pd.read_parquet(output_file)
    assert set(result["res"].unique()) == {6, 7, 8}
    assert "img_count" in result.columns
    assert {"max_class", "max_prob", "second_class"}.isdisjoint(result.columns)

    res8 = result[result["res"] == 8]
    assert len(res8) == 1
    row = res8.iloc[0]
    assert row["hex_id"] == _h3_cell(keep_lat, keep_lon, 8)
    assert row["img_count"] == 2
    expected = np.array([1.5, 4.0])
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(
        row[["e_0000", "e_0001"]].to_numpy(dtype=float), expected, rtol=1e-6
    )

    stats = json.loads(stats_file.read_text())
    assert stats["status"] == "ok"
    assert stats["city"] == city
    assert stats["model_name"] == "fake-dinov3"
    assert stats["embedding_dim"] == 2
    assert stats["image_count_before_exclusion"] == 3
    assert stats["image_count_after_exclusion"] == 2
    assert stats["excluded_image_count"] == 1
    assert stats["res8_h3_count"] == 1
    assert stats["res8_mean_image_count"] == 2.0


def test_empty_after_exclusion_writes_sidecar_and_cli_exits_nonzero(tmp_path):
    city = "Empty City"
    input_root = tmp_path / "embed"
    output_root = tmp_path / "hex"
    root = tmp_path / "root"
    train_test_root = tmp_path / "train_test"

    train_panoid = _panoid("train")
    only_embedding = _panoid("only")
    lat, lon = 35.0, -80.0

    _write_city_metadata(
        root,
        train_test_root,
        city,
        [
            {"id": 1, "panoid": train_panoid, "lat": lat, "lon": lon},
            {"id": 2, "panoid": only_embedding, "lat": lat, "lon": lon},
        ],
        [train_panoid],
    )

    city_abbr = city.lower().replace(" ", "")
    embed_dir = input_root / city_abbr
    embed_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "name": f"{only_embedding}_only.jpg",
                "panoid": only_embedding,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 1.0,
                "e_0001": 0.0,
            }
        ]
    ).to_parquet(embed_dir / "emptycity_000.parquet", index=False)

    exit_code = main(
        [
            "--city",
            city,
            "--rootfolder",
            str(root),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--train-test-folder",
            str(train_test_root),
            "--res-exclude",
            "11",
            "--log-level",
            "ERROR",
        ]
    )

    output_file = output_root / "dinov3_city=Empty City_res_exclude=11.parquet"
    stats_file = output_file.with_suffix(".json")
    assert exit_code == 1
    assert not output_file.exists()
    assert stats_file.exists()

    stats = json.loads(stats_file.read_text())
    assert stats["status"] == "empty"
    assert stats["image_count_before_exclusion"] == 1
    assert stats["image_count_after_exclusion"] == 0
    assert stats["excluded_image_count"] == 1

    allow_empty_exit = main(
        [
            "--city",
            city,
            "--rootfolder",
            str(root),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--train-test-folder",
            str(train_test_root),
            "--res-exclude",
            "11",
            "--allow-empty",
            "--log-level",
            "ERROR",
        ]
    )
    assert allow_empty_exit == 0


def test_embedding_loader_rejects_non_finite_vectors(tmp_path):
    city = "Bad City"
    input_root = tmp_path / "embed"
    output_root = tmp_path / "hex"
    root = tmp_path / "root"
    train_test_root = tmp_path / "train_test"
    city_abbr = city.lower().replace(" ", "")
    embed_dir = input_root / city_abbr
    embed_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "name": "badpanoid_abcdefghijkl.jpg",
                "panoid": "badpanoid_abcdefghijkl",
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": np.nan,
                "e_0001": 1.0,
            }
        ]
    ).to_parquet(embed_dir / "badcity_000.parquet", index=False)

    config = build_default_config()
    config.update(
        {
            "ROOTFOLDER": str(root),
            "CURATED_FOLDER": str(input_root),
            "CURATE_FOLDER_EXPORT": str(output_root),
            "TRAIN_TEST_FOLDER": str(train_test_root),
        }
    )

    aggregator = DINOv3H3HexagonAggregator(config, log_level="ERROR")

    with pytest.raises(ValueError, match="non-finite"):
        aggregator.load_embedding_data(city)
