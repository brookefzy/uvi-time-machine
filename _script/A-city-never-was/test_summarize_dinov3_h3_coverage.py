from pathlib import Path

import pandas as pd

from summarize_dinov3_h3_coverage import summarize_all_cities


def _write_h3_output(output_root: Path, city: str):
    output_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "hex_id": "hex6a",
                "res": 6,
                "img_count": 2,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 1.0,
                "e_0001": 0.0,
            },
            {
                "hex_id": "hex7a",
                "res": 7,
                "img_count": 3,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 0.0,
                "e_0001": 1.0,
            },
            {
                "hex_id": "hex7b",
                "res": 7,
                "img_count": 1,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 0.6,
                "e_0001": 0.8,
            },
            {
                "hex_id": "hex8_bad",
                "res": 8,
                "img_count": 4,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": None,
                "e_0001": 1.0,
            },
        ]
    ).to_parquet(output_root / f"dinov3_city={city}_res_exclude=None.parquet", index=False)


def test_summarize_all_cities_counts_valid_h3_grids_by_resolution(tmp_path):
    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame({"City": ["Alpha City", "Missing City"]}).to_csv(city_meta, index=False)
    output_root = tmp_path / "hex"
    _write_h3_output(output_root, "Alpha City")

    result = summarize_all_cities(
        city_meta=city_meta,
        h3_root=output_root,
        resolutions=[6, 7, 8],
    )

    rows = {(row["city"], row["res"]): row for row in result["rows"]}
    assert result["summary"]["city_count"] == 2
    assert result["summary"]["complete_city_count"] == 1
    assert result["summary"]["missing_city_count"] == 1

    assert rows[("Alpha City", 6)]["status"] == "ok"
    assert rows[("Alpha City", 6)]["valid_h3_grid_count"] == 1
    assert rows[("Alpha City", 6)]["total_image_count"] == 2

    assert rows[("Alpha City", 7)]["valid_h3_grid_count"] == 2
    assert rows[("Alpha City", 7)]["total_image_count"] == 4

    assert rows[("Alpha City", 8)]["valid_h3_grid_count"] == 0
    assert rows[("Alpha City", 8)]["invalid_embedding_row_count"] == 1

    assert rows[("Missing City", 6)]["status"] == "missing"
    assert rows[("Missing City", 7)]["status"] == "missing"
    assert rows[("Missing City", 8)]["status"] == "missing"


def test_summary_compares_all_and_equal_sampling_counts_within_h3_units(tmp_path):
    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame({"City": ["Alpha City"]}).to_csv(city_meta, index=False)
    output_root = tmp_path / "hex"
    _write_h3_output(output_root, "Alpha City")
    pd.DataFrame(
        [
            {
                "hex_id": "hex6a",
                "res": 6,
                "img_count": 1,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 1.0,
                "e_0001": 0.0,
            },
            {
                "hex_id": "hex7a",
                "res": 7,
                "img_count": 2,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 0.0,
                "e_0001": 1.0,
            },
            {
                "hex_id": "hex7b",
                "res": 7,
                "img_count": 1,
                "model_name": "fake-dinov3",
                "embedding_dim": 2,
                "e_0000": 0.6,
                "e_0001": 0.8,
            },
        ]
    ).to_parquet(
        output_root / "dinov3_city=Alpha City_res_exclude=None_sampling=equal.parquet",
        index=False,
    )

    result = summarize_all_cities(city_meta=city_meta, h3_root=output_root, resolutions=[6, 7])
    rows = {(row["city"], row["res"]): row for row in result["rows"]}

    assert rows[("Alpha City", 6)]["equal_sampling_status"] == "ok"
    assert rows[("Alpha City", 6)]["equal_total_image_count"] == 1
    assert rows[("Alpha City", 6)]["equal_image_count_difference"] == -1
    assert rows[("Alpha City", 7)]["equal_total_image_count"] == 3
    assert rows[("Alpha City", 7)]["equal_image_count_difference"] == -1
