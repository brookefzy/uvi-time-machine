from pathlib import Path

import pandas as pd
import pytest

from export_dinov3_h3_inventory import (
    build_inventory,
    summarize_image_count_distribution,
)


def _write_summary(root: Path, city: str, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        root / f"dinov3_city={city}_res_exclude=None.parquet", index=False
    )


def test_build_inventory_selects_only_required_columns_and_resolutions(tmp_path):
    _write_summary(
        tmp_path,
        "Alpha City",
        [
            {"hex_id": "h8b", "res": 8, "img_count": 3, "e_0000": 0.2},
            {"hex_id": "h6a", "res": 6, "img_count": 7, "e_0000": 0.4},
            {"hex_id": "h9x", "res": 9, "img_count": 1, "e_0000": 0.8},
        ],
    )
    _write_summary(
        tmp_path,
        "Beta",
        [{"hex_id": "h7a", "res": 7, "img_count": 2, "e_0000": 0.6}],
    )

    inventory = build_inventory(tmp_path, resolutions=[6, 7, 8])

    assert list(inventory.columns) == ["city", "res", "hex_id", "img_count"]
    assert inventory.to_dict("records") == [
        {"city": "Alpha City", "res": 6, "hex_id": "h6a", "img_count": 7},
        {"city": "Alpha City", "res": 8, "hex_id": "h8b", "img_count": 3},
        {"city": "Beta", "res": 7, "hex_id": "h7a", "img_count": 2},
    ]


def test_distribution_reports_per_city_resolution_quantiles():
    inventory = pd.DataFrame(
        {
            "city": ["Alpha"] * 4,
            "res": [8] * 4,
            "hex_id": ["a", "b", "c", "d"],
            "img_count": [1, 2, 3, 10],
        }
    )

    distribution = summarize_image_count_distribution(inventory)

    row = distribution.iloc[0]
    assert row["city"] == "Alpha"
    assert row["res"] == 8
    assert row["h3_count"] == 4
    assert row["total_image_count"] == 16
    assert row["min_image_count"] == 1
    assert row["median_image_count"] == 2.5
    assert row["max_image_count"] == 10


def test_build_inventory_rejects_duplicate_city_resolution_h3_keys(tmp_path):
    _write_summary(
        tmp_path,
        "Alpha",
        [
            {"hex_id": "duplicate", "res": 8, "img_count": 2},
            {"hex_id": "duplicate", "res": 8, "img_count": 3},
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_inventory(tmp_path, resolutions=[8])
