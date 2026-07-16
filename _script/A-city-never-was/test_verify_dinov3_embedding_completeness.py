from pathlib import Path

import pandas as pd

from verify_dinov3_embedding_completeness import check_all_embeddings


def _panoid(label):
    return f"{label:<22}"[:22].replace(" ", "0")


def _write_city_inputs(root: Path, valfolder: Path, city: str, years: list[int]):
    city_stem = city.lower().replace(" ", "")
    image_root = root / "images" / city_stem
    image_root.mkdir(parents=True)
    rows = []
    meta_rows = []
    for index, year in enumerate(years):
        panoid = _panoid(f"{city_stem}_{index}")
        name = f"{panoid}_{index}.jpg"
        path = image_root / name
        path.write_bytes(b"fake")
        rows.append({"path": str(path)})
        meta_rows.append({"panoid": panoid, "year": year})

    valfolder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(valfolder / f"{city_stem}.parquet", index=False)

    meta_dir = root / "GSV" / "gsv_rgb" / city_stem / "gsvmeta"
    meta_dir.mkdir(parents=True)
    pd.DataFrame(meta_rows).to_csv(meta_dir / "gsv_pano.csv", index=False)
    return rows


def _write_embedding_shard(output_root: Path, city: str, names: list[str]):
    city_stem = city.lower().replace(" ", "")
    output_dir = output_root / city_stem
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "name": names,
            "panoid": [name[:22] for name in names],
            "model_name": ["fake-dinov3"] * len(names),
            "embedding_dim": [2] * len(names),
            "e_0000": [1.0] * len(names),
            "e_0001": [0.0] * len(names),
        }
    ).to_parquet(output_dir / f"{city_stem}_000.parquet", index=False)


def test_check_all_embeddings_reports_complete_and_incomplete_cities(tmp_path):
    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame({"City": ["Alpha City", "Beta City"]}).to_csv(city_meta, index=False)
    valfolder = tmp_path / "val"
    output_root = tmp_path / "embed"

    alpha_rows = _write_city_inputs(
        tmp_path, valfolder, "Alpha City", [2015, 2016, 2018, 2020, 2021]
    )
    beta_rows = _write_city_inputs(tmp_path, valfolder, "Beta City", [2016])

    _write_embedding_shard(
        output_root,
        "Alpha City",
        [
            Path(alpha_rows[1]["path"]).name,
            Path(alpha_rows[2]["path"]).name,
        ],
    )
    _write_embedding_shard(output_root, "Beta City", [Path(beta_rows[0]["path"]).name])

    result = check_all_embeddings(
        city_meta=city_meta,
        valfolder=valfolder,
        output_root=output_root,
        year_metadata_root=tmp_path,
        expected_model_name="fake-dinov3",
    )

    rows = {row["city"]: row for row in result["rows"]}
    assert result["summary"]["city_count"] == 2
    assert result["summary"]["complete_city_count"] == 1
    assert result["summary"]["incomplete_city_count"] == 1

    assert rows["Alpha City"]["status"] == "incomplete"
    assert rows["Alpha City"]["expected_image_count"] == 5
    assert rows["Alpha City"]["finished_image_count"] == 2
    assert rows["Alpha City"]["missing_image_count"] == 3
    assert rows["Alpha City"]["missing_examples"]

    assert rows["Beta City"]["status"] == "complete"
    assert rows["Beta City"]["expected_image_count"] == 1
    assert rows["Beta City"]["missing_image_count"] == 0


def test_check_all_embeddings_can_still_report_year_filtered_subset(tmp_path):
    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame({"City": ["Alpha City"]}).to_csv(city_meta, index=False)
    valfolder = tmp_path / "val"
    output_root = tmp_path / "embed"

    alpha_rows = _write_city_inputs(
        tmp_path, valfolder, "Alpha City", [2015, 2016, 2018, 2020, 2021]
    )
    _write_embedding_shard(
        output_root,
        "Alpha City",
        [
            Path(alpha_rows[1]["path"]).name,
            Path(alpha_rows[2]["path"]).name,
            Path(alpha_rows[3]["path"]).name,
        ],
    )

    result = check_all_embeddings(
        city_meta=city_meta,
        valfolder=valfolder,
        output_root=output_root,
        year_metadata_root=tmp_path,
        expected_model_name="fake-dinov3",
        year_filter_enabled=True,
    )

    row = result["rows"][0]
    assert row["status"] == "complete"
    assert row["expected_image_count"] == 3
    assert row["missing_image_count"] == 0


def test_check_all_embeddings_ignores_index_rows_with_missing_image_files(tmp_path):
    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame({"City": ["Alpha City"]}).to_csv(city_meta, index=False)
    valfolder = tmp_path / "val"
    output_root = tmp_path / "embed"

    alpha_rows = _write_city_inputs(tmp_path, valfolder, "Alpha City", [2016, 2017])
    Path(alpha_rows[1]["path"]).unlink()
    _write_embedding_shard(
        output_root,
        "Alpha City",
        [Path(alpha_rows[0]["path"]).name],
    )

    result = check_all_embeddings(
        city_meta=city_meta,
        valfolder=valfolder,
        output_root=output_root,
        year_metadata_root=tmp_path,
        expected_model_name="fake-dinov3",
    )

    row = result["rows"][0]
    assert row["status"] == "complete"
    assert row["expected_image_count"] == 1
    assert row["missing_image_count"] == 0
