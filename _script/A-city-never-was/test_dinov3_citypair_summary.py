from pathlib import Path
import unittest

import pandas as pd

from B5h_summarize_dinov3_citypair_similarity import summarize_citypair_similarity


def write_parquet_dataset(path: Path, parts: list[pd.DataFrame]) -> None:
    path.mkdir(parents=True)
    for index, part in enumerate(parts):
        part.to_parquet(path / f"part_{index}.parquet", index=False)


def test_summarizes_unordered_city_pairs_from_files_and_dataset_directories(tmp_path):
    input_folder = tmp_path / "agg"
    input_folder.mkdir()

    pd.DataFrame(
        {
            "city_1": ["Alpha", "Alpha", "Beta", "Beta"],
            "city_2": ["Beta", "Beta", "Alpha", "Gamma"],
            "similarity": [0.10, 0.30, 0.50, 0.90],
        }
    ).to_parquet(input_folder / "similarity_intracity_city=Alpha_res=8.parquet", index=False)

    write_parquet_dataset(
        input_folder / "similarity_intracity_city=Beta_res=8.parquet",
        [
            pd.DataFrame(
                {
                    "city_1": ["Gamma", "Alpha"],
                    "city_2": ["Alpha", "Gamma"],
                    "similarity": [0.70, 0.20],
                }
            ),
            pd.DataFrame(
                {
                    "city_1": ["Gamma"],
                    "city_2": ["Beta"],
                    "similarity": [0.60],
                }
            ),
        ],
    )

    output = tmp_path / "summary.parquet"
    result = summarize_citypair_similarity(
        input_folder=input_folder,
        output=output,
        expected_city_count=3,
    )

    assert output.exists()
    saved = pd.read_parquet(output).sort_values(["city_1", "city_2"]).reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "city_1": ["Alpha", "Alpha", "Beta"],
            "city_2": ["Beta", "Gamma", "Gamma"],
            "dinov3_cosine_avg": [0.30, 0.45, 0.75],
            "dinov3_cosine_p50": [0.30, 0.45, 0.75],
            "dinov3_cosine_p90": [0.46, 0.65, 0.87],
            "dinov3_cosine_p95": [0.48, 0.675, 0.885],
            "dinov3_cosine_max": [0.50, 0.70, 0.90],
            "dinov3_pair_count_observed": [3, 2, 2],
        }
    )
    pd.testing.assert_frame_equal(saved, expected)
    pd.testing.assert_frame_equal(result.sort_values(["city_1", "city_2"]).reset_index(drop=True), expected)


def test_fails_loudly_when_same_city_rows_remain(tmp_path):
    input_folder = tmp_path / "agg"
    input_folder.mkdir()
    pd.DataFrame(
        {
            "city_1": ["Alpha"],
            "city_2": ["Alpha"],
            "similarity": [1.0],
        }
    ).to_parquet(input_folder / "similarity_intracity_city=Alpha_res=8.parquet", index=False)

    with unittest.TestCase().assertRaisesRegex(ValueError, "same-city"):
        summarize_citypair_similarity(
            input_folder=input_folder,
            output=tmp_path / "summary.parquet",
        )


def test_validates_expected_unordered_pair_count(tmp_path):
    input_folder = tmp_path / "agg"
    input_folder.mkdir()
    pd.DataFrame(
        {
            "city_1": ["Alpha"],
            "city_2": ["Beta"],
            "similarity": [0.25],
        }
    ).to_parquet(input_folder / "similarity_intracity_city=Alpha_res=8.parquet", index=False)

    with unittest.TestCase().assertRaisesRegex(ValueError, "Expected 3 unordered city pairs"):
        summarize_citypair_similarity(
            input_folder=input_folder,
            output=tmp_path / "summary.parquet",
            expected_city_count=3,
        )


def test_writes_csv_output(tmp_path):
    input_folder = tmp_path / "agg"
    input_folder.mkdir()
    pd.DataFrame(
        {
            "city_1": ["Beta"],
            "city_2": ["Alpha"],
            "similarity": [0.25],
        }
    ).to_parquet(input_folder / "similarity_intracity_city=Beta_res=8.parquet", index=False)

    output = tmp_path / "summary.csv"
    summarize_citypair_similarity(
        input_folder=input_folder,
        output=output,
        output_format="csv",
        expected_city_count=2,
    )

    saved = pd.read_csv(output)
    assert saved.to_dict("records") == [
        {
            "city_1": "Alpha",
            "city_2": "Beta",
            "dinov3_cosine_avg": 0.25,
            "dinov3_cosine_p50": 0.25,
            "dinov3_cosine_p90": 0.25,
            "dinov3_cosine_p95": 0.25,
            "dinov3_cosine_max": 0.25,
            "dinov3_pair_count_observed": 1,
        }
    ]
