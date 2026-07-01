#!/usr/bin/env python3
"""Summarize DINOv3 inter-city similarity rows to one row per city pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "city_1",
    "city_2",
    "dinov3_cosine_avg",
    "dinov3_cosine_p50",
    "dinov3_cosine_p90",
    "dinov3_cosine_p95",
    "dinov3_cosine_max",
    "dinov3_pair_count_observed",
]


def find_parquet_inputs(input_folder: Path) -> list[Path]:
    """Return single parquet files and parquet dataset directories."""
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_folder}")

    paths = [
        path
        for path in input_folder.iterdir()
        if path.name.endswith(".parquet") and (path.is_file() or path.is_dir())
    ]
    paths.sort(key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No parquet files or parquet dataset directories found in {input_folder}")
    return paths


def infer_expected_city_count(city_meta: Path | None) -> int | None:
    if city_meta is None:
        return None

    meta = pd.read_csv(city_meta)
    if "city" in meta.columns:
        city_column = "city"
    elif "City" in meta.columns:
        city_column = "City"
    else:
        raise ValueError(f"City metadata must contain a 'city' or 'City' column: {city_meta}")

    cities = meta[city_column].dropna().astype(str).str.strip()
    cities = cities[cities != ""].unique()
    if len(cities) < 2:
        raise ValueError(f"City metadata must contain at least two cities: {city_meta}")
    return len(cities)


def read_similarity_rows(input_folder: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in find_parquet_inputs(input_folder)]
    rows = pd.concat(frames, ignore_index=True)

    required = {"city_1", "city_2", "similarity"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Input similarity rows are missing required columns: {missing}")
    return rows


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    same_city = rows[rows["city_1"].astype(str) == rows["city_2"].astype(str)]
    if not same_city.empty:
        sample = same_city[["city_1", "city_2"]].head(5).to_dict("records")
        raise ValueError(f"Found same-city rows in final aggregated inputs; expected inter-city only: {sample}")

    working = rows[["city_1", "city_2", "similarity"]].copy()
    working["city_1"] = working["city_1"].astype(str)
    working["city_2"] = working["city_2"].astype(str)
    working["similarity"] = pd.to_numeric(working["similarity"], errors="coerce")
    if working["similarity"].isna().any():
        raise ValueError("Input similarity column contains null or non-numeric values")

    ordered = pd.DataFrame(
        {
            "city_1": working[["city_1", "city_2"]].min(axis=1),
            "city_2": working[["city_1", "city_2"]].max(axis=1),
            "similarity": working["similarity"],
        }
    )

    grouped = ordered.groupby(["city_1", "city_2"], sort=True)["similarity"]
    summary = grouped.agg(
        dinov3_cosine_avg="mean",
        dinov3_cosine_p50="median",
        dinov3_cosine_p90=lambda values: values.quantile(0.90),
        dinov3_cosine_p95=lambda values: values.quantile(0.95),
        dinov3_cosine_max="max",
        dinov3_pair_count_observed="count",
    ).reset_index()
    summary["dinov3_pair_count_observed"] = summary["dinov3_pair_count_observed"].astype("int64")
    return summary[OUTPUT_COLUMNS]


def validate_expected_pair_count(summary: pd.DataFrame, expected_city_count: int | None) -> None:
    if expected_city_count is None:
        return
    if expected_city_count < 2:
        raise ValueError("--expected-city-count must be at least 2")

    expected_pair_count = expected_city_count * (expected_city_count - 1) // 2
    observed_pair_count = len(summary)
    if observed_pair_count != expected_pair_count:
        raise ValueError(
            f"Expected {expected_pair_count} unordered city pairs from {expected_city_count} cities, "
            f"but observed {observed_pair_count}"
        )


def write_summary(summary: pd.DataFrame, output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        summary.to_parquet(output, index=False)
    elif output_format == "csv":
        summary.to_csv(output, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def summarize_citypair_similarity(
    input_folder: str | Path,
    output: str | Path,
    expected_city_count: int | None = None,
    city_meta: str | Path | None = None,
    output_format: str | None = None,
) -> pd.DataFrame:
    input_folder = Path(input_folder)
    output = Path(output)
    city_meta_path = Path(city_meta) if city_meta is not None else None

    inferred_city_count = infer_expected_city_count(city_meta_path)
    if expected_city_count is not None and inferred_city_count is not None and expected_city_count != inferred_city_count:
        raise ValueError(
            f"--expected-city-count ({expected_city_count}) does not match city metadata count "
            f"({inferred_city_count})"
        )
    expected_count = expected_city_count if expected_city_count is not None else inferred_city_count

    resolved_format = output_format or output.suffix.lstrip(".") or "parquet"
    if resolved_format not in {"parquet", "csv"}:
        raise ValueError("--output-format must be parquet or csv")

    rows = read_similarity_rows(input_folder)
    summary = summarize_rows(rows)
    validate_expected_pair_count(summary, expected_count)
    write_summary(summary, output, resolved_format)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DINOv3 B5c per-city inter-city outputs to unordered city-pair statistics."
    )
    parser.add_argument("--input-folder", required=True, type=Path, help="Folder containing B5c per-city parquet outputs")
    parser.add_argument("--output", required=True, type=Path, help="Output summary path")
    parser.add_argument("--expected-city-count", type=int, help="Validate n*(n-1)/2 unordered city-pair rows")
    parser.add_argument("--city-meta", type=Path, help="CSV with a city or City column used to infer expected city count")
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        help="Output format. Defaults to the output suffix, or parquet when no suffix is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_citypair_similarity(
        input_folder=args.input_folder,
        output=args.output,
        expected_city_count=args.expected_city_count,
        city_meta=args.city_meta,
        output_format=args.output_format,
    )
    print(f"Wrote {len(summary)} unordered city-pair rows to {args.output}")


if __name__ == "__main__":
    main()
