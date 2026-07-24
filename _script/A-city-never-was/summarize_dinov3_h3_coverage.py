#!/usr/bin/env python3
"""Summarize valid DINOv3 H3 grids per city and resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from B5e_dinov3_vector_summary import DEFAULT_OUTPUT_ROOT
from dinov3_pipeline import DEFAULT_CITY_META, CITY_COLUMNS
from dinov3_utils import discover_embedding_columns, resolve_city_file_stem


DEFAULT_VALFOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir"
)


def parse_optional_res_exclude(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "no", "false"}:
        return None
    return str(int(text))


def load_city_names(city_meta: str | Path) -> list[str]:
    meta = pd.read_csv(city_meta)
    city_column = next((column for column in CITY_COLUMNS if column in meta.columns), None)
    if city_column is None:
        raise ValueError(
            f"City metadata must contain one of these columns: {', '.join(CITY_COLUMNS)}"
        )
    values = meta[city_column].dropna().astype(str).str.strip()
    return list(dict.fromkeys(value for value in values if value))


def h3_output_path(
    h3_root: str | Path, city: str, res_exclude: int | None, equal_sampling: bool = False
) -> Path:
    suffix = "_sampling=equal" if equal_sampling else ""
    return Path(h3_root) / f"dinov3_city={city}_res_exclude={str(res_exclude)}{suffix}.parquet"


def _count_column(df: pd.DataFrame) -> str:
    if "img_count" in df.columns:
        return "img_count"
    raise ValueError("H3 output must contain img_count")


def source_image_count(city: str, valfolder: str | Path) -> int:
    """Count live source-image paths for one city, treating a missing index as empty."""
    index_path = Path(valfolder) / f"{resolve_city_file_stem(city)}.parquet"
    if not index_path.exists():
        return 0
    image_index = pd.read_parquet(index_path, columns=["path"])
    return int(sum(Path(str(path)).exists() for path in image_index["path"].dropna()))


def summarize_output(
    city: str, path: Path, resolutions: Sequence[int]
) -> list[dict[str, object]]:
    if not path.exists():
        return [
            {
                "city": city,
                "res": int(res),
                "status": "missing",
                "h3_grid_count": 0,
                "valid_h3_grid_count": 0,
                "invalid_embedding_row_count": 0,
                "total_image_count": 0,
                "mean_image_count": 0.0,
                "min_image_count": 0,
                "max_image_count": 0,
                "embedding_dim": 0,
                "path": str(path),
                "error": "",
                "included_years": [],
            }
            for res in resolutions
        ]

    try:
        df = pd.read_parquet(path)
        required = {"hex_id", "res"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        count_col = _count_column(df)
        embedding_cols = discover_embedding_columns(df)
        values = df[embedding_cols].to_numpy(dtype=float)
        finite_rows = np.isfinite(values).all(axis=1)
        positive_counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0) > 0
        valid_rows = finite_rows & positive_counts.to_numpy()
    except Exception as exc:
        return [
            {
                "city": city,
                "res": int(res),
                "status": "error",
                "h3_grid_count": 0,
                "valid_h3_grid_count": 0,
                "invalid_embedding_row_count": 0,
                "total_image_count": 0,
                "mean_image_count": 0.0,
                "min_image_count": 0,
                "max_image_count": 0,
                "embedding_dim": 0,
                "path": str(path),
                "error": str(exc),
                "included_years": [],
            }
            for res in resolutions
        ]

    sidecar = path.with_suffix(".json")
    included_years: list[int] = []
    if sidecar.exists():
        try:
            included_years = [int(year) for year in json.loads(sidecar.read_text()).get("included_years", [])]
        except (ValueError, json.JSONDecodeError):
            pass
    rows = []
    for res in resolutions:
        res_mask = df["res"].astype(int) == int(res)
        res_df = df.loc[res_mask].copy()
        res_valid = valid_rows[res_mask.to_numpy()]
        valid_df = res_df.loc[res_valid].copy()
        image_counts = pd.to_numeric(valid_df[count_col], errors="coerce").fillna(0)
        rows.append(
            {
                "city": city,
                "res": int(res),
                "status": "ok",
                "h3_grid_count": int(len(res_df)),
                "valid_h3_grid_count": int(len(valid_df)),
                "invalid_embedding_row_count": int(len(res_df) - len(valid_df)),
                "total_image_count": int(image_counts.sum()) if not image_counts.empty else 0,
                "mean_image_count": float(image_counts.mean()) if not image_counts.empty else 0.0,
                "min_image_count": int(image_counts.min()) if not image_counts.empty else 0,
                "max_image_count": int(image_counts.max()) if not image_counts.empty else 0,
                "embedding_dim": int(len(embedding_cols)),
                "path": str(path),
                "error": "",
                "included_years": included_years,
            }
        )
    return rows


def summarize_city_h3(
    city: str,
    h3_root: str | Path,
    res_exclude: str | int | None = None,
    resolutions: Sequence[int] = (6, 7, 8),
    valfolder: str | Path = DEFAULT_VALFOLDER,
) -> list[dict[str, object]]:
    all_rows = summarize_output(city, h3_output_path(h3_root, city, res_exclude), resolutions)
    image_count = source_image_count(city, valfolder)
    if image_count == 0 and all(row["status"] == "missing" for row in all_rows):
        for row in all_rows:
            row["status"] = "ignored_no_images"
    for row in all_rows:
        row["source_image_count"] = image_count
    equal_rows = summarize_output(
        city, h3_output_path(h3_root, city, res_exclude, equal_sampling=True), resolutions
    )
    for all_row, equal_row in zip(all_rows, equal_rows):
        all_row["equal_sampling_status"] = equal_row["status"]
        all_row["equal_sampling_path"] = equal_row["path"]
        all_row["equal_total_image_count"] = equal_row["total_image_count"]
        all_row["equal_valid_h3_grid_count"] = equal_row["valid_h3_grid_count"]
        all_row["equal_image_count_difference"] = (
            int(equal_row["total_image_count"]) - int(all_row["total_image_count"])
        )
    return all_rows


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    cities = sorted({str(row["city"]) for row in rows})
    city_status = {}
    for city in cities:
        statuses = {row["status"] for row in rows if row["city"] == city}
        if statuses == {"ignored_no_images"}:
            city_status[city] = "ignored_no_images"
        elif "error" in statuses:
            city_status[city] = "error"
        elif "missing" in statuses:
            city_status[city] = "missing"
        else:
            city_status[city] = "ok"
    return {
        "city_count": int(len(cities)),
        "complete_city_count": int(sum(status == "ok" for status in city_status.values())),
        "missing_city_count": int(sum(status == "missing" for status in city_status.values())),
        "error_city_count": int(sum(status == "error" for status in city_status.values())),
        "ignored_no_images_city_count": int(
            sum(status == "ignored_no_images" for status in city_status.values())
        ),
        "total_valid_h3_grid_count": int(sum(int(row["valid_h3_grid_count"]) for row in rows)),
        "total_image_count": int(sum(int(row["total_image_count"]) for row in rows)),
    }


def summarize_all_cities(
    city_meta: str | Path = DEFAULT_CITY_META,
    h3_root: str | Path = DEFAULT_OUTPUT_ROOT,
    res_exclude: str | int | None = None,
    resolutions: Sequence[int] = (6, 7, 8),
    valfolder: str | Path = DEFAULT_VALFOLDER,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for city in load_city_names(city_meta):
        rows.extend(summarize_city_h3(city, h3_root, res_exclude, resolutions, valfolder))
    return {"summary": summarize_rows(rows), "rows": rows}


def _parse_resolutions(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_outputs(result: dict[str, object], output_csv: str | Path | None, output_json: str | Path | None) -> None:
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["rows"]).to_csv(path, index=False)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))


def print_summary(result: dict[str, object]) -> None:
    summary = result["summary"]
    print("DINOv3 H3 COVERAGE SUMMARY")
    for key in [
        "city_count",
        "complete_city_count",
        "missing_city_count",
        "error_city_count",
        "ignored_no_images_city_count",
        "total_valid_h3_grid_count",
        "total_image_count",
    ]:
        print(f"{key}: {summary[key]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-meta", default=DEFAULT_CITY_META)
    parser.add_argument("--h3-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--valfolder", default=DEFAULT_VALFOLDER)
    parser.add_argument("--res-exclude", default=None)
    parser.add_argument("--resolutions", default="6,7,8")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return exit code 0 even when city H3 outputs are missing or invalid",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize_all_cities(
        city_meta=args.city_meta,
        h3_root=args.h3_root,
        valfolder=args.valfolder,
        res_exclude=parse_optional_res_exclude(args.res_exclude),
        resolutions=_parse_resolutions(args.resolutions),
    )
    write_outputs(result, args.output_csv, args.output_json)
    print_summary(result)
    summary = result["summary"]
    if args.allow_missing:
        return 0
    return 0 if summary["missing_city_count"] == 0 and summary["error_city_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
