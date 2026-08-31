#!/usr/bin/env python3
"""Check whether DINOv3 image embeddings are complete for every city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from B5d_dinov3_embed_city import (
    DEFAULT_MAX_YEAR,
    DEFAULT_MIN_YEAR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PANO_PATH_TEMPLATE,
    DEFAULT_ROOT,
    DEFAULT_VALFOLDER,
    collect_finished_names,
    filter_existing_image_paths,
    filter_image_index_by_year,
    load_city_image_index,
    load_city_year_metadata,
)
from dinov3_pipeline import DEFAULT_CITY_META, CITY_COLUMNS
from dinov3_utils import discover_embedding_columns, resolve_city_file_stem


def load_city_names(city_meta: str | Path) -> list[str]:
    meta = pd.read_csv(city_meta)
    city_column = next((column for column in CITY_COLUMNS if column in meta.columns), None)
    if city_column is None:
        raise ValueError(
            f"City metadata must contain one of these columns: {', '.join(CITY_COLUMNS)}"
        )
    values = meta[city_column].dropna().astype(str).str.strip()
    return list(dict.fromkeys(value for value in values if value))


def select_city_names(
    city_meta: str | Path,
    selected_cities: Sequence[str] | None = None,
) -> list[str]:
    """Return requested cities after validating them against the city metadata."""
    available = load_city_names(city_meta)
    if not selected_cities:
        return available

    requested = list(
        dict.fromkeys(str(city).strip() for city in selected_cities if str(city).strip())
    )
    unknown = [city for city in requested if city not in set(available)]
    if unknown:
        raise ValueError(f"Requested cities absent from city metadata: {unknown}")
    return requested


def collect_finished_names_for_audit(
    curated_city_folder: str | Path,
    expected_model_name: str | None = None,
) -> set[str]:
    """Validate shard metadata and load names without reading embedding vectors."""
    folder = Path(curated_city_folder)
    if not folder.exists():
        return set()

    names: set[str] = set()
    expected_embedding_dim: int | None = None
    expected_columns: list[str] | None = None
    for parquet_path in sorted(folder.glob("*.parquet")):
        try:
            schema_names = pq.read_schema(parquet_path).names
            required = {"name", "embedding_dim"}
            missing = sorted(required.difference(schema_names))
            if missing:
                raise ValueError(f"missing columns: {missing}")

            columns = ["name", "embedding_dim"]
            if "model_name" in schema_names:
                columns.append("model_name")
            df = pd.read_parquet(parquet_path, columns=columns)
        except Exception as exc:
            raise RuntimeError(f"failed reading existing shard {parquet_path}: {exc}") from exc

        if df.empty:
            raise ValueError(f"existing shard {parquet_path} is empty")
        if df["name"].duplicated().any():
            duplicates = df.loc[df["name"].duplicated(), "name"].head().tolist()
            raise ValueError(f"duplicate names in existing shard {parquet_path}: {duplicates}")

        shard_names = df["name"].dropna().astype(str).tolist()
        duplicate_across_shards = sorted(names.intersection(shard_names))
        if duplicate_across_shards:
            raise ValueError(
                f"duplicate names across existing shards: {duplicate_across_shards[:5]}"
            )

        if df["embedding_dim"].nunique() != 1:
            raise ValueError(f"existing shard {parquet_path} contains mixed embedding_dim values")
        shard_dim = int(df["embedding_dim"].iloc[0])
        if expected_embedding_dim is None:
            expected_embedding_dim = shard_dim
        elif shard_dim != expected_embedding_dim:
            raise ValueError(
                f"existing shard {parquet_path} embedding_dim={shard_dim} does not "
                f"match previous embedding_dim={expected_embedding_dim}"
            )

        if expected_model_name is not None and "model_name" in df.columns:
            model_values = set(df["model_name"].dropna().astype(str).tolist())
            if model_values and model_values != {expected_model_name}:
                raise ValueError(
                    f"existing shard {parquet_path} model_name values {sorted(model_values)} "
                    f"do not match requested model_name={expected_model_name}"
                )

        embedding_cols = discover_embedding_columns(
            pd.DataFrame(columns=schema_names)
        )
        if len(embedding_cols) != shard_dim:
            raise ValueError(
                f"existing shard {parquet_path} embedding_dim={shard_dim} does not match "
                f"{len(embedding_cols)} embedding columns"
            )
        if expected_columns is None:
            expected_columns = embedding_cols
        elif embedding_cols != expected_columns:
            raise ValueError(f"existing shard {parquet_path} has incompatible embedding columns")

        names.update(shard_names)
    return names


def expected_image_names(
    city: str,
    valfolder: str | Path,
    year_metadata_root: str | Path,
    pano_path_template: str = DEFAULT_PANO_PATH_TEMPLATE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    year_filter_enabled: bool = False,
) -> set[str]:
    city_stem = resolve_city_file_stem(city)
    df = load_city_image_index(valfolder, city_stem)
    if year_filter_enabled:
        year_metadata = load_city_year_metadata(
            year_metadata_root,
            city_stem,
            pano_path_template=pano_path_template,
        )
        df = filter_image_index_by_year(df, year_metadata, min_year, max_year)
    df = filter_existing_image_paths(df)
    return set(df["name"].dropna().astype(str).tolist())


def check_city_embedding(
    city: str,
    valfolder: str | Path,
    output_root: str | Path,
    year_metadata_root: str | Path,
    expected_model_name: str | None = None,
    pano_path_template: str = DEFAULT_PANO_PATH_TEMPLATE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    year_filter_enabled: bool = False,
    validate_vectors: bool = False,
    max_examples: int = 5,
) -> dict[str, object]:
    city_stem = resolve_city_file_stem(city)
    output_dir = Path(output_root) / city_stem
    shard_count = len(list(output_dir.glob("*.parquet"))) if output_dir.exists() else 0

    try:
        expected = expected_image_names(
            city=city,
            valfolder=valfolder,
            year_metadata_root=year_metadata_root,
            pano_path_template=pano_path_template,
            min_year=min_year,
            max_year=max_year,
            year_filter_enabled=year_filter_enabled,
        )
        collector = collect_finished_names if validate_vectors else collect_finished_names_for_audit
        finished = collector(output_dir, expected_model_name=expected_model_name)
    except Exception as exc:
        return {
            "city": city,
            "city_stem": city_stem,
            "status": "error",
            "expected_image_count": 0,
            "finished_image_count": 0,
            "missing_image_count": 0,
            "extra_finished_count": 0,
            "shard_count": shard_count,
            "missing_examples": "",
            "extra_examples": "",
            "error": str(exc),
        }

    missing = sorted(expected.difference(finished))
    extra = sorted(finished.difference(expected))
    if not expected:
        status = "empty_expected"
    elif missing:
        status = "incomplete"
    else:
        status = "complete"

    return {
        "city": city,
        "city_stem": city_stem,
        "status": status,
        "expected_image_count": int(len(expected)),
        "finished_image_count": int(len(finished.intersection(expected))),
        "missing_image_count": int(len(missing)),
        "extra_finished_count": int(len(extra)),
        "shard_count": int(shard_count),
        "missing_examples": ";".join(missing[:max_examples]),
        "extra_examples": ";".join(extra[:max_examples]),
        "error": "",
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    complete_statuses = {"complete", "empty_expected"}
    return {
        "city_count": int(len(rows)),
        "complete_city_count": int(sum(row["status"] in complete_statuses for row in rows)),
        "incomplete_city_count": int(sum(row["status"] == "incomplete" for row in rows)),
        "error_city_count": int(sum(row["status"] == "error" for row in rows)),
        "expected_image_count": int(sum(int(row["expected_image_count"]) for row in rows)),
        "finished_image_count": int(sum(int(row["finished_image_count"]) for row in rows)),
        "missing_image_count": int(sum(int(row["missing_image_count"]) for row in rows)),
        "extra_finished_count": int(sum(int(row["extra_finished_count"]) for row in rows)),
    }


def check_all_embeddings(
    city_meta: str | Path = DEFAULT_CITY_META,
    valfolder: str | Path = DEFAULT_VALFOLDER,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    year_metadata_root: str | Path = DEFAULT_ROOT,
    expected_model_name: str | None = None,
    pano_path_template: str = DEFAULT_PANO_PATH_TEMPLATE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    year_filter_enabled: bool = False,
    selected_cities: Sequence[str] | None = None,
    validate_vectors: bool = False,
) -> dict[str, object]:
    rows = [
        check_city_embedding(
            city=city,
            valfolder=valfolder,
            output_root=output_root,
            year_metadata_root=year_metadata_root,
            expected_model_name=expected_model_name,
            pano_path_template=pano_path_template,
            min_year=min_year,
            max_year=max_year,
            year_filter_enabled=year_filter_enabled,
            validate_vectors=validate_vectors,
        )
        for city in select_city_names(city_meta, selected_cities)
    ]
    return {"summary": summarize_rows(rows), "rows": rows}


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
    print("DINOv3 EMBEDDING COMPLETENESS")
    for key in [
        "city_count",
        "complete_city_count",
        "incomplete_city_count",
        "error_city_count",
        "expected_image_count",
        "finished_image_count",
        "missing_image_count",
        "extra_finished_count",
    ]:
        print(f"{key}: {summary[key]}")
    if summary["incomplete_city_count"] or summary["error_city_count"]:
        print("incomplete_or_error_cities:")
        for row in result["rows"]:
            if row["status"] in {"incomplete", "error"}:
                print(
                    f"  {row['city']}: status={row['status']} "
                    f"missing={row['missing_image_count']} error={row['error']}"
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-meta", default=DEFAULT_CITY_META)
    parser.add_argument("--valfolder", default=DEFAULT_VALFOLDER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--year-metadata-root", default=DEFAULT_ROOT)
    parser.add_argument("--pano-path-template", default=DEFAULT_PANO_PATH_TEMPLATE)
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR)
    parser.add_argument(
        "--enable-year-filter",
        action="store_true",
        dest="year_filter_enabled",
        help="Check completeness against the configured metadata year range; default checks all images",
    )
    parser.add_argument(
        "--disable-year-filter",
        action="store_false",
        dest="year_filter_enabled",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(year_filter_enabled=False)
    parser.add_argument("--expected-model-name")
    parser.add_argument(
        "--city",
        action="append",
        dest="selected_cities",
        help="Audit only this city; may be repeated",
    )
    parser.add_argument(
        "--validate-vectors",
        action="store_true",
        help="Read every embedding value and validate finiteness and vector norms",
    )
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return exit code 0 even when cities are incomplete; useful for report-only runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = check_all_embeddings(
        city_meta=args.city_meta,
        valfolder=args.valfolder,
        output_root=args.output_root,
        year_metadata_root=args.year_metadata_root,
        expected_model_name=args.expected_model_name,
        pano_path_template=args.pano_path_template,
        min_year=args.min_year,
        max_year=args.max_year,
        year_filter_enabled=args.year_filter_enabled,
        selected_cities=args.selected_cities,
        validate_vectors=args.validate_vectors,
    )
    write_outputs(result, args.output_csv, args.output_json)
    print_summary(result)
    summary = result["summary"]
    if args.allow_incomplete:
        return 0
    return 0 if summary["incomplete_city_count"] == 0 and summary["error_city_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
