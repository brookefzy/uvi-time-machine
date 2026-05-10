#!/usr/bin/env python3
"""
Check whether the input data required by
`B5b_compute_similarity_pairwise-optimized.py --resolution <N>` is available.
"""

import argparse
import csv
import sys
from pathlib import Path

import duckdb

VECTOR_COLUMNS = [str(i) for i in range(127)]
REQUIRED_COLUMNS = ["hex_id", "res", *VECTOR_COLUMNS]


def sql_quote(value: str) -> str:
    """Quote a string for a DuckDB SQL literal."""
    return value.replace("'", "''")


def load_cities(city_meta_path: Path) -> list[str]:
    """Load city names from the city metadata CSV."""
    with city_meta_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "City" not in reader.fieldnames:
            raise ValueError(f"'City' column not found in {city_meta_path}")
        return [row["City"] for row in reader if row.get("City")]


def inspect_parquet(
    conn: duckdb.DuckDBPyConnection, file_path: Path, resolution: int
) -> dict:
    """Inspect one parquet file and report whether it is ready for B5b."""
    quoted_path = sql_quote(str(file_path))

    schema_rows = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{quoted_path}')"
    ).fetchall()
    columns = {row[0] for row in schema_rows}
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in columns]

    if missing_columns:
        return {
            "status": "missing_columns",
            "missing_columns": missing_columns,
            "row_count": None,
            "hex_count": None,
        }

    row_count, hex_count = conn.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT hex_id) AS hex_count
        FROM read_parquet('{quoted_path}')
        WHERE res = ?
        """,
        [resolution],
    ).fetchone()

    if row_count == 0:
        return {
            "status": "missing_resolution",
            "missing_columns": [],
            "row_count": row_count,
            "hex_count": hex_count,
        }

    return {
        "status": "ok",
        "missing_columns": [],
        "row_count": row_count,
        "hex_count": hex_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify B5b optimized similarity inputs before running."
    )
    parser.add_argument(
        "--city-meta",
        default="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv",
        help="Path to the city metadata CSV with a City column",
    )
    parser.add_argument(
        "--source-dir",
        default="/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        help="Directory containing prob_city=..._res_exclude=... parquet files",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=8,
        help="H3 resolution that B5b will request",
    )
    parser.add_argument(
        "--res-exclude",
        type=int,
        default=11,
        help="Exclusion suffix used in the parquet file names",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    city_meta_path = Path(args.city_meta)
    source_dir = Path(args.source_dir)

    if not city_meta_path.exists():
        print(f"ERROR: city metadata file not found: {city_meta_path}")
        return 2

    if not source_dir.exists():
        print(f"ERROR: source directory not found: {source_dir}")
        return 2

    cities = load_cities(city_meta_path)
    conn = duckdb.connect()

    missing_files = []
    missing_columns = []
    missing_resolution = []
    ok_results = []

    try:
        for city in cities:
            file_path = (
                source_dir / f"prob_city={city}_res_exclude={args.res_exclude}.parquet"
            )

            if not file_path.exists():
                missing_files.append((city, file_path))
                print(f"MISSING FILE    city={city} file={file_path}")
                continue

            result = inspect_parquet(conn, file_path, args.resolution)

            if result["status"] == "missing_columns":
                missing_columns.append((city, file_path, result["missing_columns"]))
                preview = ",".join(result["missing_columns"][:8])
                suffix = "..." if len(result["missing_columns"]) > 8 else ""
                print(
                    f"MISSING COLUMNS city={city} file={file_path} columns={preview}{suffix}"
                )
            elif result["status"] == "missing_resolution":
                missing_resolution.append((city, file_path))
                print(
                    f"NO RESOLUTION   city={city} res={args.resolution} file={file_path}"
                )
            else:
                ok_results.append(
                    (city, file_path, result["row_count"], result["hex_count"])
                )
                print(
                    f"OK              city={city} rows={result['row_count']} "
                    f"hexes={result['hex_count']} file={file_path.name}"
                )
    finally:
        conn.close()

    print("\nSummary")
    print(f"cities_total={len(cities)}")
    print(f"cities_ready={len(ok_results)}")
    print(f"missing_files={len(missing_files)}")
    print(f"missing_columns={len(missing_columns)}")
    print(f"missing_resolution={len(missing_resolution)}")

    if missing_files or missing_columns or missing_resolution:
        print(
            f"FAIL: inputs are not fully ready for B5b at resolution {args.resolution}"
        )
        return 1

    print(f"PASS: all inputs are ready for B5b at resolution {args.resolution}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
