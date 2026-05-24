#!/usr/bin/env python3
"""
Inspect one exact B5b temp shard for row labels and source-city membership.

This script is for targeted debugging of a single shard file after a rerun. It
checks whether the saved row labels match the shard directory and whether the
hex ids on each side belong to city 1, city 2, both, or neither.

Sample remote run:

    python3 /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/inspect_single_b5b_shard.py \
      --shard-path '/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair/optimized/temp/city1=Accra/city2=Berezniki/part_res=8.parquet' \
      --output-prefix '/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/res8_b5b_single_shard/accra_berezniki_res8'
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_SOURCE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_hex_summary"
)


def parse_shard_path(shard_path: str | Path) -> tuple[str, str]:
    """Extract `(city1, city2)` from an optimized temp shard path."""
    path = Path(shard_path)
    city2_dir = path.parent
    city1_dir = city2_dir.parent
    return city1_dir.name.split("=", 1)[1], city2_dir.name.split("=", 1)[1]


def classify_membership_side(in_city1: bool, in_city2: bool) -> str:
    """Classify one hex id's membership across the two source-city files."""
    if in_city1 and not in_city2:
        return "city1_only"
    if not in_city1 and in_city2:
        return "city2_only"
    if in_city1 and in_city2:
        return "both"
    return "neither"


def summarize_shard_rows(
    city_1: str,
    city_2: str,
    df: pd.DataFrame,
) -> dict[str, int | str]:
    """Summarize one joined shard sample in pandas."""
    left_status = df.apply(
        lambda row: classify_membership_side(
            bool(row["left_in_city1"]), bool(row["left_in_city2"])
        ),
        axis=1,
    )
    right_status = df.apply(
        lambda row: classify_membership_side(
            bool(row["right_in_city1"]), bool(row["right_in_city2"])
        ),
        axis=1,
    )
    sim_eq_1 = df["similarity"] == 1.0
    same_city1_only = (left_status == "city1_only") & (right_status == "city1_only")
    same_city2_only = (left_status == "city2_only") & (right_status == "city2_only")
    strict_city1_to_city2 = (left_status == "city1_only") & (right_status == "city2_only")
    strict_city2_to_city1 = (left_status == "city2_only") & (right_status == "city1_only")
    return {
        "city_1": city_1,
        "city_2": city_2,
        "pair_row_count": int(len(df)),
        "row_label_match_count": int(
            ((df["row_city1"] == city_1) & (df["row_city2"] == city_2)).sum()
        ),
        "row_label_reversed_count": int(
            ((df["row_city1"] == city_2) & (df["row_city2"] == city_1)).sum()
        ),
        "strict_city1_to_city2_count": int(strict_city1_to_city2.sum()),
        "strict_city2_to_city1_count": int(strict_city2_to_city1.sum()),
        "same_city1_only_count": int(same_city1_only.sum()),
        "same_city2_only_count": int(same_city2_only.sum()),
        "ambiguous_count": int(((left_status == "both") | (right_status == "both")).sum()),
        "missing_count": int(((left_status == "neither") | (right_status == "neither")).sum()),
        "sim_eq_1_count": int(sim_eq_1.sum()),
        "sim_eq_1_same_city1_only_count": int((sim_eq_1 & same_city1_only).sum()),
        "sim_eq_1_same_city2_only_count": int((sim_eq_1 & same_city2_only).sum()),
    }


def sql_quote(value: str) -> str:
    """Return a SQL-safe string literal."""
    return value.replace("'", "''")


def get_source_path(source_root: str, city: str, res_exclude: int) -> str:
    """Return one city's source parquet path."""
    return str(Path(source_root) / f"prob_city={city}_res_exclude={res_exclude}.parquet")


def build_source_hex_query(source_path: str, resolution: int) -> str:
    """Return the SQL subquery for one city's source hex ids."""
    return f"""
        SELECT DISTINCT CAST(hex_id AS VARCHAR) AS hex_id
        FROM read_parquet('{sql_quote(source_path)}')
        WHERE res = {int(resolution)}
    """


def build_joined_rows_query(
    shard_path: str,
    source_root: str,
    resolution: int,
    res_exclude: int,
    city_1: str,
    city_2: str,
) -> str:
    """Return the SQL query joining one shard to both source-city hex sets."""
    city1_source = build_source_hex_query(
        get_source_path(source_root, city_1, res_exclude), resolution
    )
    city2_source = build_source_hex_query(
        get_source_path(source_root, city_2, res_exclude), resolution
    )
    return f"""
    WITH city1_source AS (
        {city1_source}
    ),
    city2_source AS (
        {city2_source}
    ),
    shard AS (
        SELECT
            CAST(hex_id1 AS VARCHAR) AS hex_id1,
            CAST(hex_id2 AS VARCHAR) AS hex_id2,
            similarity,
            city1 AS row_city1,
            city2 AS row_city2
        FROM read_parquet('{sql_quote(shard_path)}')
    )
    SELECT
        '{sql_quote(city_1)}' AS city_1,
        '{sql_quote(city_2)}' AS city_2,
        s.hex_id1,
        s.hex_id2,
        s.similarity,
        s.row_city1,
        s.row_city2,
        c1_left.hex_id IS NOT NULL AS left_in_city1,
        c2_left.hex_id IS NOT NULL AS left_in_city2,
        c1_right.hex_id IS NOT NULL AS right_in_city1,
        c2_right.hex_id IS NOT NULL AS right_in_city2
    FROM shard s
    LEFT JOIN city1_source c1_left ON s.hex_id1 = c1_left.hex_id
    LEFT JOIN city2_source c2_left ON s.hex_id1 = c2_left.hex_id
    LEFT JOIN city1_source c1_right ON s.hex_id2 = c1_right.hex_id
    LEFT JOIN city2_source c2_right ON s.hex_id2 = c2_right.hex_id
    """


def build_row_label_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts by saved row labels."""
    return (
        df.groupby(["row_city1", "row_city2"], as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values("row_count", ascending=False)
        .reset_index(drop=True)
    )


def build_membership_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts by left/right membership classification."""
    left_status = df.apply(
        lambda row: classify_membership_side(
            bool(row["left_in_city1"]), bool(row["left_in_city2"])
        ),
        axis=1,
    )
    right_status = df.apply(
        lambda row: classify_membership_side(
            bool(row["right_in_city1"]), bool(row["right_in_city2"])
        ),
        axis=1,
    )
    tmp = df.copy()
    tmp["left_membership"] = left_status
    tmp["right_membership"] = right_status
    return (
        tmp.groupby(["left_membership", "right_membership"], as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values("row_count", ascending=False)
        .reset_index(drop=True)
    )


def build_suspicious_examples_df(df: pd.DataFrame, example_limit: int) -> pd.DataFrame:
    """Return suspicious rows, prioritizing sim=1 and same-city membership."""
    left_status = df.apply(
        lambda row: classify_membership_side(
            bool(row["left_in_city1"]), bool(row["left_in_city2"])
        ),
        axis=1,
    )
    right_status = df.apply(
        lambda row: classify_membership_side(
            bool(row["right_in_city1"]), bool(row["right_in_city2"])
        ),
        axis=1,
    )
    tmp = df.copy()
    tmp["left_membership"] = left_status
    tmp["right_membership"] = right_status
    suspicious = tmp[
        (tmp["similarity"] == 1.0)
        | ((tmp["left_membership"] == "city1_only") & (tmp["right_membership"] == "city1_only"))
        | ((tmp["left_membership"] == "city2_only") & (tmp["right_membership"] == "city2_only"))
        | (tmp["left_membership"] == "both")
        | (tmp["right_membership"] == "both")
        | (tmp["left_membership"] == "neither")
        | (tmp["right_membership"] == "neither")
        | ((tmp["row_city1"] != tmp["city_1"]) | (tmp["row_city2"] != tmp["city_2"]))
    ].copy()
    return suspicious.head(example_limit).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one B5b temp shard for row labels and source-city membership."
    )
    parser.add_argument(
        "--shard-path",
        required=True,
        help="Absolute path to one optimized temp shard parquet file.",
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help="Root folder containing prob_city=... source feature parquet files.",
    )
    parser.add_argument(
        "--resolution",
        "--res",
        dest="resolution",
        type=int,
        default=8,
        help="H3 resolution to inspect.",
    )
    parser.add_argument(
        "--res-exclude",
        type=int,
        default=11,
        help="Suffix used in source feature file names.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=100,
        help="Max suspicious rows to print/export.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for CSV outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    shard_path = Path(args.shard_path)
    if not shard_path.exists():
        print(f"ERROR: shard path not found: {shard_path}")
        return 2

    city_1, city_2 = parse_shard_path(shard_path)
    con = duckdb.connect()
    try:
        joined_df = con.execute(
            build_joined_rows_query(
                shard_path=str(shard_path),
                source_root=args.source_root,
                resolution=args.resolution,
                res_exclude=args.res_exclude,
                city_1=city_1,
                city_2=city_2,
            )
        ).df()
    finally:
        con.close()

    summary_df = pd.DataFrame([summarize_shard_rows(city_1, city_2, joined_df)])
    row_label_counts_df = build_row_label_counts_df(joined_df)
    membership_counts_df = build_membership_counts_df(joined_df)
    suspicious_df = build_suspicious_examples_df(joined_df, args.example_limit)

    print("Summary")
    print(summary_df.to_string(index=False))
    print("\nRow Label Counts")
    print(row_label_counts_df.to_string(index=False))
    print("\nMembership Counts")
    print(membership_counts_df.to_string(index=False))
    if suspicious_df.empty:
        print("\nNo suspicious rows found.")
    else:
        print("\nSuspicious Rows")
        print(suspicious_df.to_string(index=False))

    if args.output_prefix:
        summary_path = f"{args.output_prefix}_summary.csv"
        row_labels_path = f"{args.output_prefix}_row_label_counts.csv"
        membership_path = f"{args.output_prefix}_membership_counts.csv"
        suspicious_path = f"{args.output_prefix}_suspicious_examples.csv"
        summary_df.to_csv(summary_path, index=False)
        row_label_counts_df.to_csv(row_labels_path, index=False)
        membership_counts_df.to_csv(membership_path, index=False)
        suspicious_df.to_csv(suspicious_path, index=False)
        print(
            f"\nWrote CSVs: {summary_path}, {row_labels_path}, {membership_path}, {suspicious_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
