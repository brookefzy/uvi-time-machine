#!/usr/bin/env python3
"""
Inspect B5b temp shards for duplicate hex-pairs and source-city membership mismatches.

This script reads existing optimized temp shards plus the source feature parquet files
for the two cities implied by each shard directory. It checks:

1. whether `(hex_id1, hex_id2)` repeats within the shard
2. whether unordered hex pairs repeat within the shard
3. whether each shard row maps cleanly to the expected source-city membership
4. whether suspicious `similarity == 1` rows are concentrated in missing, reversed,
   or ambiguous source-membership cases

Sample remote run for Hong Kong:

    python3 A-city-never-was/inspect_b5b_shard_membership.py \
      --resolution 8 \
      --city1 hongkong \
      --limit 25 \
      --example-limit 50 \
      --output-prefix '/lustre1/g/geog_pyloo/05_timemachine/_tmp/res8_b5b_shard_membership/hongkong_res8'

Smaller quick-check variant:

    python3 A-city-never-was/inspect_b5b_shard_membership.py \
      --resolution 8 \
      --city1 hongkong \
      --limit 5 \
      --example-limit 20 \
      --output-prefix '/lustre1/g/geog_pyloo/05_timemachine/_tmp/res8_b5b_shard_membership/hongkong_res8_small'
"""

import argparse
import sys
import unicodedata
from glob import glob
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_PAIRWISE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_similarity_by_pair"
)
DEFAULT_SOURCE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_hex_summary"
)
US_STATE_CODES = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
}


def normalize_city_name_for_filter(city: str) -> str:
    """Normalize a city name the same way B7 does for cross-source joins."""
    normalized = unicodedata.normalize("NFD", city)
    cleaned = normalized.lower().replace(",", "").replace(" ", "")
    if "," in city and len(cleaned) > 4 and cleaned[-2:] in US_STATE_CODES:
        return cleaned[:-2]
    return cleaned


def resolve_city_filter_value(
    requested_value: str | None, available_city_values: list[str]
) -> str | None:
    """Resolve a raw or normalized city filter to one exact shard directory value."""
    if not requested_value:
        return None
    if requested_value in available_city_values:
        return requested_value
    requested_normalized = normalize_city_name_for_filter(requested_value)
    matches = [
        value
        for value in available_city_values
        if normalize_city_name_for_filter(value) == requested_normalized
    ]
    if not matches:
        return requested_value
    if len(matches) > 1:
        raise ValueError(
            f"City filter '{requested_value}' is ambiguous across shard directories: {matches}"
        )
    return matches[0]


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


def summarize_shard_membership_df(
    city_1: str, city_2: str, df: pd.DataFrame
) -> dict[str, int | float | str]:
    """Summarize duplicate-pair and membership diagnostics for one shard."""
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
    ordered_pairs = df[["hex_id1", "hex_id2"]].drop_duplicates()
    unordered_pairs = df.assign(
        hex_low=df[["hex_id1", "hex_id2"]].min(axis=1),
        hex_high=df[["hex_id1", "hex_id2"]].max(axis=1),
    )[["hex_low", "hex_high"]].drop_duplicates()

    strict_city1_to_city2 = (left_status == "city1_only") & (
        right_status == "city2_only"
    )
    strict_city2_to_city1 = (left_status == "city2_only") & (
        right_status == "city1_only"
    )
    ambiguous = (left_status == "both") | (right_status == "both")
    missing = (left_status == "neither") | (right_status == "neither")
    sim_eq_1 = df["similarity"] == 1.0

    return {
        "city_1": city_1,
        "city_2": city_2,
        "pair_row_count": int(len(df)),
        "unique_ordered_pair_count": int(len(ordered_pairs)),
        "duplicate_ordered_pair_count": int(len(df) - len(ordered_pairs)),
        "unique_unordered_pair_count": int(len(unordered_pairs)),
        "duplicate_unordered_pair_count": int(len(df) - len(unordered_pairs)),
        "expected_orientation_count": int(
            ((df["left_in_city1"]) & (df["right_in_city2"])).sum()
        ),
        "reversed_orientation_count": int(
            ((df["left_in_city2"]) & (df["right_in_city1"])).sum()
        ),
        "strict_city1_to_city2_count": int(strict_city1_to_city2.sum()),
        "strict_city2_to_city1_count": int(strict_city2_to_city1.sum()),
        "ambiguous_membership_count": int(ambiguous.sum()),
        "missing_membership_count": int(missing.sum()),
        "sim_eq_1_count": int(sim_eq_1.sum()),
        "sim_eq_1_missing_membership_count": int((sim_eq_1 & missing).sum()),
    }


def sql_quote(value: str) -> str:
    """Return a SQL-safe string literal."""
    return value.replace("'", "''")


def get_source_path(source_root: str, city: str, res_exclude: int) -> str:
    """Return one city's source parquet path."""
    return str(
        Path(source_root) / f"prob_city={city}_res_exclude={res_exclude}.parquet"
    )


def build_source_hex_query(source_path: str, resolution: int) -> str:
    """Return the SQL subquery for one city's source hex ids."""
    return f"""
        SELECT DISTINCT CAST(hex_id AS VARCHAR) AS hex_id
        FROM read_parquet('{sql_quote(source_path)}')
        WHERE res = {int(resolution)}
    """


def collect_shard_paths(
    pairwise_root: str,
    resolution: int,
    city1_filter: str | None,
    city2_filter: str | None,
    limit: int | None,
) -> list[str]:
    """Return matching optimized temp shard parquet paths."""
    temp_root = Path(pairwise_root) / "optimized" / "temp"
    available_city1_values = sorted(
        path.name.split("=", 1)[1]
        for path in temp_root.glob("city1=*")
        if path.is_dir()
    )
    resolved_city1 = resolve_city_filter_value(city1_filter, available_city1_values)

    if resolved_city1:
        city1_root = temp_root / f"city1={resolved_city1}"
        available_city2_values = sorted(
            path.name.split("=", 1)[1]
            for path in city1_root.glob("city2=*")
            if path.is_dir()
        )
    else:
        available_city2_values = sorted(
            {
                path.name.split("=", 1)[1]
                for path in temp_root.glob("city1=*/city2=*")
                if path.is_dir()
            }
        )
    resolved_city2 = resolve_city_filter_value(city2_filter, available_city2_values)

    pattern = (
        temp_root
        / f"city1={resolved_city1 or '*'}"
        / f"city2={resolved_city2 or '*'}"
        / f"part_res={resolution}.parquet"
    )
    paths = sorted(glob(str(pattern)))
    if limit is not None:
        return paths[:limit]
    return paths


def build_shard_membership_summary_query(
    shard_path: str,
    source_root: str,
    resolution: int,
    res_exclude: int,
    city_1: str,
    city_2: str,
) -> str:
    """Return the SQL query summarizing one shard's membership diagnostics."""
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
    ),
    joined AS (
        SELECT
            s.*,
            c1_left.hex_id IS NOT NULL AS left_in_city1,
            c2_left.hex_id IS NOT NULL AS left_in_city2,
            c1_right.hex_id IS NOT NULL AS right_in_city1,
            c2_right.hex_id IS NOT NULL AS right_in_city2,
            LEAST(s.hex_id1, s.hex_id2) AS hex_low,
            GREATEST(s.hex_id1, s.hex_id2) AS hex_high
        FROM shard s
        LEFT JOIN city1_source c1_left ON s.hex_id1 = c1_left.hex_id
        LEFT JOIN city2_source c2_left ON s.hex_id1 = c2_left.hex_id
        LEFT JOIN city1_source c1_right ON s.hex_id2 = c1_right.hex_id
        LEFT JOIN city2_source c2_right ON s.hex_id2 = c2_right.hex_id
    )
    SELECT
        '{sql_quote(city_1)}' AS city_1,
        '{sql_quote(city_2)}' AS city_2,
        COUNT(*) AS pair_row_count,
        COUNT(DISTINCT concat_ws('|', hex_id1, hex_id2)) AS unique_ordered_pair_count,
        COUNT(*) - COUNT(DISTINCT concat_ws('|', hex_id1, hex_id2)) AS duplicate_ordered_pair_count,
        COUNT(DISTINCT concat_ws('|', hex_low, hex_high)) AS unique_unordered_pair_count,
        COUNT(*) - COUNT(DISTINCT concat_ws('|', hex_low, hex_high)) AS duplicate_unordered_pair_count,
        SUM(CASE WHEN left_in_city1 AND right_in_city2 THEN 1 ELSE 0 END) AS expected_orientation_count,
        SUM(CASE WHEN left_in_city2 AND right_in_city1 THEN 1 ELSE 0 END) AS reversed_orientation_count,
        SUM(
            CASE
                WHEN left_in_city1 AND NOT left_in_city2 AND right_in_city2 AND NOT right_in_city1
                THEN 1 ELSE 0
            END
        ) AS strict_city1_to_city2_count,
        SUM(
            CASE
                WHEN left_in_city2 AND NOT left_in_city1 AND right_in_city1 AND NOT right_in_city2
                THEN 1 ELSE 0
            END
        ) AS strict_city2_to_city1_count,
        SUM(
            CASE
                WHEN (left_in_city1 AND left_in_city2) OR (right_in_city1 AND right_in_city2)
                THEN 1 ELSE 0
            END
        ) AS ambiguous_membership_count,
        SUM(
            CASE
                WHEN ((NOT left_in_city1) AND (NOT left_in_city2))
                  OR ((NOT right_in_city1) AND (NOT right_in_city2))
                THEN 1 ELSE 0
            END
        ) AS missing_membership_count,
        SUM(CASE WHEN similarity = 1.0 THEN 1 ELSE 0 END) AS sim_eq_1_count,
        SUM(
            CASE
                WHEN similarity = 1.0 AND (
                    ((NOT left_in_city1) AND (NOT left_in_city2))
                    OR ((NOT right_in_city1) AND (NOT right_in_city2))
                )
                THEN 1 ELSE 0
            END
        ) AS sim_eq_1_missing_membership_count,
        SUM(CASE WHEN row_city1 = '{sql_quote(city_1)}' AND row_city2 = '{sql_quote(city_2)}' THEN 1 ELSE 0 END) AS row_label_match_count,
        SUM(CASE WHEN row_city1 = '{sql_quote(city_2)}' AND row_city2 = '{sql_quote(city_1)}' THEN 1 ELSE 0 END) AS row_label_reversed_count
    FROM joined
    """


def get_shard_summary(
    con: Any,
    shard_path: str,
    source_root: str,
    resolution: int,
    res_exclude: int,
    city_1: str,
    city_2: str,
) -> dict[str, int | float | str]:
    """Execute the summary query for one shard."""
    query = build_shard_membership_summary_query(
        shard_path=shard_path,
        source_root=source_root,
        resolution=resolution,
        res_exclude=res_exclude,
        city_1=city_1,
        city_2=city_2,
    )
    columns = [
        "city_1",
        "city_2",
        "pair_row_count",
        "unique_ordered_pair_count",
        "duplicate_ordered_pair_count",
        "unique_unordered_pair_count",
        "duplicate_unordered_pair_count",
        "expected_orientation_count",
        "reversed_orientation_count",
        "strict_city1_to_city2_count",
        "strict_city2_to_city1_count",
        "ambiguous_membership_count",
        "missing_membership_count",
        "sim_eq_1_count",
        "sim_eq_1_missing_membership_count",
        "row_label_match_count",
        "row_label_reversed_count",
    ]
    values = con.execute(query).fetchone()
    return {
        key: (
            int(value)
            if isinstance(value, (int, bool))
            or value is not None
            and key not in {"city_1", "city_2"}
            else value
        )
        for key, value in zip(columns, values)
    }


def get_suspicious_examples(
    con: Any,
    shard_path: str,
    source_root: str,
    resolution: int,
    res_exclude: int,
    city_1: str,
    city_2: str,
    example_limit: int,
) -> pd.DataFrame:
    """Return suspicious rows from one shard."""
    city1_source = build_source_hex_query(
        get_source_path(source_root, city_1, res_exclude), resolution
    )
    city2_source = build_source_hex_query(
        get_source_path(source_root, city_2, res_exclude), resolution
    )
    query = f"""
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
    ),
    joined AS (
        SELECT
            '{sql_quote(city_1)}' AS city_1,
            '{sql_quote(city_2)}' AS city_2,
            s.*,
            c1_left.hex_id IS NOT NULL AS left_in_city1,
            c2_left.hex_id IS NOT NULL AS left_in_city2,
            c1_right.hex_id IS NOT NULL AS right_in_city1,
            c2_right.hex_id IS NOT NULL AS right_in_city2,
            LEAST(s.hex_id1, s.hex_id2) AS hex_low,
            GREATEST(s.hex_id1, s.hex_id2) AS hex_high,
            COUNT(*) OVER (PARTITION BY s.hex_id1, s.hex_id2) AS ordered_pair_occurrences,
            COUNT(*) OVER (PARTITION BY LEAST(s.hex_id1, s.hex_id2), GREATEST(s.hex_id1, s.hex_id2)) AS unordered_pair_occurrences
        FROM shard s
        LEFT JOIN city1_source c1_left ON s.hex_id1 = c1_left.hex_id
        LEFT JOIN city2_source c2_left ON s.hex_id1 = c2_left.hex_id
        LEFT JOIN city1_source c1_right ON s.hex_id2 = c1_right.hex_id
        LEFT JOIN city2_source c2_right ON s.hex_id2 = c2_right.hex_id
    )
    SELECT *
    FROM joined
    WHERE ordered_pair_occurrences > 1
       OR unordered_pair_occurrences > 1
       OR ((NOT left_in_city1) AND (NOT left_in_city2))
       OR ((NOT right_in_city1) AND (NOT right_in_city2))
       OR ((left_in_city1 AND left_in_city2) OR (right_in_city1 AND right_in_city2))
       OR (left_in_city2 AND right_in_city1)
       OR similarity = 1.0
    LIMIT {int(example_limit)}
    """
    return con.execute(query).df()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect B5b temp shards for duplicate pairs and source-membership mismatches."
    )
    parser.add_argument(
        "--pairwise-root",
        default=DEFAULT_PAIRWISE_ROOT,
        help="Root folder containing optimized/temp shards.",
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
        "--city1",
        default=None,
        help="Optional shard directory city1 filter; raw or normalized forms accepted.",
    )
    parser.add_argument(
        "--city2",
        default=None,
        help="Optional shard directory city2 filter; raw or normalized forms accepted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of shard files to inspect.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=50,
        help="Max suspicious example rows to export per shard.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for CSV outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    shard_paths = collect_shard_paths(
        pairwise_root=args.pairwise_root,
        resolution=args.resolution,
        city1_filter=args.city1,
        city2_filter=args.city2,
        limit=args.limit,
    )
    if not shard_paths:
        print(
            f"ERROR: no temp shards found under {args.pairwise_root} for res={args.resolution}"
        )
        return 2

    con = duckdb.connect()
    try:
        summary_rows = []
        suspicious_frames = []
        for shard_path in shard_paths:
            city_1, city_2 = parse_shard_path(shard_path)
            summary_rows.append(
                get_shard_summary(
                    con=con,
                    shard_path=shard_path,
                    source_root=args.source_root,
                    resolution=args.resolution,
                    res_exclude=args.res_exclude,
                    city_1=city_1,
                    city_2=city_2,
                )
            )
            examples_df = get_suspicious_examples(
                con=con,
                shard_path=shard_path,
                source_root=args.source_root,
                resolution=args.resolution,
                res_exclude=args.res_exclude,
                city_1=city_1,
                city_2=city_2,
                example_limit=args.example_limit,
            )
            if not examples_df.empty:
                suspicious_frames.append(examples_df)
    finally:
        con.close()

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "missing_membership_count",
                "duplicate_unordered_pair_count",
                "reversed_orientation_count",
                "sim_eq_1_count",
            ],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    suspicious_df = (
        pd.concat(suspicious_frames, ignore_index=True)
        if suspicious_frames
        else pd.DataFrame()
    )

    print(f"shards_scanned={len(shard_paths)}")
    print("\nShard summary")
    print(summary_df.to_string(index=False))

    if suspicious_df.empty:
        print("\nNo suspicious rows found.")
    else:
        print("\nSuspicious example rows")
        print(suspicious_df.head(50).to_string(index=False))

    if args.output_prefix:
        summary_path = f"{args.output_prefix}_summary.csv"
        suspicious_path = f"{args.output_prefix}_suspicious_examples.csv"
        summary_df.to_csv(summary_path, index=False)
        suspicious_df.to_csv(suspicious_path, index=False)
        print(f"\nWrote CSVs: {summary_path}, {suspicious_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
