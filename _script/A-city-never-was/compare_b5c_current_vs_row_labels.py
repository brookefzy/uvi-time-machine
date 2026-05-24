#!/usr/bin/env python3
"""
Compare B5c's current shard-label aggregation against row-label aggregation.

This emulates the exact optimized B5c logic on existing temp shards, then
re-runs the same aggregation using the row-level `city1` / `city2` columns
already stored in each shard. Large deltas support the hypothesis that the
optimized B5c rewrite is contaminating inter-city outputs with same-city rows.
"""

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Any
import unicodedata

import duckdb
import pandas as pd


DEFAULT_PAIRWISE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_similarity_by_pair"
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


def build_metric_diff(current_df: pd.DataFrame, fixed_df: pd.DataFrame) -> pd.DataFrame:
    """Join the current and corrected pair metrics and compute deltas."""
    merged = current_df.merge(
        fixed_df,
        on=["city_1", "city_2"],
        how="outer",
        suffixes=("_current", "_fixed"),
    ).fillna(0)

    renamed = merged.rename(
        columns={
            "row_count_current": "current_row_count",
            "row_count_fixed": "fixed_row_count",
            "max_similarity_current": "current_max_similarity",
            "max_similarity_fixed": "fixed_max_similarity",
            "avg_similarity_current": "current_avg_similarity",
            "avg_similarity_fixed": "fixed_avg_similarity",
            "sum_similarity_current": "current_sum_similarity",
            "sum_similarity_fixed": "fixed_sum_similarity",
        }
    )

    renamed["row_count_delta"] = (
        renamed["current_row_count"] - renamed["fixed_row_count"]
    )
    renamed["sum_similarity_delta"] = (
        renamed["current_sum_similarity"] - renamed["fixed_sum_similarity"]
    )
    renamed["max_similarity_delta"] = (
        renamed["current_max_similarity"] - renamed["fixed_max_similarity"]
    )
    renamed["avg_similarity_delta"] = (
        renamed["current_avg_similarity"] - renamed["fixed_avg_similarity"]
    )
    renamed["row_count_ratio"] = renamed.apply(
        lambda row: (
            row["current_row_count"] / row["fixed_row_count"]
            if row["fixed_row_count"] > 0
            else float("inf")
            if row["current_row_count"] > 0
            else 0.0
        ),
        axis=1,
    )

    return renamed.sort_values(
        ["sum_similarity_delta", "row_count_delta", "max_similarity_delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def parse_shard_path(shard_path: str | Path) -> tuple[str, str]:
    """Extract `(city1, city2)` from an optimized temp shard path."""
    path = Path(shard_path)
    city2_dir = path.parent
    city1_dir = city2_dir.parent
    return city1_dir.name.split("=", 1)[1], city2_dir.name.split("=", 1)[1]


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

    available_city2_values: list[str] = []
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


def sql_quote(value: str) -> str:
    """Return a SQL-safe string literal."""
    return value.replace("'", "''")


def build_union_query(shard_paths: list[str]) -> str:
    """Return a `UNION ALL` query over selected temp shards."""
    query_parts = []
    for shard_path in shard_paths:
        shard_city1, shard_city2 = parse_shard_path(shard_path)
        query_parts.append(
            f"""
            SELECT
                hex_id1,
                hex_id2,
                similarity,
                city1 AS row_city1,
                city2 AS row_city2,
                '{sql_quote(shard_city1)}' AS shard_city1,
                '{sql_quote(shard_city2)}' AS shard_city2
            FROM read_parquet('{sql_quote(shard_path)}')
            """
        )
    return " UNION ALL ".join(query_parts)


def run_current_logic(con: Any, union_query: str) -> pd.DataFrame:
    """Emulate the current B5c optimized aggregation logic."""
    query = f"""
    WITH shard_union AS ({union_query}),
    deduped AS (
        SELECT
            LEAST(hex_id1, hex_id2) AS hex_id1,
            GREATEST(hex_id1, hex_id2) AS hex_id2,
            MAX(similarity) AS similarity,
            shard_city1 AS city_1,
            shard_city2 AS city_2
        FROM shard_union
        GROUP BY
            LEAST(hex_id1, hex_id2),
            GREATEST(hex_id1, hex_id2),
            shard_city1,
            shard_city2
    )
    SELECT
        city_1,
        city_2,
        COUNT(*) AS row_count,
        MAX(similarity) AS max_similarity,
        AVG(similarity) AS avg_similarity,
        SUM(similarity) AS sum_similarity
    FROM deduped
    WHERE city_1 != city_2
    GROUP BY city_1, city_2
    ORDER BY city_1, city_2
    """
    return con.execute(query).df()


def run_fixed_logic(con: Any, union_query: str) -> pd.DataFrame:
    """Aggregate using the row-level city labels already stored in each shard."""
    query = f"""
    WITH shard_union AS ({union_query}),
    labeled AS (
        SELECT
            LEAST(hex_id1, hex_id2) AS hex_id1,
            GREATEST(hex_id1, hex_id2) AS hex_id2,
            similarity,
            CASE
                WHEN row_city1 <= row_city2 THEN row_city1
                ELSE row_city2
            END AS city_1,
            CASE
                WHEN row_city1 <= row_city2 THEN row_city2
                ELSE row_city1
            END AS city_2
        FROM shard_union
    ),
    deduped AS (
        SELECT
            hex_id1,
            hex_id2,
            city_1,
            city_2,
            MAX(similarity) AS similarity
        FROM labeled
        GROUP BY hex_id1, hex_id2, city_1, city_2
    )
    SELECT
        city_1,
        city_2,
        COUNT(*) AS row_count,
        MAX(similarity) AS max_similarity,
        AVG(similarity) AS avg_similarity,
        SUM(similarity) AS sum_similarity
    FROM deduped
    WHERE city_1 != city_2
    GROUP BY city_1, city_2
    ORDER BY city_1, city_2
    """
    return con.execute(query).df()


def run_leakage_summary(con: Any, union_query: str) -> pd.DataFrame:
    """Summarize same-city rows sitting inside inter-city shard directories."""
    query = f"""
    WITH shard_union AS ({union_query})
    SELECT
        shard_city1,
        shard_city2,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN row_city1 = row_city2 THEN 1 ELSE 0 END) AS same_city_rows,
        SUM(CASE WHEN row_city1 != row_city2 THEN 1 ELSE 0 END) AS true_intercity_rows,
        MAX(CASE WHEN row_city1 = row_city2 THEN similarity ELSE NULL END) AS same_city_max_similarity,
        MAX(CASE WHEN row_city1 != row_city2 THEN similarity ELSE NULL END) AS true_intercity_max_similarity,
        SUM(CASE WHEN row_city1 = row_city2 THEN similarity ELSE 0 END) AS same_city_sum_similarity,
        SUM(CASE WHEN row_city1 != row_city2 THEN similarity ELSE 0 END) AS true_intercity_sum_similarity
    FROM shard_union
    WHERE shard_city1 != shard_city2
    GROUP BY shard_city1, shard_city2
    ORDER BY same_city_sum_similarity DESC, same_city_rows DESC
    """
    return con.execute(query).df()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare current B5c optimized aggregation against row-label aggregation."
    )
    parser.add_argument(
        "--pairwise-root",
        default=DEFAULT_PAIRWISE_ROOT,
        help="Root folder containing optimized/temp shards.",
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
        "--city1",
        default=None,
        help="Optional filter for shard directory city1.",
    )
    parser.add_argument(
        "--city2",
        default=None,
        help="Optional filter for shard directory city2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of shard files to inspect.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for CSV outputs. Writes '<prefix>_diff.csv' and '<prefix>_leakage.csv'.",
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

    union_query = build_union_query(shard_paths)
    con = duckdb.connect()
    try:
        current_df = run_current_logic(con, union_query)
        fixed_df = run_fixed_logic(con, union_query)
        diff_df = build_metric_diff(current_df, fixed_df)
        leakage_df = run_leakage_summary(con, union_query)
    finally:
        con.close()

    print(
        f"shards_scanned={len(shard_paths)} "
        f"current_pairs={len(current_df)} fixed_pairs={len(fixed_df)}"
    )
    print("\nTop pair-level deltas")
    if diff_df.empty:
        print("No inter-city rows found.")
    else:
        print(
            diff_df[
                [
                    "city_1",
                    "city_2",
                    "current_row_count",
                    "fixed_row_count",
                    "row_count_delta",
                    "row_count_ratio",
                    "current_max_similarity",
                    "fixed_max_similarity",
                    "max_similarity_delta",
                    "current_sum_similarity",
                    "fixed_sum_similarity",
                    "sum_similarity_delta",
                ]
            ]
            .head(25)
            .to_string(index=False)
        )

    print("\nTop shard-level leakage")
    if leakage_df.empty:
        print("No inter-city shard leakage found.")
    else:
        print(
            leakage_df[
                [
                    "shard_city1",
                    "shard_city2",
                    "total_rows",
                    "same_city_rows",
                    "true_intercity_rows",
                    "same_city_max_similarity",
                    "true_intercity_max_similarity",
                    "same_city_sum_similarity",
                    "true_intercity_sum_similarity",
                ]
            ]
            .head(25)
            .to_string(index=False)
        )

    if args.output_prefix:
        diff_path = f"{args.output_prefix}_diff.csv"
        leakage_path = f"{args.output_prefix}_leakage.csv"
        diff_df.to_csv(diff_path, index=False)
        leakage_df.to_csv(leakage_path, index=False)
        print(f"\nWrote CSVs: {diff_path}, {leakage_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
