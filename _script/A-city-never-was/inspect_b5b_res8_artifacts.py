#!/usr/bin/env python3
"""
Inspect B5b res=8 source features and pairwise shards for pathological exact matches.

This script is designed for remote artifact debugging. It does not rerun B5b. It
reads existing source feature parquet files and existing optimized temp shards, then
checks:

1. Whether source feature inputs contain duplicate `hex_id` rows or conflicting
   vectors for the same `hex_id`.
2. Whether exact `similarity == 1` shard rows are explained by exact cross-city
   vector matches.
3. Whether there are any `similarity == 1` rows whose source vectors are not
   identical, which would implicate the similarity-generation path more directly.

Sample remote run for Hong Kong:

    python3 /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/inspect_b5b_res8_artifacts.py \
      --resolution 8 \
      --city1 hongkong \
      --limit 25 \
      --example-limit 25 \
      --output-prefix '/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/res8_b5b_artifact_checks/hongkong_res8'

Smaller quick-check variant:

    python3 /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/inspect_b5b_res8_artifacts.py \
      --resolution 8 \
      --city1 hongkong \
      --limit 5 \
      --example-limit 10 \
      --output-prefix '/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/res8_b5b_artifact_checks/hongkong_res8_small'
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
VECTOR_COLUMNS = [str(i) for i in range(127)]
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


def summarize_city_source_df(city: str, df: pd.DataFrame) -> dict[str, int | str]:
    """Summarize one city's source feature diagnostics from a compact DataFrame."""
    row_count = len(df)
    unique_hex_count = int(df["hex_id"].nunique())
    duplicate_hex_row_count = row_count - unique_hex_count
    duplicate_hex_id_count = int((df.groupby("hex_id").size() > 1).sum())
    multi_vector_hex_count = int(
        (df.groupby("hex_id")["vector_hash"].nunique() > 1).sum()
    )
    unique_vector_count = int(df["vector_hash"].nunique())
    duplicate_vector_row_count = row_count - unique_vector_count
    return {
        "city": city,
        "row_count": row_count,
        "unique_hex_count": unique_hex_count,
        "duplicate_hex_row_count": duplicate_hex_row_count,
        "duplicate_hex_id_count": duplicate_hex_id_count,
        "multi_vector_hex_count": multi_vector_hex_count,
        "unique_vector_count": unique_vector_count,
        "duplicate_vector_row_count": duplicate_vector_row_count,
    }


def summarize_joined_pair_df(
    city_1: str, city_2: str, df: pd.DataFrame
) -> dict[str, int | float | str]:
    """Summarize one joined pair diagnostic sample DataFrame."""
    equal_hash_mask = (
        df["vector_hash1"].notna()
        & df["vector_hash2"].notna()
        & (df["vector_hash1"] == df["vector_hash2"])
    )
    sim_eq_1_mask = df["similarity"] == 1.0
    sim_ge_mask = df["similarity"] >= 0.999999
    return {
        "city_1": city_1,
        "city_2": city_2,
        "pair_row_count": len(df),
        "sim_eq_1_count": int(sim_eq_1_mask.sum()),
        "sim_ge_0_999999_count": int(sim_ge_mask.sum()),
        "equal_hash_pair_count": int(equal_hash_mask.sum()),
        "sim_eq_1_equal_hash_count": int((sim_eq_1_mask & equal_hash_mask).sum()),
        "sim_eq_1_unequal_hash_count": int((sim_eq_1_mask & ~equal_hash_mask).sum()),
    }


def sql_quote(value: str) -> str:
    """Return a SQL-safe string literal."""
    return value.replace("'", "''")


def build_vector_hash_sql() -> str:
    """Return a DuckDB expression hashing all 127 feature columns."""
    parts = ", ".join(
        [f"CAST(COALESCE(\"{column}\", 0) AS VARCHAR)" for column in VECTOR_COLUMNS]
    )
    return f"md5(concat_ws('|', {parts}))"


def build_source_query(source_path: str, resolution: int) -> str:
    """Return the SQL subquery for one city's source features."""
    vector_hash_sql = build_vector_hash_sql()
    return f"""
        SELECT
            CAST(hex_id AS VARCHAR) AS hex_id,
            {vector_hash_sql} AS vector_hash
        FROM read_parquet('{sql_quote(source_path)}')
        WHERE res = {int(resolution)}
    """


def get_source_path(source_root: str, city: str, res_exclude: int) -> str:
    """Return one city's source parquet path."""
    return str(Path(source_root) / f"prob_city={city}_res_exclude={res_exclude}.parquet")


def get_city_source_summary(
    con: Any, source_root: str, city: str, resolution: int, res_exclude: int
) -> dict[str, int | str]:
    """Aggregate one city's source feature diagnostics in SQL."""
    source_path = get_source_path(source_root, city, res_exclude)
    query = f"""
    WITH source AS (
        {build_source_query(source_path, resolution)}
    ),
    dup_hex AS (
        SELECT hex_id, COUNT(*) AS n_rows, COUNT(DISTINCT vector_hash) AS n_vectors
        FROM source
        GROUP BY hex_id
    )
    SELECT
        COUNT(*) AS row_count,
        COUNT(DISTINCT hex_id) AS unique_hex_count,
        COUNT(*) - COUNT(DISTINCT hex_id) AS duplicate_hex_row_count,
        COALESCE(SUM(CASE WHEN d.n_rows > 1 THEN 1 ELSE 0 END), 0) AS duplicate_hex_id_count,
        COALESCE(SUM(CASE WHEN d.n_vectors > 1 THEN 1 ELSE 0 END), 0) AS multi_vector_hex_count,
        COUNT(DISTINCT vector_hash) AS unique_vector_count,
        COUNT(*) - COUNT(DISTINCT vector_hash) AS duplicate_vector_row_count
    FROM source s
    CROSS JOIN (
        SELECT
            COALESCE(SUM(CASE WHEN n_rows > 1 THEN 1 ELSE 0 END), 0) AS duplicate_hex_id_count,
            COALESCE(SUM(CASE WHEN n_vectors > 1 THEN 1 ELSE 0 END), 0) AS multi_vector_hex_count
        FROM dup_hex
    ) d
    """
    row_count, unique_hex_count, duplicate_hex_row_count, duplicate_hex_id_count, multi_vector_hex_count, unique_vector_count, duplicate_vector_row_count = con.execute(
        query
    ).fetchone()
    return {
        "city": city,
        "row_count": int(row_count),
        "unique_hex_count": int(unique_hex_count),
        "duplicate_hex_row_count": int(duplicate_hex_row_count),
        "duplicate_hex_id_count": int(duplicate_hex_id_count),
        "multi_vector_hex_count": int(multi_vector_hex_count),
        "unique_vector_count": int(unique_vector_count),
        "duplicate_vector_row_count": int(duplicate_vector_row_count),
    }


def get_pair_summary(
    con: Any,
    shard_path: str,
    source_root: str,
    resolution: int,
    res_exclude: int,
    city_1: str,
    city_2: str,
) -> dict[str, int | float | str]:
    """Aggregate one shard's diagnostic metrics in SQL."""
    left_source_query = build_source_query(
        get_source_path(source_root, city_1, res_exclude), resolution
    )
    right_source_query = build_source_query(
        get_source_path(source_root, city_2, res_exclude), resolution
    )
    query = f"""
    WITH left_source AS (
        {left_source_query}
    ),
    right_source AS (
        {right_source_query}
    ),
    left_hash_counts AS (
        SELECT vector_hash, COUNT(*) AS n_left
        FROM left_source
        GROUP BY vector_hash
    ),
    right_hash_counts AS (
        SELECT vector_hash, COUNT(*) AS n_right
        FROM right_source
        GROUP BY vector_hash
    ),
    equal_hash_pairs AS (
        SELECT COALESCE(SUM(l.n_left * r.n_right), 0) AS exact_cross_city_vector_pair_count
        FROM left_hash_counts l
        JOIN right_hash_counts r USING (vector_hash)
    ),
    shard AS (
        SELECT
            CAST(hex_id1 AS VARCHAR) AS hex_id1,
            CAST(hex_id2 AS VARCHAR) AS hex_id2,
            similarity
        FROM read_parquet('{sql_quote(shard_path)}')
    ),
    joined AS (
        SELECT
            s.hex_id1,
            s.hex_id2,
            s.similarity,
            l.vector_hash AS vector_hash1,
            r.vector_hash AS vector_hash2
        FROM shard s
        LEFT JOIN left_source l
            ON s.hex_id1 = l.hex_id
        LEFT JOIN right_source r
            ON s.hex_id2 = r.hex_id
    )
    SELECT
        COUNT(*) AS pair_row_count,
        SUM(CASE WHEN similarity = 1.0 THEN 1 ELSE 0 END) AS sim_eq_1_count,
        SUM(CASE WHEN similarity >= 0.999999 THEN 1 ELSE 0 END) AS sim_ge_0_999999_count,
        SUM(
            CASE
                WHEN vector_hash1 IS NOT NULL AND vector_hash2 IS NOT NULL AND vector_hash1 = vector_hash2
                THEN 1 ELSE 0
            END
        ) AS equal_hash_pair_count,
        SUM(
            CASE
                WHEN similarity = 1.0
                 AND vector_hash1 IS NOT NULL
                 AND vector_hash2 IS NOT NULL
                 AND vector_hash1 = vector_hash2
                THEN 1 ELSE 0
            END
        ) AS sim_eq_1_equal_hash_count,
        SUM(
            CASE
                WHEN similarity = 1.0
                 AND (
                     vector_hash1 IS NULL OR vector_hash2 IS NULL OR vector_hash1 != vector_hash2
                 )
                THEN 1 ELSE 0
            END
        ) AS sim_eq_1_unequal_hash_count,
        AVG(similarity) AS avg_similarity,
        MAX(similarity) AS max_similarity,
        quantile_cont(similarity, 0.5) AS p50_similarity,
        quantile_cont(similarity, 0.9) AS p90_similarity,
        quantile_cont(similarity, 0.99) AS p99_similarity,
        (SELECT exact_cross_city_vector_pair_count FROM equal_hash_pairs) AS exact_cross_city_vector_pair_count
    FROM joined
    """
    (
        pair_row_count,
        sim_eq_1_count,
        sim_ge_count,
        equal_hash_pair_count,
        sim_eq_1_equal_hash_count,
        sim_eq_1_unequal_hash_count,
        avg_similarity,
        max_similarity,
        p50_similarity,
        p90_similarity,
        p99_similarity,
        exact_cross_city_vector_pair_count,
    ) = con.execute(query).fetchone()
    return {
        "city_1": city_1,
        "city_2": city_2,
        "pair_row_count": int(pair_row_count),
        "sim_eq_1_count": int(sim_eq_1_count),
        "sim_ge_0_999999_count": int(sim_ge_count),
        "equal_hash_pair_count": int(equal_hash_pair_count),
        "sim_eq_1_equal_hash_count": int(sim_eq_1_equal_hash_count),
        "sim_eq_1_unequal_hash_count": int(sim_eq_1_unequal_hash_count),
        "avg_similarity": float(avg_similarity) if avg_similarity is not None else 0.0,
        "max_similarity": float(max_similarity) if max_similarity is not None else 0.0,
        "p50_similarity": float(p50_similarity) if p50_similarity is not None else 0.0,
        "p90_similarity": float(p90_similarity) if p90_similarity is not None else 0.0,
        "p99_similarity": float(p99_similarity) if p99_similarity is not None else 0.0,
        "exact_cross_city_vector_pair_count": int(exact_cross_city_vector_pair_count),
    }


def get_exact_one_mismatch_examples(
    con: Any,
    shard_path: str,
    source_root: str,
    resolution: int,
    res_exclude: int,
    city_1: str,
    city_2: str,
    example_limit: int,
) -> pd.DataFrame:
    """Return example shard rows with sim=1 but unequal source vector hashes."""
    left_source_query = build_source_query(
        get_source_path(source_root, city_1, res_exclude), resolution
    )
    right_source_query = build_source_query(
        get_source_path(source_root, city_2, res_exclude), resolution
    )
    query = f"""
    WITH left_source AS (
        {left_source_query}
    ),
    right_source AS (
        {right_source_query}
    ),
    shard AS (
        SELECT
            CAST(hex_id1 AS VARCHAR) AS hex_id1,
            CAST(hex_id2 AS VARCHAR) AS hex_id2,
            similarity
        FROM read_parquet('{sql_quote(shard_path)}')
    ),
    joined AS (
        SELECT
            '{sql_quote(city_1)}' AS city_1,
            '{sql_quote(city_2)}' AS city_2,
            s.hex_id1,
            s.hex_id2,
            s.similarity,
            l.vector_hash AS vector_hash1,
            r.vector_hash AS vector_hash2
        FROM shard s
        LEFT JOIN left_source l
            ON s.hex_id1 = l.hex_id
        LEFT JOIN right_source r
            ON s.hex_id2 = r.hex_id
    )
    SELECT *
    FROM joined
    WHERE similarity = 1.0
      AND (
          vector_hash1 IS NULL OR vector_hash2 IS NULL OR vector_hash1 != vector_hash2
      )
    LIMIT {int(example_limit)}
    """
    return con.execute(query).df()


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect B5b artifacts for res=8 exact-match pathologies."
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
        default=25,
        help="Max number of sim=1 unequal-hash example rows to export.",
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
        unique_cities = sorted({city for path in shard_paths for city in parse_shard_path(path)})
        source_rows = [
            get_city_source_summary(
                con=con,
                source_root=args.source_root,
                city=city,
                resolution=args.resolution,
                res_exclude=args.res_exclude,
            )
            for city in unique_cities
        ]
        pair_rows = []
        example_frames = []
        for shard_path in shard_paths:
            city_1, city_2 = parse_shard_path(shard_path)
            pair_rows.append(
                get_pair_summary(
                    con=con,
                    shard_path=shard_path,
                    source_root=args.source_root,
                    resolution=args.resolution,
                    res_exclude=args.res_exclude,
                    city_1=city_1,
                    city_2=city_2,
                )
            )
            examples_df = get_exact_one_mismatch_examples(
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
                example_frames.append(examples_df)
    finally:
        con.close()

    source_df = pd.DataFrame(source_rows).sort_values("city").reset_index(drop=True)
    pair_df = pd.DataFrame(pair_rows).sort_values(
        ["sim_eq_1_unequal_hash_count", "sim_eq_1_count", "pair_row_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    examples_df = (
        pd.concat(example_frames, ignore_index=True)
        if example_frames
        else pd.DataFrame(
            columns=["city_1", "city_2", "hex_id1", "hex_id2", "similarity", "vector_hash1", "vector_hash2"]
        )
    )

    print(f"shards_scanned={len(shard_paths)} cities_scanned={len(source_df)}")
    print("\nSource feature summary")
    print(source_df.to_string(index=False))
    print("\nPair summary")
    print(
        pair_df[
            [
                "city_1",
                "city_2",
                "pair_row_count",
                "sim_eq_1_count",
                "sim_eq_1_equal_hash_count",
                "sim_eq_1_unequal_hash_count",
                "equal_hash_pair_count",
                "exact_cross_city_vector_pair_count",
                "avg_similarity",
                "max_similarity",
                "p50_similarity",
                "p90_similarity",
                "p99_similarity",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    if examples_df.empty:
        print("\nNo sim=1 unequal-hash examples found.")
    else:
        print("\nExample sim=1 unequal-hash rows")
        print(examples_df.head(50).to_string(index=False))

    if args.output_prefix:
        source_path = f"{args.output_prefix}_source_summary.csv"
        pair_path = f"{args.output_prefix}_pair_summary.csv"
        examples_path = f"{args.output_prefix}_sim1_unequal_hash_examples.csv"
        source_df.to_csv(source_path, index=False)
        pair_df.to_csv(pair_path, index=False)
        examples_df.to_csv(examples_path, index=False)
        print(f"\nWrote CSVs: {source_path}, {pair_path}, {examples_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
