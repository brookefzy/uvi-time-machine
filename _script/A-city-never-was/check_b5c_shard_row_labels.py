#!/usr/bin/env python3
"""
Inspect optimized B5b temp shards for row-level city-label contamination.

This checks the exact hypothesis that one shard directory like
`city1=Alpha/city2=Beta/part_res=8.parquet` can contain mixed row labels:
- Alpha / Alpha
- Alpha / Beta
- Beta / Alpha
- Beta / Beta

If that happens, B5c's current optimized aggregation logic is unsafe because it
relabels every row in that shard as the directory-level `(Alpha, Beta)` pair.
"""

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_PAIRWISE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_similarity_by_pair"
)


def canonicalize_city_pair(city1: str, city2: str) -> tuple[str, str]:
    """Return a stable city-pair ordering."""
    return tuple(sorted((str(city1), str(city2))))


def parse_shard_path(shard_path: str | Path) -> tuple[str, str]:
    """Extract `(city1, city2)` from an optimized temp shard path."""
    path = Path(shard_path)
    city2_dir = path.parent
    city1_dir = city2_dir.parent
    return city1_dir.name.split("=", 1)[1], city2_dir.name.split("=", 1)[1]


def summarize_grouped_label_rows(
    grouped_rows: pd.DataFrame, shard_city1: str, shard_city2: str
) -> dict[str, int | str]:
    """Summarize how a shard's row-level city labels compare to its directory pair."""
    shard_pair = canonicalize_city_pair(shard_city1, shard_city2)

    total_rows = int(grouped_rows["row_count"].sum()) if not grouped_rows.empty else 0
    same_city_mask = grouped_rows["row_city1"] == grouped_rows["row_city2"]
    true_intercity_mask = ~same_city_mask
    shard_pair_mask = grouped_rows.apply(
        lambda row: canonicalize_city_pair(row["row_city1"], row["row_city2"])
        == shard_pair,
        axis=1,
    )

    rows_matching_shard_pair = int(
        grouped_rows.loc[shard_pair_mask & true_intercity_mask, "row_count"].sum()
    )
    same_city_rows = int(grouped_rows.loc[same_city_mask, "row_count"].sum())
    true_intercity_rows = int(grouped_rows.loc[true_intercity_mask, "row_count"].sum())
    misclassified_rows = total_rows - rows_matching_shard_pair
    same_city_rows_city1 = int(
        grouped_rows.loc[
            (grouped_rows["row_city1"] == shard_city1)
            & (grouped_rows["row_city2"] == shard_city1),
            "row_count",
        ].sum()
    )
    same_city_rows_city2 = int(
        grouped_rows.loc[
            (grouped_rows["row_city1"] == shard_city2)
            & (grouped_rows["row_city2"] == shard_city2),
            "row_count",
        ].sum()
    )
    unexpected_label_rows = int(
        grouped_rows.loc[~shard_pair_mask & ~same_city_mask, "row_count"].sum()
    )

    return {
        "shard_city1": shard_city1,
        "shard_city2": shard_city2,
        "total_rows": total_rows,
        "true_intercity_rows": true_intercity_rows,
        "same_city_rows": same_city_rows,
        "same_city_rows_city1": same_city_rows_city1,
        "same_city_rows_city2": same_city_rows_city2,
        "rows_matching_shard_pair": rows_matching_shard_pair,
        "misclassified_rows": misclassified_rows,
        "unexpected_label_rows": unexpected_label_rows,
    }


def collect_shard_paths(
    pairwise_root: str,
    resolution: int,
    city1_filter: str | None,
    city2_filter: str | None,
    limit: int | None,
) -> list[str]:
    """Return matching optimized temp shard parquet paths."""
    pattern = (
        Path(pairwise_root)
        / "optimized"
        / "temp"
        / f"city1={city1_filter or '*'}"
        / f"city2={city2_filter or '*'}"
        / f"part_res={resolution}.parquet"
    )
    paths = sorted(glob(str(pattern)))
    if limit is not None:
        return paths[:limit]
    return paths


def load_grouped_label_rows(con: Any, shard_path: str) -> pd.DataFrame:
    """Return one row per `(row_city1, row_city2)` label combination for a shard."""
    query = f"""
    SELECT
        city1 AS row_city1,
        city2 AS row_city2,
        COUNT(*) AS row_count
    FROM read_parquet('{str(shard_path).replace("'", "''")}')
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return con.execute(query).df()


def run_check(
    pairwise_root: str,
    resolution: int,
    city1_filter: str | None,
    city2_filter: str | None,
    limit: int | None,
) -> pd.DataFrame:
    """Run the shard label check and return one summary row per shard."""
    shard_paths = collect_shard_paths(
        pairwise_root=pairwise_root,
        resolution=resolution,
        city1_filter=city1_filter,
        city2_filter=city2_filter,
        limit=limit,
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No temp shards found under {pairwise_root} for res={resolution}"
        )

    con = duckdb.connect()
    try:
        rows: list[dict[str, int | str]] = []
        for shard_path in shard_paths:
            shard_city1, shard_city2 = parse_shard_path(shard_path)
            grouped_rows = load_grouped_label_rows(con, shard_path)
            summary = summarize_grouped_label_rows(
                grouped_rows=grouped_rows,
                shard_city1=shard_city1,
                shard_city2=shard_city2,
            )
            summary["shard_path"] = shard_path
            rows.append(summary)
    finally:
        con.close()

    return pd.DataFrame(rows).sort_values(
        ["misclassified_rows", "same_city_rows", "total_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect B5b temp shards for row-level city-label leakage."
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
        "--only-problematic",
        action="store_true",
        help="Only print shards with misclassified rows.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the full shard summary CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    summary_df = run_check(
        pairwise_root=args.pairwise_root,
        resolution=args.resolution,
        city1_filter=args.city1,
        city2_filter=args.city2,
        limit=args.limit,
    )

    print(
        f"shards_scanned={len(summary_df)} "
        f"problematic_shards={(summary_df['misclassified_rows'] > 0).sum()} "
        f"total_rows={int(summary_df['total_rows'].sum())} "
        f"total_misclassified_rows={int(summary_df['misclassified_rows'].sum())}"
    )

    display_df = summary_df
    if args.only_problematic:
        display_df = summary_df[summary_df["misclassified_rows"] > 0].reset_index(
            drop=True
        )

    if display_df.empty:
        print("No problematic shards found.")
    else:
        print(
            display_df[
                [
                    "shard_city1",
                    "shard_city2",
                    "total_rows",
                    "true_intercity_rows",
                    "same_city_rows",
                    "same_city_rows_city1",
                    "same_city_rows_city2",
                    "misclassified_rows",
                    "unexpected_label_rows",
                ]
            ].to_string(index=False)
        )

    if args.output_csv:
        summary_df.to_csv(args.output_csv, index=False)
        print(f"\nWrote CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
