#!/usr/bin/env python3
"""Summarize exact H3 mode similarities by unordered city pair."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "city_1",
    "city_2",
    "js_similarity_avg",
    "p50",
    "p90",
    "p95",
    "max",
    "pair_count_observed",
]


def summarize_city_pairs(
    frame: pd.DataFrame, expected_pairs: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    if frame.empty:
        result = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        result = frame.groupby(["city_1", "city_2"], as_index=False).agg(
            js_similarity_avg=("js_similarity", "mean"),
            p50=("js_similarity", lambda values: values.quantile(.5)),
            p90=("js_similarity", lambda values: values.quantile(.9)),
            p95=("js_similarity", lambda values: values.quantile(.95)),
            max=("js_similarity", "max"),
            pair_count_observed=("js_similarity", "size"),
        )
    if expected_pairs is not None:
        expected = pd.DataFrame(expected_pairs, columns=["city_1", "city_2"])
        result = expected.merge(result, on=["city_1", "city_2"], how="left")
        result["pair_count_observed"] = result["pair_count_observed"].fillna(0).astype(int)
    return result


def read_similarity_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise ValueError(f"no similarity Parquet files under {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def read_manifest_similarity_input(
    root: Path, expected_pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Read only shards named by the current manifest, ignoring stale partitions."""
    files = [
        root / f"city_1={left}" / f"city_2={right}" / "part_res=8.parquet"
        for left, right in expected_pairs
    ]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} expected similarity shards; first: {missing[0]}")
    if not files:
        return pd.DataFrame(columns=["city_1", "city_2", "model_id", "js_similarity"])
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def read_pair_manifest(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 2 or not all(parts) or parts[0] >= parts[1]:
            raise ValueError(f"invalid city pair manifest row: {line}")
        pairs.append((parts[0], parts[1]))
    if len(pairs) != len(set(pairs)):
        raise ValueError("city pair manifest contains duplicates")
    return pairs


def validate_expected_pairs(frame: pd.DataFrame, cities: list[str]) -> None:
    expected = set(combinations(sorted(set(cities)), 2))
    observed = set(
        map(tuple, frame[["city_1", "city_2"]].drop_duplicates().itertuples(index=False, name=None))
    )
    if observed != expected:
        raise ValueError(f"expected {len(expected)} unordered city pairs, observed {len(observed)}")
    validate_model_ids(frame)


def validate_manifest_pairs(frame: pd.DataFrame, expected_pairs: list[tuple[str, str]]) -> None:
    expected = set(expected_pairs)
    observed = set(
        map(tuple, frame[["city_1", "city_2"]].drop_duplicates().itertuples(index=False, name=None))
    )
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"similarity input contains unexpected city pairs: {sorted(unexpected)}")
    validate_model_ids(frame)


def validate_model_ids(frame: pd.DataFrame) -> None:
    if "model_id" in frame and frame.model_id.nunique() > 1:
        raise ValueError("similarity shards contain multiple model IDs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--city-meta", type=Path)
    validation.add_argument("--pair-manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected_pairs = None
    if args.pair_manifest:
        expected_pairs = read_pair_manifest(args.pair_manifest)
        frame = read_manifest_similarity_input(args.input, expected_pairs)
        validate_manifest_pairs(frame, expected_pairs)
    else:
        frame = read_similarity_input(args.input)
        if args.city_meta:
            validate_expected_pairs(frame, pd.read_csv(args.city_meta)["City"].dropna().tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summarize_city_pairs(frame, expected_pairs=expected_pairs).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
