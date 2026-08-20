#!/usr/bin/env python3
"""Create deterministic cross-city mode-histogram pair manifests."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_dino_modality.common import validate_sparse_histogram


def collect_histogram_cities(
    expected_cities: list[str],
    histogram_root: Path,
    allow_missing: bool = False,
    expected_model_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return valid and absent cities while rejecting corrupt or mixed-model inputs."""
    available: list[str] = []
    skipped: list[str] = []
    model_id = None
    for city in sorted(set(expected_cities)):
        path = histogram_root / f"city={city}.parquet"
        if not path.exists():
            if allow_missing:
                skipped.append(city)
                continue
            raise FileNotFoundError(f"missing histogram for city {city}: {path}")
        frame = pd.read_parquet(path)
        if (
            frame.empty
            or frame.res.isna().any()
            or frame.res.nunique(dropna=False) != 1
            or int(frame.res.iloc[0]) != 8
            or frame.model_id.isna().any()
            or frame.model_id.nunique(dropna=False) != 1
        ):
            raise ValueError(f"invalid histogram: {path}")
        if frame.city.nunique(dropna=False) != 1 or frame.city.iloc[0] != city:
            raise ValueError(f"histogram {path} does not contain exactly city {city}")
        validate_sparse_histogram(frame)
        current_model_id = frame.model_id.iloc[0]
        if expected_model_id is not None and current_model_id != expected_model_id:
            raise ValueError(
                f"histogram model ID is {current_model_id}, expected {expected_model_id}: {path}"
            )
        if model_id is not None and current_model_id != model_id:
            raise ValueError("histogram model IDs differ")
        model_id = current_model_id
        available.append(city)
    return available, skipped


def city_pairs(cities: list[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(set(cities)), 2))


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-meta", type=Path, required=True)
    parser.add_argument("--histogram-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--expected-model-id")
    parser.add_argument("--available-cities-output", type=Path)
    parser.add_argument("--skipped-cities-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected = pd.read_csv(args.city_meta)["City"].dropna().astype(str).tolist()
    available, skipped = collect_histogram_cities(
        expected,
        args.histogram_root,
        allow_missing=args.allow_missing,
        expected_model_id=args.expected_model_id,
    )
    pairs = city_pairs(available)
    write_lines(args.output, [f"{left}|{right}" for left, right in pairs])
    if args.available_cities_output:
        write_lines(args.available_cities_output, available)
    if args.skipped_cities_output:
        write_lines(args.skipped_cities_output, skipped)
    if skipped:
        print(
            f"Skipping {len(skipped)} cities without histograms: {', '.join(skipped)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
