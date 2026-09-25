#!/usr/bin/env python3
"""Export a compact H3/image-count inventory from DINOv3 H3 summaries."""


from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from B5e_dinov3_vector_summary import DEFAULT_OUTPUT_ROOT

DEFAULT_INPUT_GLOB = "dinov3_city=*_res_exclude=None.parquet"
REQUIRED_COLUMNS = ("hex_id", "res", "img_count")
OUTPUT_COLUMNS = ("city", "res", "hex_id", "img_count")
SUMMARY_NAME = re.compile(
    r"^dinov3_city=(?P<city>.+)_res_exclude=[^_]+(?:_sampling=equal)?\.parquet$"
)


def parse_resolutions(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = list(value)
    resolutions = list(dict.fromkeys(int(item) for item in values))
    if not resolutions:
        raise ValueError("At least one H3 resolution is required")
    return resolutions


def city_from_summary_path(path: Path) -> str:
    match = SUMMARY_NAME.match(path.name)
    if match is None:
        raise ValueError(
            f"Cannot derive city from DINOv3 summary filename: {path.name}"
        )
    return match.group("city")


def build_inventory(
    h3_root: str | Path,
    resolutions: Sequence[int] = (6, 7, 8),
    input_glob: str = DEFAULT_INPUT_GLOB,
) -> pd.DataFrame:
    """Read only H3 identifiers/counts and combine all matching city files."""
    root = Path(h3_root)
    wanted_resolutions = parse_resolutions(resolutions)
    paths = sorted(root.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No H3 summaries matched {root / input_glob}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        schema_columns = set(pq.read_schema(path).names)
        missing = sorted(set(REQUIRED_COLUMNS).difference(schema_columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        frame = frame[frame["res"].isin(wanted_resolutions)].copy()
        if frame.empty:
            continue
        frame.insert(0, "city", city_from_summary_path(path))
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"Matched {len(paths)} H3 summary files, but none contained resolutions "
            f"{wanted_resolutions}"
        )

    inventory = pd.concat(frames, ignore_index=True)
    inventory["res"] = pd.to_numeric(inventory["res"], errors="raise").astype("int16")
    inventory["img_count"] = pd.to_numeric(
        inventory["img_count"], errors="raise"
    ).astype("int64")
    if (
        inventory["hex_id"].isna().any()
        or (inventory["hex_id"].astype(str) == "").any()
    ):
        raise ValueError("H3 inventory contains an empty hex_id")
    if (inventory["img_count"] < 0).any():
        raise ValueError("H3 inventory contains a negative img_count")

    key_columns = ["city", "res", "hex_id"]
    duplicate_mask = inventory.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        example = inventory.loc[duplicate_mask, key_columns].iloc[0].to_dict()
        raise ValueError(
            f"H3 inventory contains duplicate city/res/hex_id keys: {example}"
        )

    return inventory.loc[:, OUTPUT_COLUMNS].sort_values(
        key_columns, kind="stable", ignore_index=True
    )


def summarize_image_count_distribution(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return image-count distribution statistics for every city/resolution."""
    grouped = inventory.groupby(["city", "res"], sort=True)["img_count"]
    summary = grouped.agg(
        h3_count="size",
        total_image_count="sum",
        min_image_count="min",
        mean_image_count="mean",
        median_image_count="median",
        max_image_count="max",
    ).reset_index()
    quantiles = (
        grouped.quantile([0.25, 0.75, 0.90, 0.95, 0.99])
        .unstack()
        .rename(
            columns={
                0.25: "p25_image_count",
                0.75: "p75_image_count",
                0.90: "p90_image_count",
                0.95: "p95_image_count",
                0.99: "p99_image_count",
            }
        )
        .reset_index()
    )
    return summary.merge(quantiles, on=["city", "res"], how="left").sort_values(
        ["city", "res"], kind="stable", ignore_index=True
    )


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(output, index=False)
    elif suffix == ".csv":
        frame.to_csv(output, index=False)
    else:
        raise ValueError(f"Output must end in .parquet or .csv: {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-city DINOv3 H3 summaries into a compact H3/image-count "
            "inventory and a city-resolution distribution report."
        )
    )
    parser.add_argument("--h3-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--resolutions", default="6,7,8")
    parser.add_argument("--output-inventory", required=True)
    parser.add_argument("--output-distribution-csv", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolutions = parse_resolutions(args.resolutions)
    inventory = build_inventory(
        args.h3_root,
        resolutions=resolutions,
        input_glob=args.input_glob,
    )
    distribution = summarize_image_count_distribution(inventory)
    write_table(inventory, args.output_inventory)
    write_table(distribution, args.output_distribution_csv)

    print(
        f"Wrote {len(inventory):,} H3 rows from {inventory['city'].nunique():,} cities "
        f"to {args.output_inventory}"
    )
    print(
        f"Wrote {len(distribution):,} distribution rows to {args.output_distribution_csv}"
    )
    print(distribution.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
