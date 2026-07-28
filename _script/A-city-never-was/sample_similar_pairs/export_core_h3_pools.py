#!/usr/bin/env python3
"""Export compact core-H3 pools from per-city POI-tier Parquets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinov3_utils import resolve_city_file_stem


DEFAULT_PROFILE_ID = "pct5_sub30_z1_m05"
DEFAULT_CITIES = ["Paris", "London", "Hong Kong", "Singapore", "Sydney", "New York"]


def export_core_h3_pools(
    source_root: Path | str,
    output_root: Path | str,
    cities: list[str],
    resolution: int = 8,
    profile_id: str = DEFAULT_PROFILE_ID,
    tier_column: str = "tier_pct",
    tier_value: str = "core",
) -> dict[str, object]:
    """Write one deduplicated `hex_id` pool for each requested city."""
    source_root = Path(source_root)
    pool_dir = Path(output_root) / f"res={resolution}" / f"profile={profile_id}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "source_root": str(source_root),
        "resolution": resolution,
        "profile_id": profile_id,
        "tier_column": tier_column,
        "tier_value": tier_value,
        "cities": {},
    }
    for city in cities:
        stem = resolve_city_file_stem(city)
        input_path = source_root / f"{stem}_h3_poi_tiers.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"POI-tier input is missing for {city!r}: {input_path}")
        frame = pd.read_parquet(input_path)
        required = {"h3_index", "resolution", tier_column}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {missing}")
        frame_at_resolution = frame.loc[frame["resolution"].eq(resolution)]
        if frame_at_resolution.empty:
            found = sorted(frame["resolution"].dropna().unique().tolist())
            raise ValueError(f"{input_path} has resolutions {found}, none matching {resolution}")
        pool = (
            frame_at_resolution.loc[frame_at_resolution[tier_column].eq(tier_value), ["h3_index"]]
            .dropna()
            .astype({"h3_index": str})
            .drop_duplicates()
            .rename(columns={"h3_index": "hex_id"})
            .sort_values("hex_id", kind="stable")
            .reset_index(drop=True)
        )
        if pool.empty:
            raise ValueError(f"{input_path} has no {tier_value!r} H3 cells at resolution {resolution}")
        pool.to_parquet(pool_dir / f"{stem}.parquet", index=False)
        audit["cities"][city] = {"input_rows": len(frame), "core_hex_count": len(pool)}
    (pool_dir / "core_h3_pool_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--tier-column", default="tier_pct")
    parser.add_argument("--tier-value", default="core")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = export_core_h3_pools(
        source_root=args.source_root,
        output_root=args.output_root,
        cities=args.cities,
        resolution=args.resolution,
        profile_id=args.profile_id,
        tier_column=args.tier_column,
        tier_value=args.tier_value,
    )
    print(f"Exported {len(audit['cities'])} core-H3 pools")


if __name__ == "__main__":
    main()
