#!/usr/bin/env python3
"""Write deterministic pairs for cities with usable H3 DINOv3 inputs."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-meta", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-template", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = pd.read_csv(args.city_meta)
    if "City" not in meta:
        raise ValueError(f"{args.city_meta} must contain City")
    cities = []
    for city in sorted(meta["City"].dropna().astype(str).unique()):
        path = args.source_root / args.input_template.format(city=city)
        if path.exists() and pd.read_parquet(path, columns=["res"]).loc[lambda df: df["res"].eq(args.resolution)].shape[0]:
            cities.append(city)
    if len(cities) < 2:
        raise ValueError(f"Only {len(cities)} cities have usable res={args.resolution} H3 inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{city1}|{city2}\n" for city1, city2 in combinations(cities, 2)))
    print(f"Wrote {len(cities)} cities and {len(cities) * (len(cities) - 1) // 2} pairs to {args.output}")


if __name__ == "__main__":
    main()
