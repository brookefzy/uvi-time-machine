#!/usr/bin/env python3
"""Sample a deterministic, balanced DINOv3 image pool for one city."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_similar_pairs.common import CityVectors
from stage2_dino_modality.common import (
    attach_image_geography,
    load_city_embeddings,
    sample_per_hex,
    write_parquet_with_json_audit,
)


def sample_city(city_vectors: CityVectors, max_images_per_h3: int, before_rows: int | None = None) -> tuple[CityVectors, dict[str, int]]:
    """Select lexical image names per H3 cell while preserving their vectors."""
    metadata = city_vectors.metadata.assign(_row=np.arange(len(city_vectors.metadata)))
    sampled_metadata = sample_per_hex(metadata, max_images_per_h3)
    sampled = city_vectors.take(sampled_metadata["_row"].to_numpy(dtype=int))
    h3_sizes = metadata.groupby("hex_id", dropna=False).size()
    return sampled, {
        "before_rows": int(len(city_vectors.metadata) if before_rows is None else before_rows),
        "eligible_rows": int(len(city_vectors.metadata)),
        "sampled_rows": int(len(sampled.metadata)),
        "undersupplied_h3_count": int((h3_sizes < max_images_per_h3).sum()),
    }


def sampled_frame(city_vectors: CityVectors):
    frame = city_vectors.metadata.copy()
    frame.insert(0, "city", city_vectors.city)
    frame["res"] = frame.get("res", 8)
    for index, column in enumerate(city_vectors.vector_columns):
        frame[column] = city_vectors.vectors[:, index]
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--rootfolder", type=Path, required=True)
    parser.add_argument("--train-test-folder", type=Path, default=None)
    parser.add_argument("--res-exclude", type=int, default=None)
    parser.add_argument("--min-year", type=int, default=2012)
    parser.add_argument("--max-year", type=int, default=2022)
    parser.add_argument("--h3-resolution", type=int, default=8)
    parser.add_argument("--max-images-per-h3", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = load_city_embeddings(args.embedding_root, args.city)
    eligible = attach_image_geography(
        loaded,
        args.rootfolder,
        args.train_test_folder,
        min_year=args.min_year,
        max_year=args.max_year,
        h3_resolution=args.h3_resolution,
        res_exclude=args.res_exclude,
    )
    sampled, audit = sample_city(eligible, args.max_images_per_h3, before_rows=len(loaded.metadata))
    audit.update({"city": args.city, "h3_resolution": args.h3_resolution, "max_images_per_h3": args.max_images_per_h3})
    write_parquet_with_json_audit(sampled_frame(sampled), args.output, audit)


if __name__ == "__main__":
    main()
