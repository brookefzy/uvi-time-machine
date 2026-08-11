#!/usr/bin/env python3
"""Sample exact cross-city classifier-probability image pairs with FAISS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_similar_pairs.common import (
    load_city_classifier_probabilities,
    write_parquet_with_json_audit,
)
from sample_similar_pairs.image_pair_pipeline import run_image_pair_pipeline


DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
DEFAULT_PROBABILITY_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_classifiier_prob"
DEFAULT_CORE_H3_POOL_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_core_hex_ids"
DEFAULT_CORE_H3_PROFILE = "pct5_sub30_z1_m05"
DEFAULT_VECTOR_SCHEMA_ID = "city-classifier-train4-probabilities-v1"
DEFAULT_PAIRS = [
    "Paris|London",
    "London|Hong Kong",
    "Hong Kong|Singapore",
    "London|Sydney",
    "New York|London",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-pairs", nargs="+", default=DEFAULT_PAIRS, help="Directed CITY1|CITY2 pairs")
    parser.add_argument("--probability-root", type=Path, default=Path(DEFAULT_PROBABILITY_ROOT))
    parser.add_argument("--expected-dim", type=int, default=127)
    parser.add_argument("--vector-schema-id", default=DEFAULT_VECTOR_SCHEMA_ID)
    parser.add_argument("--rootfolder", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument(
        "--train-test-folder",
        type=Path,
        default=Path(f"{DEFAULT_ROOT}/_transformed/t_classifier_img_yolo8"),
    )
    parser.add_argument("--res-exclude", default="None")
    parser.add_argument("--min-year", type=int, default=2012)
    parser.add_argument("--max-year", type=int, default=2022)
    parser.add_argument("--h3-resolution", type=int, default=8)
    parser.add_argument(
        "--core-h3-pool-root",
        type=Path,
        default=Path(DEFAULT_CORE_H3_POOL_ROOT),
        help="Core-H3 pool root; use 'none' to disable",
    )
    parser.add_argument("--core-h3-profile", default=DEFAULT_CORE_H3_PROFILE)
    parser.add_argument("--max-images-per-h3", type=int, default=100)
    parser.add_argument(
        "--max-images-per-city",
        type=int,
        default=0,
        help="Optional total city cap; 0 keeps all sampled H3 cells",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="FAISS neighbors retrieved per source image before global reranking",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Minimum classifier-profile cosine; -1.0 retains all candidates before ranking",
    )
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-pairs-per-source-image",
        type=int,
        default=1,
        help="Hard diversity cap; 0 disables it",
    )
    parser.add_argument(
        "--max-pairs-per-hex-pair",
        type=int,
        default=1,
        help="Hard diversity cap; 0 disables it",
    )
    parser.add_argument(
        "--mmr-candidate-pool",
        type=int,
        default=200,
        help="High-score candidates retained for classifier-profile MMR reranking",
    )
    parser.add_argument(
        "--mmr-relevance-weight",
        type=float,
        default=0.7,
        help="MMR weight for classifier-profile cosine relevance versus profile novelty",
    )
    parser.add_argument("--pairs-per-city-pair", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result, audit = run_image_pair_pipeline(
        args=args,
        vector_loader=lambda city: load_city_classifier_probabilities(
            args.probability_root,
            city,
            expected_dim=args.expected_dim,
            return_stats=True,
        ),
        modality="classifier_probability",
        method_description=(
            "faiss.IndexFlatIP exact cosine over L2-normalized classifier probability profiles "
            "within deterministic spatial samples, then MMR classifier-profile diversity reranking"
        ),
        vector_schema_id=args.vector_schema_id,
        vector_root=args.probability_root,
    )
    audit["expected_dim"] = args.expected_dim
    audit["normalization"] = "row-wise L2 normalization before FAISS inner product"
    write_parquet_with_json_audit(result, args.output, audit)


if __name__ == "__main__":
    main()
