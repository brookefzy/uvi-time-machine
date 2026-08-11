#!/usr/bin/env python3
"""Sample exact cross-city DINOv3 image pairs with FAISS IndexFlatIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_similar_pairs.common import (
    load_city_embeddings,
    write_parquet_with_json_audit,
)
from sample_similar_pairs.image_pair_pipeline import (
    _city_pair_key,
    _faiss,
    apply_image_diversity_caps,
    run_image_pair_pipeline,
    search_image_pair,
    select_mmr_image_pairs,
)

DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
DEFAULT_CORE_H3_POOL_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_core_hex_ids"
DEFAULT_CORE_H3_PROFILE = "pct5_sub30_z1_m05"
DEFAULT_PAIRS = ["Paris|London", "London|Hong Kong", "Hong Kong|Singapore", "London|Sydney", "New York|London"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-pairs", nargs="+", default=DEFAULT_PAIRS, help="Directed CITY1|CITY2 pairs")
    parser.add_argument("--embedding-root", type=Path, default=Path(f"{DEFAULT_ROOT}/_curated/c_city_dinov3_embed"))
    parser.add_argument("--rootfolder", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--train-test-folder", type=Path, default=Path(f"{DEFAULT_ROOT}/_transformed/t_classifier_img_yolo8"))
    parser.add_argument("--res-exclude", default="None")
    parser.add_argument("--min-year", type=int, default=2012)
    parser.add_argument("--max-year", type=int, default=2022)
    parser.add_argument("--h3-resolution", type=int, default=8)
    parser.add_argument("--core-h3-pool-root", type=Path, default=Path(DEFAULT_CORE_H3_POOL_ROOT), help="Core-H3 pool root; use 'none' to disable")
    parser.add_argument("--core-h3-profile", default=DEFAULT_CORE_H3_PROFILE)
    parser.add_argument("--max-images-per-h3", type=int, default=100)
    parser.add_argument("--max-images-per-city", type=int, default=0, help="Optional total city cap; 0 keeps all sampled H3 cells")
    parser.add_argument("--top-k", type=int, default=30, help="FAISS neighbors retrieved per source image before global reranking")
    parser.add_argument("--threshold", type=float, default=-1.0, help="Minimum cosine; -1.0 retains all candidates before ranking")
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--max-pairs-per-source-image", type=int, default=1, help="Hard diversity cap; 0 disables it")
    parser.add_argument("--max-pairs-per-hex-pair", type=int, default=1, help="Hard diversity cap; 0 disables it")
    parser.add_argument("--mmr-candidate-pool", type=int, default=200, help="High-score candidates retained for MMR reranking")
    parser.add_argument("--mmr-relevance-weight", type=float, default=0.7, help="MMR weight for cosine score versus visual novelty")
    parser.add_argument("--pairs-per-city-pair", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result, audit = run_image_pair_pipeline(
        args=args,
        vector_loader=lambda city: load_city_embeddings(args.embedding_root, city),
        modality="dinov3",
        method_description=(
            "faiss.IndexFlatIP exact DINOv3 cosine over deterministic spatial samples, "
            "then MMR scene-diversity reranking"
        ),
        vector_schema_id="dinov3-embedding-columns-v1",
        vector_root=args.embedding_root,
    )
    write_parquet_with_json_audit(result, args.output, audit)


if __name__ == "__main__":
    main()
