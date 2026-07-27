#!/usr/bin/env python3
"""Sample exact cross-city DINOv3 image pairs with FAISS IndexFlatIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_similar_pairs.common import (
    CityVectors,
    attach_image_geography,
    load_city_embeddings,
    parse_city_pairs,
    spatially_sample_city,
    write_parquet_with_json_audit,
)

DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
DEFAULT_PAIRS = ["Paris|London", "London|Hong Kong", "Hong Kong|Singapore", "London|Sydney", "New York|London"]


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required; install faiss-cpu in the remote job environment") from exc
    return faiss


def _city_pair_key(city1: str, city2: str) -> str:
    return "|".join(sorted((city1, city2)))


def search_image_pair(
    source: CityVectors,
    target: CityVectors,
    *,
    top_k: int,
    threshold: float,
    query_batch_size: int,
    faiss_module: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Search exact target-city nearest images for one source city."""
    if source.city == target.city:
        raise ValueError("source and target cities must differ")
    if source.vector_columns != target.vector_columns:
        raise ValueError(f"incompatible vector schema for {source.city!r} and {target.city!r}")
    if top_k < 1 or query_batch_size < 1:
        raise ValueError("top_k and query_batch_size must be positive")
    if target.vectors.size == 0 or source.vectors.size == 0:
        return pd.DataFrame(), {"queried_rows": len(source.metadata), "retrieved_candidates": 0, "threshold_hits": 0}
    faiss_module = faiss_module or _faiss()
    ids = np.arange(len(target.metadata), dtype=np.int64)
    index = faiss_module.IndexIDMap2(faiss_module.IndexFlatIP(target.vectors.shape[1]))
    index.add_with_ids(np.ascontiguousarray(target.vectors, dtype=np.float32), ids)
    top_k = min(top_k, len(target.metadata))
    records: list[dict[str, object]] = []
    retrieved = 0
    for start in range(0, len(source.metadata), query_batch_size):
        scores, found_ids = index.search(source.vectors[start : start + query_batch_size], top_k)
        for local_source, (score_row, id_row) in enumerate(zip(scores, found_ids)):
            source_row = source.metadata.iloc[start + local_source]
            for score, found_id in zip(score_row, id_row):
                if int(found_id) < 0 or float(score) < threshold:
                    continue
                retrieved += 1
                target_row = target.metadata.iloc[int(found_id)]
                records.append(
                    {
                        "city_1": source.city, "name_1": source_row["name"], "panoid_1": source_row["panoid"], "hex_id_1": source_row["hex_id"], "lat_1": float(source_row["lat"]), "lon_1": float(source_row["lon"]),
                        "city_2": target.city, "name_2": target_row["name"], "panoid_2": target_row["panoid"], "hex_id_2": target_row["hex_id"], "lat_2": float(target_row["lat"]), "lon_2": float(target_row["lon"]),
                        "cosine_similarity": float(score), "city_pair_key": _city_pair_key(source.city, target.city),
                    }
                )
    result = pd.DataFrame(records)
    return result, {"queried_rows": len(source.metadata), "retrieved_candidates": len(source.metadata) * top_k, "threshold_hits": len(result)}


def apply_image_diversity_caps(
    candidates: pd.DataFrame,
    *,
    max_pairs_per_source_image: int,
    max_pairs_per_hex_pair: int,
    pairs_per_city_pair: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    if max_pairs_per_source_image < 0 or max_pairs_per_hex_pair < 0 or pairs_per_city_pair < 1:
        raise ValueError("diversity caps cannot be negative and pairs_per_city_pair must be positive")
    working = candidates.copy()
    working["_image_key"] = working.apply(lambda row: "|".join(sorted((str(row.name_1), str(row.name_2)))), axis=1)
    working["_hex_key"] = working.apply(lambda row: "|".join(sorted((str(row.hex_id_1), str(row.hex_id_2)))), axis=1)
    working = working.sort_values(["cosine_similarity", "name_1", "name_2"], ascending=[False, True, True], kind="stable")
    image_seen: set[str] = set()
    source_counts: dict[str, int] = {}
    hex_counts: dict[str, int] = {}
    accepted: list[int] = []
    for index, row in working.iterrows():
        source_name, image_key, hex_key = str(row.name_1), row._image_key, row._hex_key
        source_at_cap = max_pairs_per_source_image > 0 and source_counts.get(source_name, 0) >= max_pairs_per_source_image
        hex_at_cap = max_pairs_per_hex_pair > 0 and hex_counts.get(hex_key, 0) >= max_pairs_per_hex_pair
        if image_key in image_seen or source_at_cap or hex_at_cap:
            continue
        image_seen.add(image_key)
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        hex_counts[hex_key] = hex_counts.get(hex_key, 0) + 1
        accepted.append(index)
        if len(accepted) >= pairs_per_city_pair:
            break
    return working.loc[accepted].drop(columns=["_image_key", "_hex_key"]).reset_index(drop=True)


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
    parser.add_argument("--max-images-per-h3", type=int, default=100)
    parser.add_argument("--max-images-per-city", type=int, default=0, help="Optional total city cap; 0 keeps all sampled H3 cells")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=-1.0, help="Minimum cosine; -1.0 retains all candidates before ranking")
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--max-pairs-per-source-image", type=int, default=0, help="Optional diversity cap; 0 disables it")
    parser.add_argument("--max-pairs-per-hex-pair", type=int, default=0, help="Optional diversity cap; 0 disables it")
    parser.add_argument("--pairs-per-city-pair", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pairs = parse_city_pairs(args.city_pairs)
    res_exclude = None if str(args.res_exclude).lower() in {"none", "null", ""} else int(args.res_exclude)
    cities = sorted({city for pair in pairs for city in pair})
    loaded: dict[str, CityVectors] = {}
    for city in cities:
        vectors = load_city_embeddings(args.embedding_root, city)
        vectors = attach_image_geography(vectors, args.rootfolder, args.train_test_folder, args.min_year, args.max_year, args.h3_resolution, res_exclude)
        loaded[city] = spatially_sample_city(vectors, args.max_images_per_h3, args.max_images_per_city)
    outputs: list[pd.DataFrame] = []
    audit: dict[str, object] = {"method": "faiss.IndexFlatIP exact cosine over deterministic spatial samples", "threshold": args.threshold, "city_pairs": {}}
    for source_city, target_city in pairs:
        candidates, stats = search_image_pair(loaded[source_city], loaded[target_city], top_k=args.top_k, threshold=args.threshold, query_batch_size=args.query_batch_size)
        accepted = apply_image_diversity_caps(candidates, max_pairs_per_source_image=args.max_pairs_per_source_image, max_pairs_per_hex_pair=args.max_pairs_per_hex_pair, pairs_per_city_pair=args.pairs_per_city_pair)
        outputs.append(accepted)
        audit["city_pairs"][f"{source_city}|{target_city}"] = {**stats, "source_sample_rows": len(loaded[source_city].metadata), "target_sample_rows": len(loaded[target_city].metadata), "accepted_pairs": len(accepted)}
    columns = ["city_1", "name_1", "panoid_1", "hex_id_1", "lat_1", "lon_1", "city_2", "name_2", "panoid_2", "hex_id_2", "lat_2", "lon_2", "cosine_similarity", "city_pair_key"]
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=columns)
    write_parquet_with_json_audit(result, args.output, audit)


if __name__ == "__main__":
    main()
