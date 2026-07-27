#!/usr/bin/env python3
"""Sample exact cross-city DINOv3 H3 pairs with FAISS IndexFlatIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinov3_utils import discover_embedding_columns
from sample_similar_pairs.common import CityVectors, parse_city_pairs, write_parquet_with_json_audit
from sample_similar_pairs.sample_image_pairs_faiss import DEFAULT_PAIRS, _city_pair_key, _faiss

DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"


def load_h3_vectors(h3_root: Path | str, city: str, input_template: str, resolution: int) -> CityVectors:
    path = Path(h3_root) / input_template.format(city=city)
    if not path.exists():
        raise FileNotFoundError(f"H3 vector input is missing for {city!r}: {path}")
    frame = pd.read_parquet(path)
    required = {"hex_id", "res", "img_count", "embedding_dim"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"H3 vector input for {city!r} is missing columns: {sorted(missing)}")
    frame = frame[frame["res"] == resolution].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"H3 vector input for {city!r} has no res={resolution} rows")
    if frame["hex_id"].duplicated().any():
        raise ValueError(f"H3 vector input for {city!r} has duplicate hex_id values")
    if frame["embedding_dim"].nunique() != 1:
        raise ValueError(f"H3 vector input for {city!r} has mixed embedding_dim values")
    columns = discover_embedding_columns(frame)
    if len(columns) != int(frame["embedding_dim"].iloc[0]):
        raise ValueError(f"H3 vector input for {city!r} has an incompatible vector schema")
    vectors = np.ascontiguousarray(frame[columns].to_numpy(dtype=np.float32), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 1e-12) or not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"H3 vector input for {city!r} must contain L2-normalized finite vectors")
    return CityVectors(city, frame.drop(columns=columns), columns, vectors)


def search_h3_pair(
    source: CityVectors,
    target: CityVectors,
    *,
    top_k: int,
    threshold: float,
    query_batch_size: int,
    faiss_module: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if source.city == target.city:
        raise ValueError("source and target cities must differ")
    if source.vector_columns != target.vector_columns:
        raise ValueError(f"incompatible vector schema for {source.city!r} and {target.city!r}")
    faiss_module = faiss_module or _faiss()
    index = faiss_module.IndexIDMap2(faiss_module.IndexFlatIP(target.vectors.shape[1]))
    index.add_with_ids(target.vectors, np.arange(len(target.metadata), dtype=np.int64))
    top_k = min(top_k, len(target.metadata))
    records: list[dict[str, object]] = []
    for start in range(0, len(source.metadata), query_batch_size):
        scores, found_ids = index.search(source.vectors[start : start + query_batch_size], top_k)
        for source_offset, (score_row, id_row) in enumerate(zip(scores, found_ids)):
            source_row = source.metadata.iloc[start + source_offset]
            for score, found_id in zip(score_row, id_row):
                if int(found_id) < 0 or float(score) < threshold:
                    continue
                target_row = target.metadata.iloc[int(found_id)]
                records.append({
                    "city_1": source.city, "hex_id_1": source_row["hex_id"], "img_count_1": int(source_row["img_count"]),
                    "city_2": target.city, "hex_id_2": target_row["hex_id"], "img_count_2": int(target_row["img_count"]),
                    "cosine_similarity": float(score), "city_pair_key": _city_pair_key(source.city, target.city),
                })
    result = pd.DataFrame(records)
    return result, {"queried_rows": len(source.metadata), "retrieved_candidates": len(source.metadata) * top_k, "threshold_hits": len(result)}


def apply_h3_caps(candidates: pd.DataFrame, pairs_per_city_pair: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    if pairs_per_city_pair < 1:
        raise ValueError("pairs_per_city_pair must be positive")
    working = candidates.copy()
    working["_key"] = working.apply(lambda row: "|".join(sorted((str(row.hex_id_1), str(row.hex_id_2)))), axis=1)
    working = working.sort_values(["cosine_similarity", "hex_id_1", "hex_id_2"], ascending=[False, True, True], kind="stable")
    return working.drop_duplicates("_key").head(pairs_per_city_pair).drop(columns="_key").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--h3-root", type=Path, default=Path(f"{DEFAULT_ROOT}/_curated/c_city_dinov3_hex_summary"))
    parser.add_argument("--input-template", default="dinov3_city={city}_res_exclude=None.parquet")
    parser.add_argument("--h3-resolution", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--pairs-per-city-pair", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pairs = parse_city_pairs(args.city_pairs)
    cities = sorted({city for pair in pairs for city in pair})
    loaded = {city: load_h3_vectors(args.h3_root, city, args.input_template, args.h3_resolution) for city in cities}
    outputs: list[pd.DataFrame] = []
    audit: dict[str, object] = {"method": "faiss.IndexFlatIP exact H3 cosine", "threshold": args.threshold, "h3_resolution": args.h3_resolution, "city_pairs": {}}
    for source_city, target_city in pairs:
        candidates, stats = search_h3_pair(loaded[source_city], loaded[target_city], top_k=args.top_k, threshold=args.threshold, query_batch_size=args.query_batch_size)
        accepted = apply_h3_caps(candidates, args.pairs_per_city_pair)
        outputs.append(accepted)
        audit["city_pairs"][f"{source_city}|{target_city}"] = {**stats, "source_h3_rows": len(loaded[source_city].metadata), "target_h3_rows": len(loaded[target_city].metadata), "accepted_pairs": len(accepted)}
    columns = ["city_1", "hex_id_1", "img_count_1", "city_2", "hex_id_2", "img_count_2", "cosine_similarity", "city_pair_key"]
    write_parquet_with_json_audit(pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=columns), args.output, audit)


if __name__ == "__main__":
    main()
