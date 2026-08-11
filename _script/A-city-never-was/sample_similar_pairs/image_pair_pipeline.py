"""Modality-neutral exact image-pair search and diversity selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sample_similar_pairs.common import (
    CityVectors,
    attach_image_geography,
    filter_city_to_core_h3_pool,
    parse_city_pairs,
    spatially_sample_city,
)


RESULT_COLUMNS = [
    "city_1", "name_1", "panoid_1", "hex_id_1", "lat_1", "lon_1",
    "city_2", "name_2", "panoid_2", "hex_id_2", "lat_2", "lon_2",
    "cosine_similarity", "city_pair_key",
]


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
        return pd.DataFrame(), {
            "queried_rows": len(source.metadata),
            "retrieved_candidates": 0,
            "threshold_hits": 0,
        }
    faiss_module = faiss_module or _faiss()
    ids = np.arange(len(target.metadata), dtype=np.int64)
    index = faiss_module.IndexIDMap2(faiss_module.IndexFlatIP(target.vectors.shape[1]))
    index.add_with_ids(np.ascontiguousarray(target.vectors, dtype=np.float32), ids)
    top_k = min(top_k, len(target.metadata))
    records: list[dict[str, object]] = []
    for start in range(0, len(source.metadata), query_batch_size):
        scores, found_ids = index.search(source.vectors[start : start + query_batch_size], top_k)
        for local_source, (score_row, id_row) in enumerate(zip(scores, found_ids)):
            source_row = source.metadata.iloc[start + local_source]
            for score, found_id in zip(score_row, id_row):
                if int(found_id) < 0 or float(score) < threshold:
                    continue
                target_row = target.metadata.iloc[int(found_id)]
                records.append(
                    {
                        "city_1": source.city,
                        "name_1": source_row["name"],
                        "panoid_1": source_row["panoid"],
                        "hex_id_1": source_row["hex_id"],
                        "lat_1": float(source_row["lat"]),
                        "lon_1": float(source_row["lon"]),
                        "city_2": target.city,
                        "name_2": target_row["name"],
                        "panoid_2": target_row["panoid"],
                        "hex_id_2": target_row["hex_id"],
                        "lat_2": float(target_row["lat"]),
                        "lon_2": float(target_row["lon"]),
                        "cosine_similarity": float(score),
                        "city_pair_key": _city_pair_key(source.city, target.city),
                        "_source_index": start + local_source,
                        "_target_index": int(found_id),
                    }
                )
    result = pd.DataFrame(records)
    return result, {
        "queried_rows": len(source.metadata),
        "retrieved_candidates": len(source.metadata) * top_k,
        "threshold_hits": len(result),
    }


def apply_image_diversity_caps(
    candidates: pd.DataFrame,
    *,
    max_pairs_per_source_image: int,
    max_pairs_per_hex_pair: int,
    pairs_per_city_pair: int,
    candidate_pool_size: int | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    if max_pairs_per_source_image < 0 or max_pairs_per_hex_pair < 0 or pairs_per_city_pair < 1:
        raise ValueError("diversity caps cannot be negative and pairs_per_city_pair must be positive")
    if candidate_pool_size is not None and candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive when provided")
    working = candidates.copy()
    working["_image_key"] = working.apply(
        lambda row: "|".join(sorted((str(row.name_1), str(row.name_2)))), axis=1
    )
    working["_hex_key"] = working.apply(
        lambda row: "|".join(sorted((str(row.hex_id_1), str(row.hex_id_2)))), axis=1
    )
    working = working.sort_values(
        ["cosine_similarity", "name_1", "name_2"],
        ascending=[False, True, True],
        kind="stable",
    )
    image_seen: set[str] = set()
    source_counts: dict[str, int] = {}
    hex_counts: dict[str, int] = {}
    accepted: list[int] = []
    for index, row in working.iterrows():
        source_name, image_key, hex_key = str(row.name_1), row._image_key, row._hex_key
        source_at_cap = (
            max_pairs_per_source_image > 0
            and source_counts.get(source_name, 0) >= max_pairs_per_source_image
        )
        hex_at_cap = (
            max_pairs_per_hex_pair > 0
            and hex_counts.get(hex_key, 0) >= max_pairs_per_hex_pair
        )
        if image_key in image_seen or source_at_cap or hex_at_cap:
            continue
        image_seen.add(image_key)
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        hex_counts[hex_key] = hex_counts.get(hex_key, 0) + 1
        accepted.append(index)
        if len(accepted) >= (candidate_pool_size or pairs_per_city_pair):
            break
    return working.loc[accepted].drop(columns=["_image_key", "_hex_key"]).reset_index(drop=True)


def select_mmr_image_pairs(
    candidates: pd.DataFrame,
    source: CityVectors,
    target: CityVectors,
    *,
    candidate_pool_size: int,
    relevance_weight: float,
    pairs_per_city_pair: int,
) -> pd.DataFrame:
    """Rerank high-scoring candidates to retain vector-profile diversity."""
    if candidates.empty:
        return candidates.copy()
    if candidate_pool_size < 1 or pairs_per_city_pair < 1:
        raise ValueError("candidate_pool_size and pairs_per_city_pair must be positive")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")
    required_columns = {"_source_index", "_target_index", "cosine_similarity"}
    missing = required_columns.difference(candidates.columns)
    if missing:
        raise ValueError(f"MMR candidates are missing columns: {sorted(missing)}")

    working = candidates.sort_values(
        ["cosine_similarity", "name_1", "name_2"],
        ascending=[False, True, True],
        kind="stable",
    ).head(candidate_pool_size).reset_index(drop=True)
    source_indices = working["_source_index"].to_numpy(dtype=np.int64)
    target_indices = working["_target_index"].to_numpy(dtype=np.int64)
    pair_vectors = source.vectors[source_indices] + target.vectors[target_indices]
    norms = np.linalg.norm(pair_vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("cannot construct an MMR pair embedding from opposite image vectors")
    pair_vectors = pair_vectors / norms[:, np.newaxis]
    relevance = working["cosine_similarity"].to_numpy(dtype=np.float32)

    selected: list[int] = []
    available = np.ones(len(working), dtype=bool)
    while available.any() and len(selected) < pairs_per_city_pair:
        if selected:
            redundancy = (pair_vectors @ pair_vectors[selected].T).max(axis=1)
            mmr_scores = relevance_weight * relevance - (1.0 - relevance_weight) * redundancy
        else:
            mmr_scores = relevance.copy()
        mmr_scores[~available] = -np.inf
        next_index = int(np.argmax(mmr_scores))
        selected.append(next_index)
        available[next_index] = False
    return working.iloc[selected].drop(columns=["_source_index", "_target_index"]).reset_index(drop=True)


def _parse_res_exclude(value: object) -> int | None:
    return None if value is None or str(value).lower() in {"none", "null", ""} else int(value)


def run_image_pair_pipeline(
    *,
    args: Any,
    vector_loader: Callable[[str], CityVectors | tuple[CityVectors, dict[str, object]]],
    modality: str,
    method_description: str,
    vector_schema_id: str,
    vector_root: Path | str,
    faiss_module: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run shared eligibility, exact search, and diversity selection for one modality."""
    pairs = parse_city_pairs(args.city_pairs)
    res_exclude = _parse_res_exclude(args.res_exclude)
    cities = sorted({city for pair in pairs for city in pair})
    loaded: dict[str, CityVectors] = {}
    input_stats: dict[str, dict[str, object]] = {}
    core_pool_stats: dict[str, dict[str, int]] = {}
    for city in cities:
        loaded_value = vector_loader(city)
        if isinstance(loaded_value, tuple):
            vectors, city_input_stats = loaded_value
            input_stats[city] = city_input_stats
        else:
            vectors = loaded_value
        vectors = attach_image_geography(
            vectors,
            args.rootfolder,
            args.train_test_folder,
            args.min_year,
            args.max_year,
            args.h3_resolution,
            res_exclude,
        )
        if str(args.core_h3_pool_root).lower() != "none":
            vectors, core_pool_stats[city] = filter_city_to_core_h3_pool(
                vectors,
                args.core_h3_pool_root,
                args.h3_resolution,
                args.core_h3_profile,
            )
        loaded[city] = spatially_sample_city(
            vectors,
            args.max_images_per_h3,
            args.max_images_per_city,
        )

    outputs: list[pd.DataFrame] = []
    audit: dict[str, object] = {
        "modality": modality,
        "method": method_description,
        "vector_schema_id": vector_schema_id,
        "vector_root": str(vector_root),
        "input_stats": input_stats,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "query_batch_size": args.query_batch_size,
        "max_images_per_h3": args.max_images_per_h3,
        "max_images_per_city": args.max_images_per_city,
        "max_pairs_per_source_image": args.max_pairs_per_source_image,
        "max_pairs_per_hex_pair": args.max_pairs_per_hex_pair,
        "mmr_candidate_pool": args.mmr_candidate_pool,
        "mmr_relevance_weight": args.mmr_relevance_weight,
        "pairs_per_city_pair": args.pairs_per_city_pair,
        "core_h3_pool_root": str(args.core_h3_pool_root),
        "core_h3_profile": args.core_h3_profile,
        "core_h3_pool_stats": core_pool_stats,
        "city_pairs": {},
    }
    pair_audits = audit["city_pairs"]
    assert isinstance(pair_audits, dict)
    for source_city, target_city in pairs:
        candidates, stats = search_image_pair(
            loaded[source_city],
            loaded[target_city],
            top_k=args.top_k,
            threshold=args.threshold,
            query_batch_size=args.query_batch_size,
            faiss_module=faiss_module,
        )
        capped_candidates = apply_image_diversity_caps(
            candidates,
            max_pairs_per_source_image=args.max_pairs_per_source_image,
            max_pairs_per_hex_pair=args.max_pairs_per_hex_pair,
            pairs_per_city_pair=args.pairs_per_city_pair,
            candidate_pool_size=args.mmr_candidate_pool,
        )
        accepted = select_mmr_image_pairs(
            capped_candidates,
            loaded[source_city],
            loaded[target_city],
            candidate_pool_size=args.mmr_candidate_pool,
            relevance_weight=args.mmr_relevance_weight,
            pairs_per_city_pair=args.pairs_per_city_pair,
        )
        outputs.append(accepted)
        pair_audits[f"{source_city}|{target_city}"] = {
            **stats,
            "source_sample_rows": len(loaded[source_city].metadata),
            "target_sample_rows": len(loaded[target_city].metadata),
            "candidates_after_hard_caps": len(capped_candidates),
            "accepted_pairs": len(accepted),
        }
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return result.reindex(columns=RESULT_COLUMNS), audit
