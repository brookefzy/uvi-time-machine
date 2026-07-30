"""Shared contracts for globally comparable DINOv3 visual modes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sample_similar_pairs.common import (
    CityVectors,
    attach_image_geography,
    load_city_embeddings,
    spatially_sample_city,
)


def sample_per_hex(frame: pd.DataFrame, max_images_per_hex: int = 50) -> pd.DataFrame:
    """Select a deterministic lexical image sample from every H3 cell."""
    if max_images_per_hex < 1:
        raise ValueError("max_images_per_hex must be positive")
    missing = {"hex_id", "name"} - set(frame.columns)
    if missing:
        raise ValueError(f"sampling data is missing columns: {sorted(missing)}")
    return (
        frame.sort_values(["hex_id", "name"], kind="stable")
        .groupby("hex_id", dropna=False, group_keys=False)
        .head(max_images_per_hex)
        .reset_index(drop=True)
    )


def validate_vectors(values: np.ndarray, embedding_dim: int | None = None) -> np.ndarray:
    """Return a finite, C-contiguous float32 two-dimensional vector matrix."""
    vectors = np.ascontiguousarray(values, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if embedding_dim is not None and vectors.shape[1] != embedding_dim:
        raise ValueError(f"embedding dimension is {vectors.shape[1]}, expected {embedding_dim}")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain non-finite values")
    return vectors


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize vectors, rejecting zero rows instead of silently retaining them."""
    vectors = validate_vectors(values)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("vectors contain a zero-norm row")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def centroid_checksum(centroids: np.ndarray, immutable_config: dict) -> str:
    """Hash canonical centroids/config before a derived model ID is embedded."""
    matrix = validate_vectors(centroids)
    little_endian = np.ascontiguousarray(matrix.astype("<f4", copy=False))
    digest = hashlib.sha256()
    digest.update(little_endian.tobytes(order="C"))
    digest.update(_canonical_json(immutable_config))
    return digest.hexdigest()


def build_model_id(checksum: str, immutable_config: dict) -> str:
    """Construct a readable immutable model version from its pre-ID checksum."""
    k = immutable_config.get("k")
    if not isinstance(k, int) or k < 1:
        raise ValueError("immutable_config must contain a positive integer k")
    identity = hashlib.sha256(checksum.encode("ascii") + _canonical_json(immutable_config)).hexdigest()
    return f"k={k}-{identity[:16]}"


def write_parquet_with_json_audit(frame: pd.DataFrame, output: Path | str, audit: dict) -> None:
    """Atomically publish a Parquet artifact and deterministic sidecar audit."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = output_path.with_name(f".{output_path.name}.tmp")
    tmp_json = output_path.with_suffix(".json.tmp")
    try:
        frame.to_parquet(tmp_parquet, index=False)
        tmp_json.write_bytes(_canonical_json(audit))
        tmp_parquet.replace(output_path)
        tmp_json.replace(output_path.with_suffix(".json"))
    finally:
        tmp_parquet.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)


def require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("FAISS is required; install faiss-cpu in the job environment") from exc
    return faiss


def js_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jensen--Shannon distance using base-2 logarithms."""
    p = validate_vectors(np.asarray(left, dtype=np.float32).reshape(1, -1))[0]
    q = validate_vectors(np.asarray(right, dtype=np.float32).reshape(1, -1), embedding_dim=len(p))[0]
    if np.any(p < 0) or np.any(q < 0) or not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("histograms must be non-negative and sum to one")
    midpoint = (p + q) / 2
    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))
    return float(np.sqrt((divergence(p) + divergence(q)) / 2))


def js_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return 1.0 - js_distance(left, right)


def validate_sparse_histogram(frame: pd.DataFrame) -> None:
    required = {"city", "hex_id", "res", "mode_id", "mode_image_count", "sampled_image_count", "mode_fraction", "model_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sparse histogram is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("sparse histogram must not be empty")
    if (frame["mode_image_count"] <= 0).any() or (frame["sampled_image_count"] <= 0).any():
        raise ValueError("sparse histogram counts must be positive")
    grouped = frame.groupby(["city", "hex_id", "res", "model_id"], dropna=False)
    if not np.allclose(grouped["mode_fraction"].sum().to_numpy(), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("sparse histogram fractions must sum to one per H3 cell")
    if not (grouped["mode_image_count"].sum() == grouped["sampled_image_count"].first()).all():
        raise ValueError("sparse histogram counts do not match sampled image counts")
