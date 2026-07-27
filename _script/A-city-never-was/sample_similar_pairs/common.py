"""Shared loading, eligibility, sampling, and output helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h3
import numpy as np
import pandas as pd

from dinov3_utils import discover_embedding_columns, resolve_city_file_stem


@dataclass(frozen=True)
class CityVectors:
    city: str
    metadata: pd.DataFrame
    vector_columns: list[str]
    vectors: np.ndarray

    def take(self, indices: np.ndarray | list[int]) -> "CityVectors":
        positions = np.asarray(indices, dtype=int)
        return CityVectors(
            self.city,
            self.metadata.iloc[positions].reset_index(drop=True),
            self.vector_columns,
            np.ascontiguousarray(self.vectors[positions], dtype=np.float32),
        )


def parse_city_pairs(values: Iterable[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_value in values:
        parts = [part.strip() for part in raw_value.split("|")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"city pair must be CITY1|CITY2: {raw_value!r}")
        pair = (parts[0], parts[1])
        if pair[0] == pair[1]:
            raise ValueError(f"city pair cannot use the same city twice: {raw_value!r}")
        if pair in seen:
            raise ValueError(f"duplicate directed city pair: {raw_value!r}")
        seen.add(pair)
        pairs.append(pair)
    if not pairs:
        raise ValueError("at least one city pair is required")
    return pairs


def _embedding_files(embedding_root: Path, city: str) -> list[Path]:
    stem = resolve_city_file_stem(city)
    files = sorted((embedding_root / stem).glob("*.parquet"))
    if not files:
        files = sorted(embedding_root.glob(f"*/{stem}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no embedding Parquet shards found for {city!r} under {embedding_root}")
    return files


def load_city_embeddings(embedding_root: Path | str, city: str) -> CityVectors:
    """Load and validate normalized DINOv3 embeddings for one city."""
    files = _embedding_files(Path(embedding_root), city)
    frames = [pd.read_parquet(file_path) for file_path in files]
    frame = pd.concat(frames, ignore_index=True)
    required = {"name", "panoid", "embedding_dim"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"embedding data for {city!r} is missing columns: {sorted(missing)}")
    if frame["name"].duplicated().any():
        raise ValueError(f"embedding data for {city!r} has duplicate names")
    if frame["embedding_dim"].nunique() != 1:
        raise ValueError(f"embedding data for {city!r} has mixed embedding_dim values")
    columns = discover_embedding_columns(frame)
    embedding_dim = int(frame["embedding_dim"].iloc[0])
    if len(columns) != embedding_dim:
        raise ValueError(
            f"embedding data for {city!r} has embedding_dim={embedding_dim} but {len(columns)} vector columns"
        )
    vectors = np.ascontiguousarray(frame[columns].to_numpy(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError(f"embedding data for {city!r} contains non-finite vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12) or not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"embedding data for {city!r} must contain L2-normalized nonzero vectors")
    return CityVectors(city, frame.drop(columns=columns).reset_index(drop=True), columns, vectors)


def _latlng_to_cell(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, resolution)
    return h3.latlng_to_cell(lat, lon, resolution)


def attach_image_geography(
    city_vectors: CityVectors,
    rootfolder: Path | str,
    train_test_folder: Path | str | None,
    min_year: int = 2012,
    max_year: int = 2022,
    h3_resolution: int = 8,
    res_exclude: int | None = None,
) -> CityVectors:
    """Join embeddings to eligible panorama metadata and calculate H3 membership."""
    if min_year > max_year:
        raise ValueError("min_year must be less than or equal to max_year")
    stem = resolve_city_file_stem(city_vectors.city)
    metadata_dir = Path(rootfolder) / "GSV" / "gsv_rgb" / stem / "gsvmeta"
    pano_file = metadata_dir / "gsv_pano.csv"
    path_file = metadata_dir / "gsv_path.csv"
    if not pano_file.exists() or not path_file.exists():
        raise FileNotFoundError(f"missing panorama metadata for {city_vectors.city!r}: {metadata_dir}")
    pano = pd.read_csv(pano_file)
    required = {"panoid", "year", "lat", "lon"}
    missing = required - set(pano.columns)
    if missing:
        raise ValueError(f"{pano_file} is missing columns: {sorted(missing)}")
    valid_paths = pd.read_csv(path_file, usecols=["panoid"]).drop_duplicates()
    pano["year"] = pd.to_numeric(pano["year"], errors="coerce")
    pano = pano.merge(valid_paths, on="panoid", how="inner")
    pano = pano[pano["year"].between(min_year, max_year)].copy()
    if pano.empty:
        raise ValueError(f"no panorama metadata remains for {city_vectors.city!r} in {min_year}-{max_year}")
    pano["hex_id"] = [
        _latlng_to_cell(float(row.lat), float(row.lon), h3_resolution)
        for row in pano.itertuples(index=False)
    ]
    if res_exclude is not None:
        if train_test_folder is None:
            raise ValueError("train_test_folder is required when res_exclude is set")
        pano["exclude_hex_id"] = [
            _latlng_to_cell(float(row.lat), float(row.lon), res_exclude)
            for row in pano.itertuples(index=False)
        ]
        train_panoids = {
            path.stem[:22]
            for path in Path(train_test_folder).glob(f"*/{city_vectors.city}/*.jpg")
        }
        excluded = set(pano.loc[pano["panoid"].isin(train_panoids), "exclude_hex_id"])
        pano = pano[~pano["exclude_hex_id"].isin(excluded)].copy()
    embedding_metadata = city_vectors.metadata.assign(_embedding_row=np.arange(len(city_vectors.metadata)))
    joined = embedding_metadata.merge(
        pano[["panoid", "year", "lat", "lon", "hex_id"]], on="panoid", how="inner"
    )
    if joined.empty:
        raise ValueError(f"no eligible embedding rows remain for {city_vectors.city!r}")
    indices = joined.pop("_embedding_row").to_numpy(dtype=int)
    return CityVectors(city_vectors.city, joined.reset_index(drop=True), city_vectors.vector_columns, city_vectors.vectors[indices])


def spatially_sample_images(
    frame: pd.DataFrame, max_images_per_h3: int, max_images_per_city: int
) -> pd.DataFrame:
    if max_images_per_h3 < 1 or max_images_per_city < 0:
        raise ValueError("max_images_per_h3 must be positive and max_images_per_city cannot be negative")
    required = {"hex_id", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sampling data is missing columns: {sorted(missing)}")
    sampled = (
        frame.sort_values(["hex_id", "name"], kind="stable")
        .groupby("hex_id", dropna=False, group_keys=False)
        .head(max_images_per_h3)
        .reset_index(drop=True)
    )
    return sampled if max_images_per_city == 0 else sampled.head(max_images_per_city).reset_index(drop=True)


def spatially_sample_city(
    city_vectors: CityVectors, max_images_per_h3: int, max_images_per_city: int
) -> CityVectors:
    sampled = spatially_sample_images(city_vectors.metadata.assign(_row=np.arange(len(city_vectors.metadata))), max_images_per_h3, max_images_per_city)
    return city_vectors.take(sampled["_row"].to_numpy(dtype=int))


def write_parquet_with_json_audit(
    frame: pd.DataFrame, output: Path | str, audit: dict
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    output_path.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True))
