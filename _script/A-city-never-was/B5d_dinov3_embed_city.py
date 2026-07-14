#!/usr/bin/env python3
"""Per-city DINOv3 image embedding export."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from dinov3_utils import (
    atomic_write_parquet,
    build_embedding_columns,
    discover_embedding_columns,
    l2_normalize_rows,
    load_embedding_backend,
    resolve_city_file_stem,
)


DEFAULT_VALFOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir"
)
DEFAULT_OUTPUT_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed"
)
DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
DEFAULT_PANO_PATH_TEMPLATE = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
DEFAULT_MIN_YEAR = 2016
DEFAULT_MAX_YEAR = 2020


def load_city_image_index(valfolder: str | Path, city_file_stem: str) -> pd.DataFrame:
    """Load one city's image index parquet and ensure `name` exists."""
    path = Path(valfolder) / f"{city_file_stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"city image index not found: {path}")

    df = pd.read_parquet(path).copy()
    if "path" not in df.columns:
        raise ValueError(f"{path} must contain a path column")
    if "name" not in df.columns:
        df["name"] = df["path"].apply(lambda value: Path(str(value)).name)
    return df


def load_city_year_metadata(
    year_metadata_root: str | Path,
    city_file_stem: str,
    pano_path_template: str = DEFAULT_PANO_PATH_TEMPLATE,
) -> pd.DataFrame:
    """Load panoid/year metadata for one city from the GSV pano metadata file."""
    path = Path(
        pano_path_template.format(
            ROOTFOLDER=year_metadata_root,
            cityabbr=city_file_stem,
        )
    )
    if not path.exists():
        raise FileNotFoundError(f"city pano metadata not found: {path}")

    df = pd.read_csv(path)
    required = {"panoid", "year"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} must contain columns: {missing}")

    result = df.loc[:, ["panoid", "year"]].copy()
    result["panoid"] = result["panoid"].astype(str)
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result = result.dropna(subset=["panoid", "year"]).drop_duplicates(subset=["panoid"])
    result["year"] = result["year"].astype(int)
    return result


def filter_image_index_by_year(
    df: pd.DataFrame,
    year_metadata: pd.DataFrame,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """Keep only image rows whose panoid metadata year is in the inclusive range."""
    if min_year > max_year:
        raise ValueError("min_year must be less than or equal to max_year")

    working = df.copy()
    working["panoid"] = working["name"].astype(str).str[:22]
    merged = working.merge(year_metadata, on="panoid", how="inner")
    filtered = merged[
        (merged["year"] >= min_year) & (merged["year"] <= max_year)
    ].copy()
    return filtered.drop(columns=["year"]).reset_index(drop=True)


def collect_finished_names(
    curated_city_folder: str | Path,
    expected_model_name: str | None = None,
) -> set[str]:
    """Collect finished names after validating previous embedding chunks."""
    folder = Path(curated_city_folder)
    if not folder.exists():
        return set()

    names: set[str] = set()
    expected_embedding_dim: int | None = None
    expected_columns: list[str] | None = None
    for parquet_path in sorted(folder.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            raise RuntimeError(f"failed reading existing shard {parquet_path}: {exc}") from exc

        required = {"name", "embedding_dim"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"existing shard {parquet_path} is missing columns: {missing}")

        if df["name"].duplicated().any():
            duplicates = df.loc[df["name"].duplicated(), "name"].head().tolist()
            raise ValueError(f"duplicate names in existing shard {parquet_path}: {duplicates}")

        shard_names = df["name"].dropna().astype(str).tolist()
        duplicate_across_shards = sorted(set(names).intersection(shard_names))
        if duplicate_across_shards:
            raise ValueError(
                f"duplicate names across existing shards: {duplicate_across_shards[:5]}"
            )

        if df["embedding_dim"].nunique() != 1:
            raise ValueError(f"existing shard {parquet_path} contains mixed embedding_dim values")
        shard_dim = int(df["embedding_dim"].iloc[0])
        if expected_embedding_dim is None:
            expected_embedding_dim = shard_dim
        elif shard_dim != expected_embedding_dim:
            raise ValueError(
                f"existing shard {parquet_path} embedding_dim={shard_dim} does not "
                f"match previous embedding_dim={expected_embedding_dim}"
            )

        if expected_model_name is not None and "model_name" in df.columns:
            model_values = set(df["model_name"].dropna().astype(str).tolist())
            if model_values and model_values != {expected_model_name}:
                raise ValueError(
                    f"existing shard {parquet_path} model_name values {sorted(model_values)} "
                    f"do not match requested model_name={expected_model_name}"
                )

        embedding_cols = discover_embedding_columns(df)
        if len(embedding_cols) != shard_dim:
            raise ValueError(
                f"existing shard {parquet_path} embedding_dim={shard_dim} does not match "
                f"{len(embedding_cols)} embedding columns"
            )
        if expected_columns is None:
            expected_columns = embedding_cols
        elif embedding_cols != expected_columns:
            raise ValueError(f"existing shard {parquet_path} has incompatible embedding columns")

        values = df[embedding_cols].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"existing shard {parquet_path} contains non-finite values")
        norms = np.linalg.norm(values, axis=1)
        nonzero = norms > 1e-12
        if nonzero.any() and not np.allclose(norms[nonzero], 1.0, atol=1e-5):
            raise ValueError(f"existing shard {parquet_path} contains non-normalized vectors")

        names.update(shard_names)
    return names


def build_pending_batches(
    df: pd.DataFrame,
    finished_names: set[str],
    chunk_size: int,
) -> list[pd.DataFrame]:
    """Return pending rows split into chunk-size batches."""
    pending = df[~df["name"].isin(finished_names)].reset_index(drop=True)
    return [
        pending.iloc[start : start + chunk_size].reset_index(drop=True)
        for start in range(0, len(pending), chunk_size)
    ]


def validate_embedding_frame(df: pd.DataFrame) -> None:
    """Validate one output shard before writing."""
    if df["name"].duplicated().any():
        duplicates = df.loc[df["name"].duplicated(), "name"].head().tolist()
        raise ValueError(f"duplicate names in embedding shard: {duplicates}")

    if df["embedding_dim"].nunique() != 1:
        raise ValueError("embedding shard contains multiple embedding_dim values")

    embedding_cols = discover_embedding_columns(df)
    values = df[embedding_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("embedding shard contains non-finite values")

    norms = np.linalg.norm(values, axis=1)
    nonzero = norms > 1e-12
    if nonzero.any() and not np.allclose(norms[nonzero], 1.0, atol=1e-5):
        raise ValueError("nonzero embedding rows must be L2-normalized")


def _build_embedding_frame(
    batch: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    normalized = l2_normalize_rows(np.asarray(embeddings, dtype=float))
    if normalized.ndim != 2:
        raise ValueError("embedder must return a 2D array")
    if len(batch) != normalized.shape[0]:
        raise ValueError("embedder row count does not match input batch")

    embedding_cols = build_embedding_columns(normalized.shape[1])
    df = pd.DataFrame(normalized, columns=embedding_cols)
    df.insert(0, "embedding_dim", normalized.shape[1])
    df.insert(0, "model_name", model_name)
    df.insert(0, "panoid", batch["name"].astype(str).str[:22].to_numpy())
    df.insert(0, "name", batch["name"].astype(str).to_numpy())
    return df


def embed_city(
    city: str,
    city_file_stem: str | None,
    valfolder: str | Path,
    output_root: str | Path,
    model_name: str,
    backend_name: str,
    batch_size: int,
    device: str,
    local_files_only: bool,
    ignore_mismatched_sizes: bool = False,
    limit: int | None = None,
    year_metadata_root: str | Path = DEFAULT_ROOT,
    pano_path_template: str = DEFAULT_PANO_PATH_TEMPLATE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    year_filter_enabled: bool = True,
    backend_loader: Callable[..., tuple] = load_embedding_backend,
    embedder: Callable[..., np.ndarray] | None = None,
) -> list[Path]:
    """Embed one city and return written parquet shard paths."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    city_stem = resolve_city_file_stem(city, city_file_stem)
    output_dir = Path(output_root) / city_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_city_image_index(valfolder, city_stem)
    if year_filter_enabled:
        year_metadata = load_city_year_metadata(
            year_metadata_root,
            city_stem,
            pano_path_template=pano_path_template,
        )
        df = filter_image_index_by_year(df, year_metadata, min_year, max_year)

    finished_names = collect_finished_names(output_dir, expected_model_name=model_name)
    pending = df[~df["name"].isin(finished_names)].reset_index(drop=True)
    if limit is not None:
        pending = pending.head(limit).reset_index(drop=True)
    batches = build_pending_batches(pending, set(), batch_size)

    if not batches:
        return []

    model, processor = backend_loader(
        model_name=model_name,
        device=device,
        backend=backend_name,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    embedder = embedder or __import__("dinov3_utils").embed_image_batch

    written: list[Path] = []
    for batch_index, batch in enumerate(batches):
        embeddings = embedder(batch["path"].tolist(), model, processor, device)
        shard = _build_embedding_frame(batch, embeddings, model_name)
        validate_embedding_frame(shard)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_path = output_dir / f"{city_stem}_{timestamp}_{batch_index:04d}.parquet"
        atomic_write_parquet(shard, output_path)
        written.append(output_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed one city's images with DINOv3")
    parser.add_argument("--city", required=True, help="Display city name, e.g. Hong Kong")
    parser.add_argument("--city-file-stem", default=None, help="Override city parquet stem")
    parser.add_argument("--valfolder", default=DEFAULT_VALFOLDER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--year-metadata-root",
        default=DEFAULT_ROOT,
        help="Root containing GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv",
    )
    parser.add_argument(
        "--pano-path-template",
        default=DEFAULT_PANO_PATH_TEMPLATE,
        help="Format string used to locate pano metadata with ROOTFOLDER and cityabbr",
    )
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR)
    parser.add_argument(
        "--disable-year-filter",
        action="store_true",
        help="Embed all images instead of filtering to the configured metadata year range",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--backend", default="transformers", choices=["transformers", "timm"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local/cache model files; omit once on a connected node to download a verified model ID",
    )
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help=(
            "Pass ignore_mismatched_sizes=True to transformers for staged checkpoints "
            "with expected size mismatches; inspect the mismatch report before using"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = embed_city(
        city=args.city,
        city_file_stem=args.city_file_stem,
        valfolder=args.valfolder,
        output_root=args.output_root,
        model_name=args.model_name,
        backend_name=args.backend,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        limit=args.limit,
        year_metadata_root=args.year_metadata_root,
        pano_path_template=args.pano_path_template,
        min_year=args.min_year,
        max_year=args.max_year,
        year_filter_enabled=not args.disable_year_filter,
    )
    print(f"Wrote {len(written)} shard(s)")


if __name__ == "__main__":
    main()
