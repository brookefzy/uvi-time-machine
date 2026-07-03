#!/usr/bin/env python3
"""Verify one-city DINOv3 smoke-test embeddings before full batch jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from dinov3_utils import discover_embedding_columns, resolve_city_file_stem


DEFAULT_SMOKE_ROOT = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_embed_smoke"


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def find_smoke_files(smoke_root: str | Path, city: str, city_file_stem: str | None = None) -> list[Path]:
    root = Path(smoke_root)
    city_stem = resolve_city_file_stem(city, city_file_stem)
    files = sorted((root / city_stem).glob("*.parquet"))
    if not files:
        files = sorted(root.glob(f"*/{city_stem}*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No smoke parquet files found for city={city!r} under {root}"
        )
    return files


def load_smoke_frame(files: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def verify_smoke_output(
    city: str,
    smoke_root: str | Path = DEFAULT_SMOKE_ROOT,
    expected_model_name: str | None = None,
    city_file_stem: str | None = None,
    min_rows: int = 1,
    norm_atol: float = 1e-5,
) -> dict[str, object]:
    files = find_smoke_files(smoke_root, city, city_file_stem=city_file_stem)
    rows_by_file = {str(path): int(len(pd.read_parquet(path))) for path in files}
    df = load_smoke_frame(files)

    required = {"name", "panoid", "model_name", "embedding_dim"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Smoke output is missing required columns: {missing}")

    if len(df) < min_rows:
        raise ValueError(f"Smoke output has {len(df)} rows, expected at least {min_rows}")

    duplicate_name_count = int(df["name"].duplicated().sum())
    if duplicate_name_count:
        examples = df.loc[df["name"].duplicated(), "name"].head(5).tolist()
        raise ValueError(f"Smoke output contains duplicate names: {examples}")

    if df["embedding_dim"].nunique() != 1:
        raise ValueError("Smoke output contains mixed embedding_dim values")

    embedding_cols = discover_embedding_columns(df)
    embedding_dim = int(df["embedding_dim"].iloc[0])
    if embedding_dim != len(embedding_cols):
        raise ValueError(
            f"embedding_dim={embedding_dim} does not match {len(embedding_cols)} e_* columns"
        )

    if expected_model_name is not None:
        observed_models = set(df["model_name"].dropna().astype(str).tolist())
        if observed_models != {expected_model_name}:
            raise ValueError(
                f"Smoke model_name values {sorted(observed_models)} do not match "
                f"expected {expected_model_name!r}"
            )

    values = df[embedding_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Smoke output contains non-finite embedding values")

    norms = np.linalg.norm(values, axis=1)
    nonzero = norms > 1e-12
    if not nonzero.any():
        raise ValueError("Smoke output contains only zero vectors")
    if not np.allclose(norms[nonzero], 1.0, atol=norm_atol):
        raise ValueError("Smoke output vectors are not unit-normalized")

    norm_distribution = {
        "min": float(np.min(norms)),
        "p50": float(np.quantile(norms, 0.50)),
        "p95": float(np.quantile(norms, 0.95)),
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
    }
    model_counts = {
        str(key): int(value)
        for key, value in df["model_name"].value_counts(dropna=False).to_dict().items()
    }

    print("SMOKE CHECK PASSED")
    print(f"city: {city}")
    print(f"smoke_root: {Path(smoke_root)}")
    print(f"file_count: {len(files)}")
    print(f"row_count: {len(df)}")
    print(f"embedding_dim: {embedding_dim}")
    print(f"duplicate_name_count: {duplicate_name_count}")
    print("rows_by_file:")
    for path, count in rows_by_file.items():
        print(f"  {path}: {count}")
    print("model_name_counts:")
    for model_name, count in model_counts.items():
        print(f"  {model_name}: {count}")
    print("norm_distribution:")
    for key in ["min", "p50", "p95", "max", "mean"]:
        print(f"  {key}: {_format_float(norm_distribution[key])}")

    return {
        "city": city,
        "file_count": len(files),
        "row_count": int(len(df)),
        "embedding_dim": embedding_dim,
        "duplicate_name_count": duplicate_name_count,
        "rows_by_file": rows_by_file,
        "model_name_counts": model_counts,
        "norm_distribution": norm_distribution,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="Hong Kong")
    parser.add_argument("--city-file-stem")
    parser.add_argument("--smoke-root", default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--expected-model-name")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--norm-atol", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verify_smoke_output(
        city=args.city,
        smoke_root=args.smoke_root,
        expected_model_name=args.expected_model_name,
        city_file_stem=args.city_file_stem,
        min_rows=args.min_rows,
        norm_atol=args.norm_atol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
