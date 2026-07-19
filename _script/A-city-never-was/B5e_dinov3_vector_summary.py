#!/usr/bin/env python3
"""
DINOv3 H3 vector aggregation.

Aggregates per-image DINOv3 embeddings to H3 cells while excluding validation
images that overlap train/test panoramas at high H3 resolutions.
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import h3
import numpy as np
import pandas as pd

from dinov3_utils import (
    atomic_write_parquet,
    discover_embedding_columns,
    l2_normalize_rows,
    resolve_city_file_stem,
)


DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
DEFAULT_INPUT_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_embed"
DEFAULT_OUTPUT_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_hex_summary"
DEFAULT_TRAIN_TEST_FOLDER = f"{DEFAULT_ROOT}/_transformed/t_classifier_img_yolo8"
DEFAULT_MIN_YEAR = 2016
DEFAULT_MAX_YEAR = 2020


def parse_optional_res_exclude(value: str | int | None) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "no", "false"}:
        return None
    return int(text)


def spatially_stratified_sample(
    df: pd.DataFrame, target_per_h3: int = 20, sampling_resolution: int = 8
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Select up to a fixed number of images per fine H3 cell deterministically."""
    if target_per_h3 < 1:
        raise ValueError("target_per_h3 must be at least 1")
    column = f"hex_{sampling_resolution}"
    if column not in df.columns:
        raise ValueError(f"Missing spatial sampling column: {column}")
    tie_breaker = "name" if "name" in df.columns else "panoid"
    if tie_breaker not in df.columns:
        raise ValueError("Spatial sampling requires a name or panoid column")
    ordered = df.sort_values([column, tie_breaker], kind="stable")
    counts = ordered.groupby(column, dropna=True).size()
    sampled = (
        ordered.groupby(column, dropna=True, group_keys=False)
        .head(target_per_h3)
        .reset_index(drop=True)
    )
    return sampled, {
        "sampling_h3_resolution": sampling_resolution,
        "equal_sampling_target_per_h3": target_per_h3,
        "equal_sampling_undersupplied_h3_count": int((counts < target_per_h3).sum()),
        "equal_sampling_selected_image_count": int(len(sampled)),
    }


def build_default_config() -> Dict[str, object]:
    return {
        "ROOTFOLDER": DEFAULT_ROOT,
        "CURATED_FOLDER": DEFAULT_INPUT_ROOT,
        "TRAIN_TEST_FOLDER": DEFAULT_TRAIN_TEST_FOLDER,
        "CURATE_FOLDER_EXPORT": DEFAULT_OUTPUT_ROOT,
        "PANO_PATH": "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv",
        "PATH_PATH": "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv",
        "summary_resolutions": [6, 7, 8],
        "exclude_resolutions": [11, 12, 13],
        "year_filter_enabled": True,
        "min_year": DEFAULT_MIN_YEAR,
        "max_year": DEFAULT_MAX_YEAR,
    }


class DINOv3H3HexagonAggregator:
    """Aggregate DINOv3 image embeddings to H3 cells."""

    def __init__(self, config: Dict[str, object], log_level: str = "INFO"):
        self.config = dict(config)
        self.summary_resolutions = list(self.config.get("summary_resolutions", [6, 7, 8]))
        self.exclude_resolutions = list(self.config.get("exclude_resolutions", [11, 12, 13]))
        self.vector_columns: List[str] = []
        self.setup_logging(log_level)
        self.setup_directories()
        self._validate_h3_version()

    def setup_logging(self, log_level: str) -> None:
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self) -> None:
        Path(str(self.config["CURATE_FOLDER_EXPORT"])).mkdir(parents=True, exist_ok=True)

    def _validate_h3_version(self) -> None:
        if hasattr(h3, "geo_to_h3"):
            self.h3_convert = lambda lat, lon, res: h3.geo_to_h3(lat, lon, res)
        elif hasattr(h3, "latlng_to_cell"):
            self.h3_convert = lambda lat, lon, res: h3.latlng_to_cell(lat, lon, res)
        else:
            raise ImportError("Incompatible H3 version: missing lat/lon to cell API")

    def output_path(
        self, city: str, res_exclude: Optional[int], equal_sampling: bool = False
    ) -> Path:
        sampling_suffix = "_sampling=equal" if equal_sampling else ""
        return (
            Path(str(self.config["CURATE_FOLDER_EXPORT"]))
            / f"dinov3_city={city}_res_exclude={str(res_exclude)}{sampling_suffix}.parquet"
        )

    def load_pano_metadata(self, city: str) -> pd.DataFrame:
        city_abbr = resolve_city_file_stem(city)
        pano_path = Path(
            str(self.config["PANO_PATH"]).format(
                ROOTFOLDER=self.config["ROOTFOLDER"], cityabbr=city_abbr
            )
        )
        path_path = Path(
            str(self.config["PATH_PATH"]).format(
                ROOTFOLDER=self.config["ROOTFOLDER"], cityabbr=city_abbr
            )
        )

        if not pano_path.exists() or not path_path.exists():
            self.logger.warning("Metadata files not found for %s", city)
            return pd.DataFrame()

        df_pano = pd.read_csv(pano_path)
        if "year" not in df_pano.columns:
            raise ValueError(f"{pano_path} must contain a year column")
        df_pano["year"] = pd.to_numeric(df_pano["year"], errors="coerce")

        df_path = pd.read_csv(path_path, usecols=["panoid"]).drop_duplicates()
        df_pano = df_pano.merge(df_path, on="panoid", how="inner")
        if df_pano.empty:
            return df_pano

        if bool(self.config.get("year_filter_enabled", True)):
            min_year = int(self.config.get("min_year", DEFAULT_MIN_YEAR))
            max_year = int(self.config.get("max_year", DEFAULT_MAX_YEAR))
            if min_year > max_year:
                raise ValueError("min_year must be less than or equal to max_year")
            df_pano = df_pano[
                (df_pano["year"] >= min_year) & (df_pano["year"] <= max_year)
            ].copy()
            if df_pano.empty:
                return df_pano

        return self._add_h3_indices(df_pano)

    def _add_h3_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        all_resolutions = sorted(set(self.summary_resolutions + self.exclude_resolutions))
        result = df.copy()
        for res in all_resolutions:
            result[f"hex_{res}"] = [
                self.h3_convert(float(row.lat), float(row.lon), res)
                for row in result.itertuples(index=False)
            ]
        return result

    def load_embedding_data(self, city: str) -> pd.DataFrame:
        city_abbr = resolve_city_file_stem(city)
        input_root = Path(str(self.config["CURATED_FOLDER"]))
        files = sorted((input_root / city_abbr).glob("*.parquet"))
        if not files:
            files = sorted(input_root.glob(f"*/{city_abbr}*.parquet"))
        if not files:
            self.logger.warning("No DINOv3 embedding files found for %s", city)
            return pd.DataFrame()

        frames = [pd.read_parquet(path) for path in files]
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return df

        if "panoid" not in df.columns:
            if "name" not in df.columns:
                raise ValueError("DINOv3 embedding rows must include 'panoid' or 'name'")
            df["panoid"] = df["name"].astype(str).str[:22]

        if "name" in df.columns:
            df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
        else:
            df = df.drop_duplicates(subset=["panoid"]).reset_index(drop=True)

        self.vector_columns = discover_embedding_columns(df)
        embedding_dim = df.get("embedding_dim")
        if embedding_dim is not None and int(embedding_dim.nunique()) != 1:
            raise ValueError("Expected exactly one embedding_dim per city input")
        if embedding_dim is not None and int(embedding_dim.iloc[0]) != len(self.vector_columns):
            raise ValueError(
                f"embedding_dim={int(embedding_dim.iloc[0])} does not match "
                f"{len(self.vector_columns)} discovered columns"
            )

        values = df[self.vector_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("DINOv3 embedding rows contain non-finite vector values")

        return df

    def load_train_test_panoids(self, city: str) -> Set[str]:
        pattern = f"*/{city}/*.jpg"
        return {
            path.stem[:22]
            for path in Path(str(self.config["TRAIN_TEST_FOLDER"])).glob(pattern)
        }

    def apply_exclusion(
        self,
        df_embeddings: pd.DataFrame,
        df_pano: pd.DataFrame,
        train_panoids: Set[str],
        res_exclude: Optional[int],
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        df_merged = df_embeddings.merge(
            df_pano.drop(columns=["id"], errors="ignore"), on="panoid", how="inner"
        )
        before = len(df_merged)

        if res_exclude is not None:
            if res_exclude not in self.exclude_resolutions:
                raise ValueError(
                    f"res_exclude={res_exclude} is not in {self.exclude_resolutions}"
                )
            train_df = df_pano[df_pano["panoid"].isin(train_panoids)]
            train_hexagons = set(train_df[f"hex_{res_exclude}"].dropna().unique())
            df_merged = df_merged[
                ~df_merged[f"hex_{res_exclude}"].isin(train_hexagons)
            ].copy()

        after = len(df_merged)
        return df_merged, {
            "image_count_before_exclusion": before,
            "image_count_after_exclusion": after,
            "excluded_image_count": before - after,
        }

    def aggregate_to_hexagons(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        results = []
        for res in self.summary_resolutions:
            grouped = df.groupby(f"hex_{res}", sort=True, dropna=True)
            vectors = grouped[self.vector_columns].mean()
            vectors.loc[:, self.vector_columns] = l2_normalize_rows(
                vectors[self.vector_columns].to_numpy(dtype=float)
            )

            out = vectors.reset_index().rename(columns={f"hex_{res}": "hex_id"})
            out.insert(1, "res", res)
            out.insert(2, "img_count", grouped.size().to_numpy())
            out.insert(
                3,
                "model_name",
                grouped["model_name"].first().to_numpy()
                if "model_name" in df.columns
                else "",
            )
            out.insert(
                4,
                "embedding_dim",
                grouped["embedding_dim"].first().to_numpy()
                if "embedding_dim" in df.columns
                else len(self.vector_columns),
            )
            values = out[self.vector_columns].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Aggregated H3 vectors contain non-finite values")
            results.append(out)

        return pd.concat(results, ignore_index=True)


    def build_stats(
        self,
        city: str,
        df_embeddings: pd.DataFrame,
        df_aggregated: pd.DataFrame,
        counts: Dict[str, int],
        status: str,
    ) -> Dict[str, object]:
        model_name = ""
        if "model_name" in df_embeddings.columns and not df_embeddings.empty:
            model_values = df_embeddings["model_name"].dropna().unique()
            if len(model_values) == 1:
                model_name = str(model_values[0])
            elif len(model_values) > 1:
                model_name = "mixed"

        embedding_dim = len(self.vector_columns)
        if "embedding_dim" in df_embeddings.columns and not df_embeddings.empty:
            dim_values = df_embeddings["embedding_dim"].dropna().unique()
            if len(dim_values) == 1:
                embedding_dim = int(dim_values[0])

        res8 = (
            df_aggregated[df_aggregated["res"] == 8]
            if not df_aggregated.empty and "res" in df_aggregated.columns
            else pd.DataFrame()
        )
        return {
            "status": status,
            "city": city,
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "image_count_before_exclusion": int(
                counts.get("image_count_before_exclusion", 0)
            ),
            "image_count_after_exclusion": int(
                counts.get("image_count_after_exclusion", 0)
            ),
            "excluded_image_count": int(counts.get("excluded_image_count", 0)),
            "res8_h3_count": int(len(res8)),
            "res8_mean_image_count": float(res8["img_count"].mean())
            if not res8.empty
            else 0.0,
            "included_years": sorted(
                int(year)
                for year in pd.to_numeric(df_embeddings["year"], errors="coerce")
                .dropna()
                .unique()
            )
            if "year" in df_embeddings.columns
            else [],
        }

    def write_sidecar(self, output_file: Path, stats: Dict[str, object]) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.with_suffix(".json").write_text(json.dumps(stats, indent=2))

    def process_city(
        self,
        city: str,
        res_exclude: Optional[int] = None,
        allow_empty: bool = False,
        equal_sampling: bool = False,
        equal_sampling_target_per_h3: int = 20,
        equal_sampling_min_images: int = 20,
    ) -> bool:
        output_file = self.output_path(city, res_exclude, equal_sampling)

        df_pano = self.load_pano_metadata(city)
        df_embeddings = self.load_embedding_data(city)
        if df_pano.empty or df_embeddings.empty:
            counts = {
                "image_count_before_exclusion": 0,
                "image_count_after_exclusion": 0,
                "excluded_image_count": 0,
            }
            stats = self.build_stats(city, df_embeddings, pd.DataFrame(), counts, "empty")
            self.write_sidecar(output_file, stats)
            return allow_empty

        train_panoids = self.load_train_test_panoids(city)
        df_filtered, counts = self.apply_exclusion(
            df_embeddings, df_pano, train_panoids, res_exclude
        )

        if equal_sampling and not df_filtered.empty:
            df_filtered, sampling_audit = spatially_stratified_sample(
                df_filtered, target_per_h3=equal_sampling_target_per_h3
            )
            counts.update(sampling_audit)
            if len(df_filtered) < equal_sampling_min_images:
                raise ValueError(
                    f"Equal sampling selected {len(df_filtered)} images for {city}; "
                    f"at least {equal_sampling_min_images} are required"
                )

        if df_filtered.empty:
            stats = self.build_stats(city, df_embeddings, pd.DataFrame(), counts, "empty")
            self.write_sidecar(output_file, stats)
            if output_file.exists():
                output_file.unlink()
            return allow_empty

        df_aggregated = self.aggregate_to_hexagons(df_filtered)
        stats = self.build_stats(city, df_filtered, df_aggregated, counts, "ok")
        stats["sampling_mode"] = "equal" if equal_sampling else "all"
        atomic_write_parquet(df_aggregated, output_file)
        self.write_sidecar(output_file, stats)
        gc.collect()
        return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DINOv3 vectors to H3 cells")
    parser.add_argument("--city", default="Hong Kong", help="City name")
    parser.add_argument(
        "--rootfolder",
        default=DEFAULT_ROOT,
        help="Root folder containing GSV metadata",
    )
    parser.add_argument(
        "--equal-sampling",
        action="store_true",
        help="Cap each occupied H3 resolution-8 cell to a spatially stratified sample",
    )
    parser.add_argument("--equal-sampling-target-per-h3", type=int, default=20)
    parser.add_argument("--equal-sampling-min-images", type=int, default=20)
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help="DINOv3 per-image embedding root",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="DINOv3 H3 summary output root",
    )
    parser.add_argument(
        "--train-test-folder",
        default=DEFAULT_TRAIN_TEST_FOLDER,
        help="Train/test image folder used for H3 exclusion",
    )
    parser.add_argument(
        "--res-exclude",
        type=parse_optional_res_exclude,
        default=None,
        help="Optional H3 resolution used to exclude train/test leakage; default is no exclusion",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success when the join or post-exclusion set is empty",
    )
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR)
    parser.add_argument(
        "--disable-year-filter",
        action="store_true",
        help="Aggregate all embedding years instead of filtering pano metadata to the configured year range",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_default_config()
    config.update(
        {
            "ROOTFOLDER": args.rootfolder,
            "CURATED_FOLDER": args.input_root,
            "CURATE_FOLDER_EXPORT": args.output_root,
            "TRAIN_TEST_FOLDER": args.train_test_folder,
            "year_filter_enabled": not args.disable_year_filter,
            "min_year": args.min_year,
            "max_year": args.max_year,
        }
    )
    aggregator = DINOv3H3HexagonAggregator(config, log_level=args.log_level)
    ok = aggregator.process_city(
        args.city,
        res_exclude=args.res_exclude,
        allow_empty=args.allow_empty,
        equal_sampling=args.equal_sampling,
        equal_sampling_target_per_h3=args.equal_sampling_target_per_h3,
        equal_sampling_min_images=args.equal_sampling_min_images,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
