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

    def output_path(self, city: str, res_exclude: Optional[int]) -> Path:
        return (
            Path(str(self.config["CURATE_FOLDER_EXPORT"]))
            / f"dinov3_city={city}_res_exclude={str(res_exclude)}.parquet"
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
        df_path = pd.read_csv(path_path, usecols=["panoid"]).drop_duplicates()
        df_pano = df_pano.merge(df_path, on="panoid", how="inner")
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
        }

    def write_sidecar(self, output_file: Path, stats: Dict[str, object]) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.with_suffix(".json").write_text(json.dumps(stats, indent=2))

    def process_city(
        self,
        city: str,
        res_exclude: Optional[int] = 11,
        allow_empty: bool = False,
    ) -> bool:
        output_file = self.output_path(city, res_exclude)

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

        if df_filtered.empty:
            stats = self.build_stats(city, df_embeddings, pd.DataFrame(), counts, "empty")
            self.write_sidecar(output_file, stats)
            if output_file.exists():
                output_file.unlink()
            return allow_empty

        df_aggregated = self.aggregate_to_hexagons(df_filtered)
        stats = self.build_stats(city, df_embeddings, df_aggregated, counts, "ok")
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
        type=int,
        default=11,
        choices=[11, 12, 13],
        help="H3 resolution used to exclude train/test leakage",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success when the join or post-exclusion set is empty",
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
        }
    )
    aggregator = DINOv3H3HexagonAggregator(config, log_level=args.log_level)
    ok = aggregator.process_city(
        args.city, res_exclude=args.res_exclude, allow_empty=args.allow_empty
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
