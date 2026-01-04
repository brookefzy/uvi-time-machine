#!/usr/bin/env python
"""
H3 Hexagon Aggregation Processor
Aggregates urban visual features to H3 hexagons with robust data exclusion
to prevent train-test data leakage in machine learning experiments.
"""

import os
import logging
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import warnings

import duckdb
import pandas as pd
import numpy as np
import h3
from tqdm import tqdm

from urban_utils import (
    UrbanDataConfig,
    DuckDBManager,
    DataValidator,
    CheckpointManager,
    ProgressTracker,
)


class H3HexagonAggregator:
    """Aggregate urban visual features to H3 hexagons with data leakage prevention."""

    def __init__(self, config: Dict[str, any], log_level: str = "INFO"):
        """
        Initialize the aggregator with configuration and logging.

        Args:
            config: Configuration dictionary with paths and parameters
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config = config
        self.setup_logging(log_level)
        self.db_manager = DuckDBManager(logger=self.logger)
        self.checkpoint_manager = CheckpointManager("checkpoints/hexagon_agg")
        self.setup_directories()

        # Feature columns (0-126 for visual features)
        self.vector_columns = [str(x) for x in range(127)]

        # H3 resolutions
        self.summary_resolutions = config.get("summary_resolutions", [6, 7, 8])
        self.exclude_resolutions = config.get("exclude_resolutions", [11, 12, 13])

        # Cities to overwrite even if processed
        self.cities_to_overwrite = set(
            config.get("cities_to_overwrite", ["Santiago", "Minneapolis"])
        )

        # Validate H3 version
        self._validate_h3_version()

    def setup_logging(self, log_level: str) -> None:
        """Configure logging with both file and console handlers."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"h3_aggregation_{timestamp}.log"

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [self.config["CURATE_FOLDER_EXPORT"], "checkpoints/hexagon_agg"]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def _validate_h3_version(self) -> None:
        """Validate and log H3 library version."""
        h3_version = h3.__version__
        self.logger.info(f"Using H3 version: {h3_version}")

        # Check for API differences between versions
        if hasattr(h3, "geo_to_h3"):
            self.h3_convert = lambda lat, lon, res: h3.geo_to_h3(lat, lon, res)
            self.logger.debug("Using h3.geo_to_h3 API")
        elif hasattr(h3, "latlng_to_cell"):
            self.h3_convert = lambda lat, lon, res: h3.latlng_to_cell(lat, lon, res)
            self.logger.debug("Using h3.latlng_to_cell API")
        else:
            raise ImportError(
                "Incompatible H3 version - cannot find conversion function"
            )

    def load_pano_metadata(self, city: str) -> pd.DataFrame:
        """
        Load panorama metadata using DuckDB for efficiency.

        Args:
            city: City name

        Returns:
            DataFrame with panorama metadata and H3 indices
        """
        city_abbr = city.lower().replace(" ", "")
        pano_path = self.config["PANO_PATH"].format(
            ROOTFOLDER=self.config["ROOTFOLDER"], cityabbr=city_abbr
        )
        path_path = self.config["PATH_PATH"].format(
            ROOTFOLDER=self.config["ROOTFOLDER"], cityabbr=city_abbr
        )

        if not Path(pano_path).exists() or not Path(path_path).exists():
            self.logger.warning(f"Metadata files not found for {city}")
            return pd.DataFrame()

        # Use DuckDB for efficient loading and filtering
        query = f"""
            WITH pano AS (
                SELECT * FROM read_csv_auto('{pano_path}')
            ),
            path AS (
                SELECT DISTINCT panoid FROM read_csv_auto('{path_path}')
            )
            SELECT pano.*
            FROM pano
            INNER JOIN path ON pano.panoid = path.panoid
        """

        try:
            df_pano = self.db_manager.execute_query(query)
            self.logger.info(f"Loaded {len(df_pano)} panoramas with paths for {city}")

            # Add H3 indices for all resolutions
            df_pano = self._add_h3_indices(df_pano)

            return df_pano.drop(columns=["lat", "lon"])

        except Exception as e:
            self.logger.error(f"Error loading metadata for {city}: {e}")
            return pd.DataFrame()

    def _add_h3_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add H3 indices for multiple resolutions using vectorized operations.

        Args:
            df: DataFrame with lat/lon columns

        Returns:
            DataFrame with added H3 index columns
        """
        all_resolutions = self.summary_resolutions + self.exclude_resolutions

        for res in tqdm(all_resolutions, desc="Adding H3 indices"):
            # Vectorized H3 conversion for better performance
            h3_indices = []
            for _, row in df.iterrows():
                h3_idx = self.h3_convert(row["lat"], row["lon"], res)
                h3_indices.append(h3_idx)

            df[f"hex_{res}"] = h3_indices
            self.logger.debug(f"Added H3 indices for resolution {res}")

        return df

    def load_prediction_data(self, city: str) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Load prediction data and training/test image IDs using DuckDB.

        Args:
            city: City name

        Returns:
            Tuple of (predictions DataFrame, training panoid set)
        """
        city_abbr = city.lower().replace(" ", "")

        # Find prediction files
        pattern = f"*/{city_abbr}*.parquet"
        pred_files = list(Path(self.config["CURATED_FOLDER"]).glob(pattern))

        if not pred_files:
            self.logger.warning(f"No prediction files found for {city}")
            return pd.DataFrame(), set()

        self.logger.info(f"Found {len(pred_files)} prediction files for {city}")

        # Load predictions using DuckDB
        all_predictions = []
        for file_path in pred_files:
            query = f"""
                SELECT DISTINCT *
                FROM read_parquet('{file_path}')
            """
            df_pred = self.db_manager.execute_query(query)
            all_predictions.append(df_pred)

        # Combine all predictions
        df_combined = pd.concat(all_predictions, ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["name"]).reset_index(
            drop=True
        )
        df_combined["panoid"] = df_combined["name"].apply(lambda x: x[:22])

        # Get training/test image IDs
        train_test_pattern = f"*/{city}/*.jpg"
        train_test_files = list(
            Path(self.config["TRAIN_TEST_FOLDER"]).glob(train_test_pattern)
        )
        train_test_panoids = {Path(f).stem[:22] for f in train_test_files}

        self.logger.info(
            f"Loaded {len(df_combined)} predictions and {len(train_test_panoids)} train/test IDs"
        )

        return df_combined, train_test_panoids

    def apply_robust_exclusion(
        self,
        df_predictions: pd.DataFrame,
        df_pano: pd.DataFrame,
        train_panoids: Set[str],
        res_exclude: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Apply robust data exclusion to prevent train-test leakage.

        Args:
            df_predictions: Prediction DataFrame
            df_pano: Panorama metadata DataFrame
            train_panoids: Set of training panorama IDs
            res_exclude: H3 resolution for exclusion (None = no exclusion)

        Returns:
            Aggregated features by hexagon
        """
        # Merge predictions with panorama metadata
        df_merged = df_predictions.merge(
            df_pano.drop("id", axis=1, errors="ignore"), on="panoid", how="inner"
        )

        if res_exclude is not None:
            self.logger.info(f"Applying exclusion at resolution {res_exclude}")

            # Get training hexagons at exclusion resolution
            train_df = df_pano[df_pano["panoid"].isin(train_panoids)]
            train_hexagons = set(train_df[f"hex_{res_exclude}"].unique())

            # Exclude validation data in same hexagons as training
            initial_count = len(df_merged)
            df_merged = df_merged[~df_merged[f"hex_{res_exclude}"].isin(train_hexagons)]
            excluded_count = initial_count - len(df_merged)

            self.logger.info(
                f"Excluded {excluded_count} samples ({excluded_count/initial_count*100:.1f}%)"
            )
        else:
            self.logger.info("No exclusion applied")

        # Aggregate to hexagons using DuckDB for efficiency
        return self._aggregate_to_hexagons(df_merged)

    def _aggregate_to_hexagons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features to H3 hexagons using DuckDB.

        Args:
            df: DataFrame with features and hexagon indices

        Returns:
            Aggregated features by hexagon
        """
        # Register DataFrame with DuckDB
        self.db_manager.register_dataframe(df, "features")

        results = []
        vector_cols = ", ".join(
            [f'AVG("{col}") as "{col}"' for col in self.vector_columns]
        )

        for res in self.summary_resolutions:
            query = f"""
                SELECT 
                    hex_{res} as hex_id,
                    {res} as res,
                    {vector_cols}
                FROM features
                GROUP BY hex_{res}
            """

            df_agg = self.db_manager.execute_query(query)
            results.append(df_agg)

            self.logger.debug(f"Aggregated {len(df_agg)} hexagons at resolution {res}")

        # Combine all resolutions
        df_combined = pd.concat(results, ignore_index=True)

        # Clean up
        self.db_manager.conn.execute("DROP VIEW IF EXISTS features")

        return df_combined

    def compute_statistics(self, df: pd.DataFrame, resolution: int = 8) -> Dict:
        """
        Compute statistics for visualization and validation.

        Args:
            df: Aggregated hexagon DataFrame
            resolution: H3 resolution for analysis

        Returns:
            Dictionary of statistics
        """
        df_res = df[df["res"] == resolution].copy()

        if df_res.empty:
            return {}

        # Find dominant classes
        df_res["max_class"] = df_res[self.vector_columns].idxmax(axis=1)
        df_res["max_prob"] = df_res[self.vector_columns].max(axis=1)

        # Second highest class
        def get_second_max(row):
            values = row[self.vector_columns]
            max_col = row["max_class"]
            return values.drop(max_col).idxmax()

        df_res["second_class"] = df_res.apply(get_second_max, axis=1)

        stats = {
            "total_hexagons": len(df_res),
            "avg_max_prob": df_res["max_prob"].mean(),
            "top_classes": df_res["max_class"].value_counts().head(10).to_dict(),
            "second_classes": df_res["second_class"].value_counts().head(10).to_dict(),
        }

        self.logger.info(
            f"Resolution {resolution} stats: {stats['total_hexagons']} hexagons, "
            f"avg max prob: {stats['avg_max_prob']:.3f}"
        )

        return stats

    def process_city(self, city: str, force_overwrite: bool = False) -> bool:
        """
        Process a single city with all exclusion levels.

        Args:
            city: City name
            force_overwrite: Whether to overwrite existing results

        Returns:
            Success status
        """
        self.logger.info(f"Processing city: {city}")

        # Check if already processed
        if not force_overwrite and not (city in self.cities_to_overwrite):
            existing_files = list(
                Path(self.config["CURATE_FOLDER_EXPORT"]).glob(
                    f"*city={city}_*.parquet"
                )
            )
            if existing_files:
                self.logger.info(f"City {city} already processed, skipping")
                return True

        try:
            # Load panorama metadata
            df_pano = self.load_pano_metadata(city)
            if df_pano.empty:
                self.logger.warning(f"No panorama data for {city}")
                return False

            # Load predictions and training IDs
            df_predictions, train_panoids = self.load_prediction_data(city)
            if df_predictions.empty:
                self.logger.warning(f"No prediction data for {city}")
                return False

            # Process with different exclusion levels
            exclusion_levels = [None] + self.exclude_resolutions

            for res_exclude in exclusion_levels:
                self.logger.info(f"Processing with exclusion level: {res_exclude}")

                # Apply robust exclusion and aggregate
                df_aggregated = self.apply_robust_exclusion(
                    df_predictions, df_pano, train_panoids, res_exclude
                )

                # Compute statistics
                stats = self.compute_statistics(df_aggregated)

                # Save results
                output_file = (
                    Path(self.config["CURATE_FOLDER_EXPORT"])
                    / f"prob_city={city}_res_exclude={str(res_exclude)}.parquet"
                )

                df_aggregated.to_parquet(output_file, index=False, compression="snappy")
                self.logger.info(f"Saved results to {output_file}")

                # Save statistics
                if stats:
                    stats_file = output_file.with_suffix(".json")
                    import json

                    with open(stats_file, "w") as f:
                        json.dump(stats, f, indent=2)

            # Clean up memory
            gc.collect()

            self.logger.info(f"Successfully processed {city}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to process {city}: {e}", exc_info=True)
            return False

    def get_processed_cities(self) -> Set[str]:
        """
        Get list of already processed cities.

        Returns:
            Set of processed city names
        """
        existing_files = list(
            Path(self.config["CURATE_FOLDER_EXPORT"]).glob("*.parquet")
        )

        processed = set()
        for file_path in existing_files:
            parts = file_path.stem.split("_")
            for part in parts:
                if part.startswith("city="):
                    city = part.replace("city=", "")
                    if city not in self.cities_to_overwrite:
                        processed.add(city)
                    break

        return processed

    def run(
        self,
        city: str = "all",
        city_meta_path: str = None,
        resume: bool = False,
        parallel: bool = False,
    ) -> None:
        """
        Main processing pipeline.

        Args:
            city: City name or "all" for all cities
            city_meta_path: Path to city metadata CSV
            resume: Whether to resume from checkpoint
            parallel: Whether to use parallel processing
        """
        self.logger.info(f"Starting H3 aggregation pipeline for: {city}")

        try:
            if city == "all":
                # Process all cities
                if not city_meta_path:
                    city_meta_path = (
                        "/home/yuanzf/uvi-time-machine/_script/city_meta.csv"
                    )

                city_df = pd.read_csv(city_meta_path)
                all_cities = city_df["City"].values

                # Get already processed cities
                processed = self.get_processed_cities()
                remaining = [c for c in all_cities if c not in processed]

                self.logger.info(
                    f"Total cities: {len(all_cities)}, "
                    f"Processed: {len(processed)}, "
                    f"Remaining: {len(remaining)}"
                )

                # Check for checkpoint
                start_idx = 0
                if resume:
                    checkpoint = self.checkpoint_manager.load_checkpoint("all_cities")
                    if checkpoint:
                        start_idx = checkpoint.get("last_index", 0)
                        self.logger.info(f"Resuming from index {start_idx}")

                # Process remaining cities
                success_count = 0
                fail_count = 0

                for idx, city_name in enumerate(
                    tqdm(remaining[start_idx:], desc="Processing cities")
                ):
                    # Check again in case processed by another process
                    processed = self.get_processed_cities()
                    if city_name in processed:
                        self.logger.info(
                            f"{city_name} already processed by another process"
                        )
                        continue

                    success = self.process_city(city_name)
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1

                    # Save checkpoint
                    if (idx + 1) % 5 == 0:
                        self.checkpoint_manager.save_checkpoint(
                            {"last_index": start_idx + idx + 1}, "all_cities"
                        )

                self.logger.info(
                    f"Processing complete. Success: {success_count}, Failed: {fail_count}"
                )

                # Clean up checkpoint
                if resume:
                    self.checkpoint_manager.remove_checkpoint("all_cities")

            else:
                # Process single city
                success = self.process_city(city, force_overwrite=True)
                if not success:
                    raise RuntimeError(f"Failed to process {city}")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            self.db_manager.close()
            gc.collect()


def main():
    """Main entry point with CLI interface."""

    parser = argparse.ArgumentParser(description="H3 Hexagon Aggregation Processor")

    parser.add_argument(
        "--city",
        type=str,
        default="Hong Kong",
        help="City name or 'all' for all cities",
    )

    parser.add_argument(
        "--city-meta",
        default="/home/yuanzf/uvi-time-machine/_script/city_meta.csv",
        help="Path to city metadata CSV",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument("--config", help="Path to configuration JSON file")

    parser.add_argument(
        "--overwrite",
        nargs="+",
        default=["Santiago", "Minneapolis"],
        help="Cities to overwrite even if processed",
    )

    args = parser.parse_args()

    # Configuration
    config = {
        "ROOTFOLDER": "/lustre1/g/geog_pyloo/05_timemachine",
        "VALFOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir",
        "CURATED_FOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob",
        "TRAIN_TEST_FOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8",
        "CURATE_FOLDER_EXPORT": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        "PANO_PATH": "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv",
        "PATH_PATH": "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv",
        "summary_resolutions": [6, 7, 8],
        "exclude_resolutions": [11, 12, 13],
        "cities_to_overwrite": args.overwrite,
    }

    # Load custom config if provided
    if args.config and Path(args.config).exists():
        import json

        with open(args.config, "r") as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Initialize and run processor
    processor = H3HexagonAggregator(config, log_level=args.log_level)

    processor.run(city=args.city, city_meta_path=args.city_meta, resume=args.resume)


if __name__ == "__main__":
    main()
