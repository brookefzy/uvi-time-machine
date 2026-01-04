#!/usr/bin/env python
"""
City Pair Similarity Processor
Computes pairwise cosine similarity between cities using H3 hexagon features
with efficient DuckDB operations and parallel processing support.
"""

import os
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import gc
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from urban_utils import (
    UrbanDataConfig,
    DuckDBManager,
    DataValidator,
    CheckpointManager,
    ProgressTracker,
    batch_process,
)


class CityPairSimilarityProcessor:
    """Process pairwise similarity between cities using vectorized features."""

    def __init__(self, config: Dict[str, any], log_level: str = "INFO"):
        """
        Initialize the processor with configuration and logging.

        Args:
            config: Configuration dictionary with paths and parameters
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config = config
        self.setup_logging(log_level)
        self.db_manager = DuckDBManager(logger=self.logger)
        self.checkpoint_manager = CheckpointManager("checkpoints/similarity_pairs")
        self.setup_directories()
        self.vector_columns = [str(x) for x in range(127)]

    def setup_logging(self, log_level: str) -> None:
        """Configure logging with both file and console handlers."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"city_pair_similarity_{timestamp}.log"

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config["CURATE_FOLDER_EXPORT"],
            Path(self.config["CURATE_FOLDER_EXPORT"]) / "temp",
            "checkpoints/similarity_pairs",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def create_city_pairs(
        self, city_meta_path: str, exclude_self: bool = True
    ) -> np.ndarray:
        """
        Create all unique city pairs for comparison.

        Args:
            city_meta_path: Path to city metadata CSV
            exclude_self: Whether to exclude same-city pairs

        Returns:
            Array of city pairs
        """
        self.logger.info(f"Creating city pairs from: {city_meta_path}")

        # Load city metadata using DuckDB
        query = f"""
            SELECT DISTINCT City 
            FROM read_csv_auto('{city_meta_path}')
            WHERE City IS NOT NULL
            ORDER BY City
        """

        city_df = self.db_manager.execute_query(query)
        cities = city_df["City"].values
        self.logger.info(f"Found {len(cities)} cities")

        # Generate all pairs
        if exclude_self:
            # Create pairs excluding self-pairs
            pairs = []
            for i, city1 in enumerate(cities):
                for city2 in cities[i + 1 :]:  # Only upper triangle to avoid duplicates
                    pairs.append([city1, city2])
            pairs = np.array(pairs)
        else:
            # Create all pairs including self
            pairs = np.array(np.meshgrid(cities, cities)).T.reshape(-1, 2)

        self.logger.info(f"Created {len(pairs)} city pairs")
        return pairs

    def load_city_features(self, city_name: str, resolution: int) -> pd.DataFrame:
        """
        Load feature vectors for a specific city using DuckDB.

        Args:
            city_name: Name of the city
            resolution: H3 resolution level

        Returns:
            DataFrame with hex_id and feature vectors
        """
        pattern = (
            f"prob_city={city_name}_res_exclude={self.config['RES_EXCLUDE']}.parquet"
        )
        file_path = Path(self.config["CURATE_FOLDER_SOURCE"]) / pattern

        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        # Load using DuckDB for efficiency
        query = f"""
            SELECT hex_id, {', '.join([f'"{col}"' for col in self.vector_columns])}
            FROM read_parquet('{file_path}')
            WHERE res = {resolution}
        """

        try:
            df = self.db_manager.execute_query(query)
            df["city"] = city_name
            self.logger.debug(f"Loaded {len(df)} features for {city_name}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading features for {city_name}: {e}")
            return pd.DataFrame()

    def compute_similarity_matrix_batch(
        self, features1: np.ndarray, features2: np.ndarray, batch_size: int = 1000
    ) -> np.ndarray:
        """
        Compute cosine similarity in batches to manage memory.

        Args:
            features1: First feature matrix
            features2: Second feature matrix
            batch_size: Size of batches for computation

        Returns:
            Similarity matrix
        """
        n1, n2 = features1.shape[0], features2.shape[0]
        similarity_matrix = np.zeros((n1, n2))

        # Process in batches to manage memory
        for i in range(0, n1, batch_size):
            end_i = min(i + batch_size, n1)
            batch1 = features1[i:end_i]

            for j in range(0, n2, batch_size):
                end_j = min(j + batch_size, n2)
                batch2 = features2[j:end_j]

                # Compute similarity for this batch
                similarity_matrix[i:end_i, j:end_j] = cosine_similarity(batch1, batch2)

        return similarity_matrix

    def compute_city_pair_similarity(
        self, city1: str, city2: str, resolution: int
    ) -> pd.DataFrame:
        """
        Compute similarity between two cities.

        Args:
            city1: First city name
            city2: Second city name
            resolution: H3 resolution level

        Returns:
            DataFrame with similarity scores
        """
        self.logger.info(
            f"Computing similarity: {city1} <-> {city2} at resolution {resolution}"
        )

        # Load features for both cities
        df1 = self.load_city_features(city1, resolution)
        df2 = self.load_city_features(city2, resolution)

        if df1.empty or df2.empty:
            self.logger.warning(f"Missing data for {city1} or {city2}")
            return pd.DataFrame()

        # Combine and deduplicate if same city
        if city1 == city2:
            df_combined = df1
        else:
            df_combined = (
                pd.concat([df1, df2]).drop_duplicates("hex_id").reset_index(drop=True)
            )

        self.logger.debug(f"Combined features shape: {df_combined.shape}")

        # Extract feature matrices
        features = df_combined[self.vector_columns].values
        hex_ids = df_combined["hex_id"].values

        # Compute similarity matrix
        if len(features) > 5000:  # Use batch processing for large matrices
            self.logger.info(f"Using batch processing for {len(features)} features")
            similarity_matrix = self.compute_similarity_matrix_batch(features, features)
        else:
            similarity_matrix = cosine_similarity(features)

        # Keep only upper triangle to avoid duplicates
        similarity_matrix = np.triu(similarity_matrix, k=1)

        # Convert to sparse format for efficient storage
        results = []
        for i in range(len(hex_ids)):
            for j in range(i + 1, len(hex_ids)):
                if similarity_matrix[i, j] > 0:  # Only store non-zero similarities
                    results.append(
                        {
                            "hex_id1": hex_ids[i],
                            "hex_id2": hex_ids[j],
                            "similarity": similarity_matrix[i, j],
                        }
                    )

        similarity_df = pd.DataFrame(results)

        # Clean up memory
        del similarity_matrix
        del features
        gc.collect()

        self.logger.info(f"Computed {len(similarity_df)} similarity pairs")
        return similarity_df

    def save_temp_similarity(
        self, similarity_df: pd.DataFrame, city1: str, city2: str, resolution: int
    ) -> None:
        """
        Save temporary similarity results.

        Args:
            similarity_df: DataFrame with similarity scores
            city1: First city name
            city2: Second city name
            resolution: H3 resolution level
        """
        if similarity_df.empty:
            return

        temp_dir = Path(self.config["CURATE_FOLDER_EXPORT"]) / "temp" / f"city1={city1}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        output_file = temp_dir / f"similarity_city2={city2}_res={resolution}.parquet"
        similarity_df.to_parquet(output_file, index=False)

        self.logger.debug(f"Saved temp similarity to: {output_file}")

    def merge_and_deduplicate(self, city: str, resolution: int) -> pd.DataFrame:
        """
        Merge all similarity files for a city and remove duplicates using DuckDB.

        Args:
            city: City name
            resolution: H3 resolution level

        Returns:
            Deduplicated similarity DataFrame
        """
        self.logger.info(f"Merging and deduplicating results for {city}")

        temp_dir = Path(self.config["CURATE_FOLDER_EXPORT"]) / "temp" / f"city1={city}"
        pattern = f"similarity_city2=*_res={resolution}.parquet"
        files = list(temp_dir.glob(pattern))

        if not files:
            self.logger.warning(f"No temporary files found for {city}")
            return pd.DataFrame()

        self.logger.info(f"Found {len(files)} files to merge")

        # Use DuckDB for efficient merging and deduplication
        query_parts = []
        for idx, file_path in enumerate(files):
            view_name = f"temp_sim_{idx}"
            self.db_manager.conn.execute(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{file_path}')
            """
            )
            query_parts.append(f"SELECT * FROM {view_name}")

        # Combine all and deduplicate
        union_query = " UNION ALL ".join(query_parts)
        dedup_query = f"""
            WITH combined AS ({union_query}),
            deduped AS (
                SELECT 
                    hex_id1,
                    hex_id2,
                    MAX(similarity) as similarity
                FROM combined
                WHERE hex_id1 != hex_id2
                GROUP BY hex_id1, hex_id2
            )
            SELECT * FROM deduped
            ORDER BY similarity DESC
        """

        result_df = self.db_manager.execute_query(dedup_query)

        # Clean up views
        for idx in range(len(files)):
            self.db_manager.conn.execute(f"DROP VIEW IF EXISTS temp_sim_{idx}")

        self.logger.info(f"Merged {len(result_df)} unique similarity pairs")
        return result_df

    def cleanup_temp_files(self, city: str) -> None:
        """
        Remove temporary files for a city.

        Args:
            city: City name
        """
        temp_dir = Path(self.config["CURATE_FOLDER_EXPORT"]) / "temp" / f"city1={city}"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            self.logger.debug(f"Cleaned up temp files for {city}")

    def process_city_batch(
        self, city_pairs: List[Tuple[str, str]], resolution: int
    ) -> Dict[str, int]:
        """
        Process a batch of city pairs.

        Args:
            city_pairs: List of (city1, city2) tuples
            resolution: H3 resolution level

        Returns:
            Statistics dictionary
        """
        stats = {"processed": 0, "failed": 0, "total_similarities": 0}

        for city1, city2 in city_pairs:
            try:
                similarity_df = self.compute_city_pair_similarity(
                    city1, city2, resolution
                )

                if not similarity_df.empty:
                    self.save_temp_similarity(similarity_df, city1, city2, resolution)
                    stats["total_similarities"] += len(similarity_df)

                stats["processed"] += 1

            except Exception as e:
                self.logger.error(f"Failed to process {city1}-{city2}: {e}")
                stats["failed"] += 1

        return stats

    def run(
        self,
        city_meta_path: str,
        resolution: int = 6,
        parallel: bool = False,
        max_workers: int = 4,
        resume: bool = False,
    ) -> None:
        """
        Main processing pipeline.

        Args:
            city_meta_path: Path to city metadata CSV
            resolution: H3 resolution level
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            resume: Whether to resume from checkpoint
        """
        self.logger.info(f"Starting city pair similarity processing")
        self.logger.info(f"Parameters: resolution={resolution}, parallel={parallel}")

        try:
            # Create city pairs
            city_pairs = self.create_city_pairs(city_meta_path, exclude_self=True)

            # Check for checkpoint
            start_idx = 0
            if resume:
                checkpoint = self.checkpoint_manager.load_checkpoint(
                    f"res_{resolution}"
                )
                if checkpoint:
                    start_idx = checkpoint.get("last_index", 0)
                    self.logger.info(f"Resuming from index {start_idx}")

            # Process pairs
            total_pairs = len(city_pairs)

            if parallel and total_pairs > 10:
                self._process_parallel(city_pairs[start_idx:], resolution, max_workers)
            else:
                self._process_sequential(city_pairs[start_idx:], resolution, start_idx)

            # Merge and deduplicate for each city
            self.logger.info("Starting merge and deduplication phase")
            unique_cities = np.unique(city_pairs.flatten())

            for city in tqdm(unique_cities, desc="Merging city results"):
                result_df = self.merge_and_deduplicate(city, resolution)

                if not result_df.empty:
                    output_file = (
                        Path(self.config["CURATE_FOLDER_EXPORT"])
                        / f"similarity_city={city}_res={resolution}.parquet"
                    )
                    result_df.to_parquet(output_file, index=False)
                    self.logger.info(
                        f"Saved final results for {city}: {len(result_df)} pairs"
                    )

                self.cleanup_temp_files(city)

            # Clean up checkpoint
            if resume:
                self.checkpoint_manager.remove_checkpoint(f"res_{resolution}")

            self.logger.info("Processing completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

        finally:
            self.db_manager.close()
            gc.collect()

    def _process_sequential(
        self, city_pairs: np.ndarray, resolution: int, start_offset: int = 0
    ) -> None:
        """Process city pairs sequentially."""
        total_stats = {"processed": 0, "failed": 0, "total_similarities": 0}

        for idx, (city1, city2) in enumerate(
            tqdm(city_pairs, desc="Processing city pairs")
        ):
            try:
                similarity_df = self.compute_city_pair_similarity(
                    city1, city2, resolution
                )

                if not similarity_df.empty:
                    self.save_temp_similarity(similarity_df, city1, city2, resolution)
                    total_stats["total_similarities"] += len(similarity_df)

                total_stats["processed"] += 1

                # Save checkpoint every 10 pairs
                if (idx + 1) % 10 == 0:
                    self.checkpoint_manager.save_checkpoint(
                        {"last_index": start_offset + idx + 1}, f"res_{resolution}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to process {city1}-{city2}: {e}")
                total_stats["failed"] += 1

        self.logger.info(f"Sequential processing stats: {total_stats}")

    def _process_parallel(
        self, city_pairs: np.ndarray, resolution: int, max_workers: int
    ) -> None:
        """Process city pairs in parallel."""
        self.logger.info(f"Starting parallel processing with {max_workers} workers")

        # Split pairs into batches
        batch_size = max(1, len(city_pairs) // (max_workers * 4))
        batches = [
            city_pairs[i : i + batch_size]
            for i in range(0, len(city_pairs), batch_size)
        ]

        total_stats = {"processed": 0, "failed": 0, "total_similarities": 0}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_city_batch, batch, resolution): i
                for i, batch in enumerate(batches)
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                try:
                    batch_stats = future.result()
                    for key in total_stats:
                        total_stats[key] += batch_stats[key]
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")

        self.logger.info(f"Parallel processing stats: {total_stats}")


def main():
    """Main entry point with CLI interface."""

    parser = argparse.ArgumentParser(description="City Pair Similarity Processor")

    parser.add_argument(
        "--res-sel",
        type=int,
        default=6,
        help="H3 resolution level for similarity computation",
    )

    parser.add_argument(
        "--city-meta",
        default="/home/yuanzf/uvi-time-machine/_script/city_meta.csv",
        help="Path to city metadata CSV",
    )

    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel processing"
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
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

    args = parser.parse_args()

    # Configuration
    config = {
        "ROOTFOLDER": "/lustre1/g/geog_pyloo/05_timemachine",
        "CURATED_FOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob",
        "CURATE_FOLDER_SOURCE": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        "CURATE_FOLDER_EXPORT": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
        "RES_EXCLUDE": 11,
    }

    # Load custom config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Initialize and run processor
    processor = CityPairSimilarityProcessor(config, log_level=args.log_level)

    processor.run(
        city_meta_path=args.city_meta,
        resolution=args.res_sel,
        parallel=args.parallel,
        max_workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
