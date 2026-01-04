#!/usr/bin/env python
"""
H3 Hexagon Distance Processor
Computes distances between H3 hexagons and urban centers, as well as pairwise distances
between hexagons using efficient DuckDB operations and vectorized calculations.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import gc

import duckdb
import pandas as pd
import numpy as np
import h3
import haversine as hs
from haversine import Unit
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm


class H3DistanceProcessor:
    """Process H3 hexagon distance calculations using DuckDB and vectorized operations."""

    def __init__(self, config: Dict[str, any], log_level: str = "INFO"):
        """
        Initialize the processor with configuration and logging.

        Args:
            config: Configuration dictionary with paths and parameters
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config = config
        self.setup_logging(log_level)
        self.conn = duckdb.connect(":memory:")
        self.setup_directories()

        # Install and load spatial extension for DuckDB if available
        self._setup_duckdb_extensions()

    def setup_logging(self, log_level: str) -> None:
        """Configure logging with both file and console handlers."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"h3_distance_{timestamp}.log"

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [self.config["CURATE_FOLDER_EXPORT"]]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def _setup_duckdb_extensions(self) -> None:
        """Setup DuckDB extensions for spatial operations if available."""
        try:
            self.conn.execute("INSTALL spatial")
            self.conn.execute("LOAD spatial")
            self.logger.info("DuckDB spatial extension loaded successfully")
        except Exception as e:
            self.logger.debug(f"Spatial extension not available: {e}")

    def load_city_metadata(self, city_meta_path: str) -> pd.DataFrame:
        """
        Load and preprocess city metadata.

        Args:
            city_meta_path: Path to city metadata CSV file

        Returns:
            DataFrame with city metadata including normalized city names
        """
        self.logger.info(f"Loading city metadata from: {city_meta_path}")

        # Use DuckDB to load and process city metadata
        query = f"""
            SELECT 
                center_lat,
                center_lng,
                City,
                LOWER(REPLACE(City, ' ', '')) as city_lower
            FROM read_csv_auto('{city_meta_path}')
            WHERE center_lat IS NOT NULL 
                AND center_lng IS NOT NULL
                AND City IS NOT NULL
        """

        city_meta = self.conn.execute(query).fetchdf()
        self.logger.info(f"Loaded metadata for {len(city_meta)} cities")

        return city_meta

    def load_hexagon_data(self, resolutions: List[int]) -> pd.DataFrame:
        """
        Load hexagon data from multiple files using DuckDB for efficiency.

        Args:
            resolutions: List of H3 resolutions to include

        Returns:
            DataFrame with hexagon data
        """
        self.logger.info(f"Loading hexagon data for resolutions: {resolutions}")

        # Get list of files
        pattern = f"/*res_exclude={self.config['RES_EXCLUDE']}.parquet"
        files = list(
            Path(self.config["CURATE_FOLDER_SOURCE"]).glob(pattern.lstrip("/"))
        )

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        self.logger.info(f"Found {len(files)} files to process")

        # Build query to load and filter data
        query_parts = []
        for idx, file_path in enumerate(tqdm(files, desc="Loading hexagon files")):
            # Extract city name from filename
            city_name = (
                file_path.stem.split("_")[1]
                .replace("city=", "")
                .replace(" ", "")
                .lower()
            )

            # Create a view for each file
            view_name = f"hex_data_{idx}"
            res_filter = ", ".join(map(str, resolutions))

            self.conn.execute(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT 
                    hex_id,
                    res,
                    '{city_name}' as city_lower
                FROM read_parquet('{file_path}')
                WHERE res IN ({res_filter})
            """
            )

            query_parts.append(f"SELECT * FROM {view_name}")

        # Combine all data and remove duplicates
        if query_parts:
            union_query = " UNION ALL ".join(query_parts)
            final_query = f"""
                SELECT DISTINCT hex_id, res, city_lower
                FROM ({union_query})
            """

            df_all = self.conn.execute(final_query).fetchdf()

            # Clean up views
            for idx in range(len(files)):
                self.conn.execute(f"DROP VIEW IF EXISTS hex_data_{idx}")

            self.logger.info(f"Loaded {len(df_all)} unique hexagons")
            return df_all

        return pd.DataFrame()

    def compute_hex_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add latitude and longitude coordinates for hexagon centers.

        Args:
            df: DataFrame with hex_id column

        Returns:
            DataFrame with added lat/lon columns
        """
        self.logger.info("Computing hexagon center coordinates")

        # Vectorized computation using list comprehension (faster than apply)
        coords = [
            h3.cell_to_latlng(hex_id)
            for hex_id in tqdm(df["hex_id"], desc="Computing coordinates")
        ]
        df["lat"] = [coord[0] for coord in coords]
        df["lon"] = [coord[1] for coord in coords]

        self.logger.info(f"Computed coordinates for {len(df)} hexagons")
        return df

    def compute_cbd_distances(
        self, df_hex: pd.DataFrame, city_meta: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute distances from hexagons to city centers (CBD).

        Args:
            df_hex: DataFrame with hexagon data including coordinates
            city_meta: DataFrame with city metadata including center coordinates

        Returns:
            DataFrame with CBD distances added
        """
        self.logger.info("Computing distances to city centers")

        # Register dataframes with DuckDB
        self.conn.register("df_hex", df_hex)
        self.conn.register("city_meta", city_meta)

        # Merge using DuckDB
        query = """
            SELECT 
                h.*,
                c.center_lat,
                c.center_lng,
                c.City
            FROM df_hex h
            INNER JOIN city_meta c ON h.city_lower = c.city_lower
        """

        df_merged = self.conn.execute(query).fetchdf()

        # Vectorized distance computation
        self.logger.info(
            f"Computing CBD distances for {len(df_merged)} hexagon-city pairs"
        )

        # Prepare coordinate arrays
        hex_coords = df_merged[["lat", "lon"]].values
        cbd_coords = df_merged[["center_lat", "center_lng"]].values

        # Compute distances in batch (more efficient than row-by-row)
        distances = []
        batch_size = 10000

        for i in tqdm(
            range(0, len(df_merged), batch_size), desc="Computing CBD distances"
        ):
            batch_end = min(i + batch_size, len(df_merged))
            batch_distances = [
                hs.haversine(
                    (hex_coords[j][0], hex_coords[j][1]),
                    (cbd_coords[j][0], cbd_coords[j][1]),
                    unit=Unit.METERS,
                )
                for j in range(i, batch_end)
            ]
            distances.extend(batch_distances)

        df_merged["h3_cbd_dist"] = distances

        self.logger.info("CBD distance computation complete")
        return df_merged

    def compute_pairwise_distances(
        self, df_hex: pd.DataFrame, resolution: int
    ) -> pd.DataFrame:
        """
        Compute pairwise distances between all hexagons at a given resolution.

        Args:
            df_hex: DataFrame with hexagon data including coordinates
            resolution: H3 resolution level

        Returns:
            DataFrame with pairwise distances
        """
        self.logger.info(f"Computing pairwise distances for resolution {resolution}")

        # Filter for specific resolution
        df_current = (
            df_hex[df_hex["res"] == resolution]
            .drop_duplicates("hex_id")
            .reset_index(drop=True)
        )

        if len(df_current) == 0:
            self.logger.warning(f"No hexagons found for resolution {resolution}")
            return pd.DataFrame()

        n_hexagons = len(df_current)
        self.logger.info(
            f"Computing distances for {n_hexagons} hexagons ({n_hexagons * (n_hexagons - 1) // 2} pairs)"
        )

        # Convert lat/lon to radians for sklearn (it expects radians)
        coords_rad = np.radians(df_current[["lat", "lon"]].values)

        # Compute distance matrix using sklearn (returns distances in radians)
        dist_matrix_rad = haversine_distances(coords_rad, coords_rad)

        # Convert from radians to kilometers (Earth's radius ≈ 6371 km)
        dist_matrix_km = dist_matrix_rad * 6371

        # Keep only upper triangle (avoid duplicates)
        dist_matrix_upper = np.triu(dist_matrix_km, k=1)

        # Convert to DataFrame using DuckDB for efficient processing
        self.logger.info("Converting distance matrix to DataFrame")

        # Get hexagon IDs
        hex_ids = df_current["hex_id"].values

        # Create pairs with distances > 0 (upper triangle only)
        pairs = []
        for i in range(len(hex_ids)):
            for j in range(i + 1, len(hex_ids)):
                if dist_matrix_upper[i, j] > 0:
                    pairs.append(
                        {
                            "hex_id1": hex_ids[i],
                            "hex_id2": hex_ids[j],
                            "dist_pair": dist_matrix_upper[i, j],
                        }
                    )

        df_distances = pd.DataFrame(pairs)

        self.logger.info(f"Generated {len(df_distances)} unique hexagon pairs")
        return df_distances

    def save_results(
        self, df: pd.DataFrame, filename: str, format: str = "parquet"
    ) -> None:
        """
        Save DataFrame to file with logging.

        Args:
            df: DataFrame to save
            filename: Output filename
            format: File format ('parquet' or 'csv')
        """
        output_path = Path(self.config["CURATE_FOLDER_EXPORT"]) / filename

        try:
            if format == "parquet":
                df.to_parquet(output_path, index=False)
            elif format == "csv":
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Saved {len(df)} rows to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save file {filename}: {e}")
            raise

    def run(
        self, city_meta_path: str, resolutions: List[int], compute_pairwise: bool = True
    ) -> None:
        """
        Main processing pipeline.

        Args:
            city_meta_path: Path to city metadata CSV
            resolutions: List of H3 resolutions to process
            compute_pairwise: Whether to compute pairwise distances
        """
        self.logger.info("Starting H3 distance processing pipeline")
        self.logger.info(
            f"Parameters: resolutions={resolutions}, compute_pairwise={compute_pairwise}"
        )

        try:
            # Step 1: Load city metadata
            city_meta = self.load_city_metadata(city_meta_path)

            # Step 2: Load hexagon data
            df_hex = self.load_hexagon_data(resolutions)

            if df_hex.empty:
                self.logger.warning("No hexagon data loaded. Exiting.")
                return

            # Step 3: Compute hexagon coordinates
            df_hex = self.compute_hex_coordinates(df_hex)

            # Step 4: Compute CBD distances
            df_cbd = self.compute_cbd_distances(df_hex, city_meta)

            # Save CBD distances
            cbd_filename = (
                f"c_city_cbd_dist_res_exclude={self.config['RES_EXCLUDE']}.csv"
            )
            self.save_results(df_cbd, cbd_filename, format="csv")

            # Step 5: Compute pairwise distances if requested
            if compute_pairwise:
                for resolution in resolutions:
                    self.logger.info(
                        f"Processing pairwise distances for resolution {resolution}"
                    )

                    df_pairwise = self.compute_pairwise_distances(df_hex, resolution)

                    if not df_pairwise.empty:
                        pairwise_filename = (
                            f"h3_hex_pairwise_dist_res={resolution}.parquet"
                        )
                        self.save_results(
                            df_pairwise, pairwise_filename, format="parquet"
                        )

                    # Clean up memory after each resolution
                    gc.collect()

            self.logger.info("Pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

        finally:
            # Clean up
            self.conn.close()
            gc.collect()
            self.logger.info("Cleanup complete")


def main():
    """Main entry point for the script."""

    # Configuration
    config = {
        "RES_EXCLUDE": 11,
        "CURATE_FOLDER_SOURCE": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        "CURATE_FOLDER_EXPORT": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity",
    }

    # Initialize processor
    processor = H3DistanceProcessor(config, log_level="INFO")

    # Run the pipeline for resolutions 6 and 7
    processor.run(
        city_meta_path="../city_meta.csv",
        resolutions=[6, 7],
        compute_pairwise=True,
    )


if __name__ == "__main__":
    main()
