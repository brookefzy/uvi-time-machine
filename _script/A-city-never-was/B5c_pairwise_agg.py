#!/usr/bin/env python
"""
Urban Visual Similarity Processor
Processes hexagonal grid data to compute inter-city visual similarity metrics
using DuckDB for efficient memory management and data aggregation.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import gc

import duckdb
import pandas as pd
from tqdm import tqdm


class UrbanSimilarityProcessor:
    """Process urban visual similarity data using DuckDB for efficient computation."""

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

    def setup_logging(self, log_level: str) -> None:
        """Configure logging with both file and console handlers."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"urban_similarity_{timestamp}.log"

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
            self.config["EXPORT_FOLDER"],
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def load_hexagon_data(self) -> pd.DataFrame:
        """
        Load and aggregate hexagon data from multiple city files using DuckDB.

        Returns:
            DataFrame with hex_id to city mapping
        """
        self.logger.info("Loading hexagon data from source files...")

        # Get list of files
        pattern = f"/*res_exclude={self.config['RES_EXCLUDE']}.parquet"
        files = list(
            Path(self.config["CURATE_FOLDER_SOURCE"]).glob(pattern.lstrip("/"))
        )

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        self.logger.info(f"Found {len(files)} files to process")

        # Process files using DuckDB for efficiency
        query_parts = []
        for idx, file_path in enumerate(files):
            city_name = file_path.stem.split("_")[1].replace("city=", "")

            # Register parquet file with DuckDB
            table_name = f"city_data_{idx}"
            self.conn.execute(
                f"""
                CREATE OR REPLACE VIEW {table_name} AS 
                SELECT *, '{city_name}' as city 
                FROM read_parquet('{file_path}')
                WHERE res = {self.config['RES_SEL']}
            """
            )
            query_parts.append(f"SELECT * FROM {table_name}")

        # Combine all city data
        union_query = " UNION ALL ".join(query_parts)
        combined_query = f"""
            WITH combined AS ({union_query})
            SELECT DISTINCT hex_id, city, {', '.join([f'"{i}"' for i in range(127)])}
            FROM combined
            ORDER BY hex_id
        """

        df_all = self.conn.execute(combined_query).fetchdf()

        # Create mapping table
        df_map = df_all[["hex_id", "city"]].drop_duplicates().reset_index(drop=True)

        self.logger.info(
            f"Loaded {len(df_all)} unique hexagons from {len(files)} cities"
        )

        # Clean up memory
        gc.collect()

        return df_map

    def process_city_similarity(
        self, city: str, df_map: pd.DataFrame
    ) -> Tuple[int, int]:
        """
        Process similarity data for a single city using DuckDB.

        Args:
            city: City name to process
            df_map: DataFrame with hex_id to city mapping

        Returns:
            Tuple of (inner_city_count, inter_city_count)
        """
        self.logger.info(f"Processing similarity data for city: {city}")

        # Construct file path
        filename = f"similarity_city={city}_res={self.config['RES_SEL']}.parquet"
        filepath = Path(self.config["CURATE_FOLDER_EXPORT2"]) / filename

        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            return 0, 0

        try:
            # Load similarity data into DuckDB
            self.conn.execute(
                f"""
                CREATE OR REPLACE VIEW city_similarity AS 
                SELECT * FROM read_parquet('{filepath}')
            """
            )

            # Register mapping table
            self.conn.register("df_map", df_map)

            # Process with DuckDB: aggregate duplicates and join with city mapping
            query = """
                WITH deduped AS (
                    SELECT 
                        LEAST(hex_id1, hex_id2) as hex_1,
                        GREATEST(hex_id1, hex_id2) as hex_2,
                        MAX(similarity) as similarity
                    FROM city_similarity
                    GROUP BY LEAST(hex_id1, hex_id2), GREATEST(hex_id1, hex_id2)
                ),
                joined AS (
                    SELECT 
                        d.hex_1 as hex_id1,
                        d.hex_2 as hex_id2,
                        d.similarity,
                        m1.city as city_1,
                        m2.city as city_2
                    FROM deduped d
                    JOIN df_map m1 ON d.hex_1 = m1.hex_id
                    JOIN df_map m2 ON d.hex_2 = m2.hex_id
                )
                SELECT * FROM joined
            """

            result_df = self.conn.execute(query).fetchdf()

            # Split into inner-city and inter-city
            inner_city = result_df[result_df["city_1"] == result_df["city_2"]]
            inter_city = result_df[result_df["city_1"] != result_df["city_2"]]

            inner_count = len(inner_city)
            inter_count = len(inter_city)

            self.logger.info(
                f"City {city}: {inner_count} inner-city pairs, {inter_count} inter-city pairs"
            )

            # Save inter-city results if any
            if inter_count > 0:
                output_file = (
                    Path(self.config["EXPORT_FOLDER"])
                    / f"similarity_intracity_city={city}_res={self.config['RES_SEL']}.parquet"
                )
                inter_city.to_parquet(output_file, index=False)
                self.logger.debug(f"Saved inter-city results to: {output_file}")

            # Clean up views
            self.conn.execute("DROP VIEW IF EXISTS city_similarity")

            return inner_count, inter_count

        except Exception as e:
            self.logger.error(f"Error processing city {city}: {str(e)}")
            return 0, 0

    def run(self, city_meta_path: str) -> None:
        """
        Main processing pipeline.

        Args:
            city_meta_path: Path to city metadata CSV file
        """
        self.logger.info("Starting urban similarity processing pipeline")

        try:
            # Load city metadata
            city_meta = pd.read_csv(city_meta_path)
            cities = city_meta["City"].values
            self.logger.info(f"Processing {len(cities)} cities")

            # Load hexagon mapping data
            df_map = self.load_hexagon_data()

            # Process each city
            total_inner = 0
            total_inter = 0

            for city in tqdm(cities, desc="Processing cities"):
                inner_count, inter_count = self.process_city_similarity(city, df_map)
                total_inner += inner_count
                total_inter += inter_count

            self.logger.info(
                f"Processing complete. Total: {total_inner} inner-city pairs, {total_inter} inter-city pairs"
            )

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
        "ROOTFOLDER": "/lustre1/g/geog_pyloo/05_timemachine",
        "VALFOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir",
        "CURATED_FOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob",
        "TRAIN_TEST_FOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8",
        "CURATE_FOLDER_SOURCE": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        "CURATE_FOLDER_EXPORT": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity",
        "CURATE_FOLDER_EXPORT2": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
        "RES_EXCLUDE": 11,
        "RES_SEL": 7,
    }

    # Add dynamic export folder with timestamp
    today = datetime.now().strftime("%Y%m%d")
    config["EXPORT_FOLDER"] = (
        f"/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_{today}"
    )

    # Initialize and run processor
    processor = UrbanSimilarityProcessor(config, log_level="INFO")

    # Run the pipeline
    processor.run("../city_meta.csv")


if __name__ == "__main__":
    main()
