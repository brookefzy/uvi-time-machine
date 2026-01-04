#!/usr/bin/env python
"""
Urban Analysis Utilities
Shared utilities for urban similarity and distance processing pipelines.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import hashlib
import json

import duckdb
import pandas as pd
import numpy as np


class UrbanDataConfig:
    """Configuration management for urban data processing."""

    DEFAULT_PATHS = {
        "ROOTFOLDER": "/lustre1/g/geog_pyloo/05_timemachine",
        "VALFOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir",
        "CURATED_FOLDER": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob",
        "CURATE_FOLDER_SOURCE": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        "CURATE_FOLDER_EXPORT": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity",
    }

    def __init__(
        self, config_file: Optional[str] = None, overrides: Optional[Dict] = None
    ):
        """
        Initialize configuration from file or defaults.

        Args:
            config_file: Optional path to JSON config file
            overrides: Dictionary of config values to override
        """
        self.config = self.DEFAULT_PATHS.copy()

        if config_file and Path(config_file).exists():
            with open(config_file, "r") as f:
                file_config = json.load(f)
                self.config.update(file_config)

        if overrides:
            self.config.update(overrides)

        # Add timestamp-based folders
        self.config["TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config["TODAY"] = datetime.now().strftime("%Y%m%d")

    def get(self, key: str, default: any = None) -> any:
        """Get configuration value."""
        return self.config.get(key, default)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)


class DuckDBManager:
    """Manager for DuckDB connections and operations."""

    def __init__(
        self, database: str = ":memory:", logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DuckDB connection manager.

        Args:
            database: Database path or ':memory:' for in-memory
            logger: Optional logger instance
        """
        self.database = database
        self.conn = duckdb.connect(database)
        self.logger = logger or logging.getLogger(__name__)
        self._setup_extensions()

    def _setup_extensions(self) -> None:
        """Setup useful DuckDB extensions."""
        extensions = ["parquet", "csv"]

        for ext in extensions:
            try:
                self.conn.execute(f"INSTALL {ext}")
                self.conn.execute(f"LOAD {ext}")
                self.logger.debug(f"Loaded DuckDB extension: {ext}")
            except Exception as e:
                self.logger.debug(f"Extension {ext} not available: {e}")

    def load_parquet_files(
        self, files: List[Path], filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load multiple parquet files efficiently using DuckDB.

        Args:
            files: List of parquet file paths
            filters: Optional filters to apply (e.g., {'res': [6, 7]})

        Returns:
            Combined DataFrame
        """
        if not files:
            return pd.DataFrame()

        # Build query with UNION ALL
        query_parts = []

        for idx, file_path in enumerate(files):
            view_name = f"temp_view_{idx}"

            # Create view for file
            self.conn.execute(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS 
                SELECT * FROM read_parquet('{file_path}')
            """
            )

            # Add filters if provided
            if filters:
                filter_conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ", ".join(map(str, values))
                        filter_conditions.append(f"{col} IN ({values_str})")
                    else:
                        filter_conditions.append(f"{col} = {values}")

                where_clause = " WHERE " + " AND ".join(filter_conditions)
                query_parts.append(f"SELECT * FROM {view_name}{where_clause}")
            else:
                query_parts.append(f"SELECT * FROM {view_name}")

        # Combine all queries
        if query_parts:
            union_query = " UNION ALL ".join(query_parts)
            result = self.conn.execute(union_query).fetchdf()

            # Clean up views
            for idx in range(len(files)):
                self.conn.execute(f"DROP VIEW IF EXISTS temp_view_{idx}")

            return result

        return pd.DataFrame()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        return self.conn.execute(query).fetchdf()

    def register_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Register a DataFrame as a virtual table."""
        self.conn.register(name, df)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


class DataValidator:
    """Validate data quality and integrity."""

    @staticmethod
    def validate_hexagon_data(
        df: pd.DataFrame, required_cols: List[str] = None
    ) -> bool:
        """
        Validate hexagon data structure and content.

        Args:
            df: DataFrame to validate
            required_cols: List of required column names

        Returns:
            True if valid, raises ValueError otherwise
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        default_required = ["hex_id", "res"]
        required_cols = required_cols or default_required

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for invalid hex IDs
        if "hex_id" in df.columns:
            null_hex = df["hex_id"].isnull().sum()
            if null_hex > 0:
                raise ValueError(f"Found {null_hex} null hex_id values")

        # Check resolution values
        if "res" in df.columns:
            valid_res = set(range(0, 16))  # H3 resolutions 0-15
            invalid_res = set(df["res"].unique()) - valid_res
            if invalid_res:
                raise ValueError(f"Invalid resolution values: {invalid_res}")

        return True

    @staticmethod
    def validate_coordinates(
        df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"
    ) -> bool:
        """
        Validate geographic coordinates.

        Args:
            df: DataFrame with coordinate columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            True if valid, raises ValueError otherwise
        """
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Missing coordinate columns: {lat_col} or {lon_col}")

        # Check latitude bounds (-90 to 90)
        invalid_lat = df[(df[lat_col] < -90) | (df[lat_col] > 90)]
        if not invalid_lat.empty:
            raise ValueError(f"Found {len(invalid_lat)} invalid latitude values")

        # Check longitude bounds (-180 to 180)
        invalid_lon = df[(df[lon_col] < -180) | (df[lon_col] > 180)]
        if not invalid_lon.empty:
            raise ValueError(f"Found {len(invalid_lon)} invalid longitude values")

        return True


class ProgressTracker:
    """Track and report processing progress."""

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            desc: Description of the task
            logger: Optional logger for progress updates
        """
        self.total = total
        self.current = 0
        self.desc = desc
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = datetime.now()

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
        if self.current % max(1, self.total // 10) == 0:  # Log every 10%
            pct = (self.current / self.total) * 100
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            self.logger.info(
                f"{self.desc}: {pct:.1f}% ({self.current}/{self.total}) - {rate:.1f} items/sec"
            )

    def finish(self) -> None:
        """Mark task as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.desc} complete: {self.total} items in {elapsed:.1f} seconds"
        )


class CheckpointManager:
    """Manage checkpoints for long-running processes."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, data: Dict, name: str) -> None:
        """
        Save checkpoint data.

        Args:
            data: Dictionary of data to save
            name: Checkpoint name/identifier
        """
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.json"

        # Convert numpy arrays and other non-serializable objects
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                serializable_data[key] = value.to_dict("records")
            else:
                serializable_data[key] = value

        with open(checkpoint_file, "w") as f:
            json.dump(serializable_data, f)

    def load_checkpoint(self, name: str) -> Optional[Dict]:
        """
        Load checkpoint data if exists.

        Args:
            name: Checkpoint name/identifier

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.json"

        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                return json.load(f)

        return None

    def remove_checkpoint(self, name: str) -> None:
        """Remove checkpoint file."""
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()


def calculate_file_hash(filepath: Union[str, Path]) -> str:
    """
    Calculate SHA256 hash of a file for integrity checking.

    Args:
        filepath: Path to file

    Returns:
        Hexadecimal hash string
    """
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def batch_process(
    items: List, batch_size: int, process_func: callable, desc: str = "Batch processing"
):
    """
    Process items in batches with progress tracking.

    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        desc: Description for progress tracking

    Returns:
        List of results from all batches
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_num = i // batch_size + 1

        logging.info(f"{desc}: Processing batch {batch_num}/{total_batches}")
        batch_results = process_func(batch)
        results.extend(batch_results)

    return results
