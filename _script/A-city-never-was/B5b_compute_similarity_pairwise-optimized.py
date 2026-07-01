#!/usr/bin/env python
"""
Optimized City Similarity Processor
Advanced implementation with sparse matrices, approximate algorithms, and GPU support.
"""

import os
import logging
import argparse
import warnings
import re
from pathlib import Path
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import json
import shutil

import duckdb
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Optional GPU support
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPU support not available. Install cupy for GPU acceleration.")


DEFAULT_CURATE_FOLDER_SOURCE = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary"
)
DEFAULT_CURATE_FOLDER_EXPORT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair"
)
DEFAULT_INPUT_TEMPLATE = "prob_city={city}_res_exclude={res_exclude}.parquet"


class OptimizedSimilarityProcessor:
    """Optimized similarity computation with multiple algorithmic improvements."""

    def __init__(self, config: Dict[str, Any], log_level: str = "INFO"):
        """
        Initialize the optimized processor.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.setup_logging(log_level)
        self.conn = duckdb.connect(":memory:")
        self.setup_duckdb_optimizations()
        self.source_root = Path(
            config.get("CURATE_FOLDER_SOURCE", DEFAULT_CURATE_FOLDER_SOURCE)
        )
        self.output_root = Path(
            config.get("CURATE_FOLDER_EXPORT", DEFAULT_CURATE_FOLDER_EXPORT)
        )
        self.input_template = config.get("INPUT_TEMPLATE", DEFAULT_INPUT_TEMPLATE)
        self.feature_prefix = config.get("FEATURE_PREFIX")
        if self.feature_prefix == "":
            self.feature_prefix = None
        configured_metric = config.get("METRIC_LABEL")
        self.metric_label = configured_metric if self.feature_prefix else None
        self.keep_temp = config.get("keep_temp", False)
        self._vector_contract_by_city: Dict[str, List[str]] = {}
        self.vector_columns = (
            self.resolve_vector_columns(None)
            if self.feature_prefix
            else [str(x) for x in range(127)]
        )

        # Optimization parameters
        self.use_gpu = config.get("use_gpu", False) and GPU_AVAILABLE
        self.use_sparse = config.get("use_sparse", True)
        self.use_approximation = config.get("use_approximation", False)
        self.approximation_dims = config.get("approximation_dims", 64)
        self.batch_size = config.get("batch_size", 5000)
        self.row_block_size = config.get("row_block_size", 1000)
        self.similarity_threshold = config.get("similarity_threshold", 0.01)
        self.resume_enabled = config.get("resume", True)

        self.logger.info(
            f"Optimization settings: GPU={self.use_gpu}, Sparse={self.use_sparse}, Approx={self.use_approximation}"
        )

    def setup_logging(self, log_level: str) -> None:
        """Configure optimized logging."""
        log_dir = self.get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = (log_dir / f"optimized_similarity_{timestamp}.log").resolve()

        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        self.logger.info("Logging to %s", self.log_file)

    def get_log_dir(self) -> Path:
        """Return the local directory used for run logs."""
        configured = self.config.get("log_dir")
        base_dir = Path(__file__).resolve().parent

        if not configured:
            return base_dir / "logs"

        log_dir = Path(configured).expanduser()
        if not log_dir.is_absolute():
            log_dir = base_dir / log_dir
        return log_dir.resolve()

    def get_output_dir(self) -> Path:
        """Return the directory where optimized parquet outputs are stored."""
        output_dir = self.output_root / "optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def build_input_path(self, city: str) -> Path:
        """Return the configured H3 vector parquet path for one city."""
        return self.source_root / self.input_template.format(
            city=city,
            res_exclude=self.config["RES_EXCLUDE"],
        )

    def _schema_columns(self, path: Path) -> List[str]:
        """Return parquet schema columns without materializing data when possible."""
        try:
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        except Exception:
            return list(pd.read_parquet(path).columns)

    def resolve_vector_columns(
        self, df_or_path: Optional[Union[pd.DataFrame, Path, str]]
    ) -> List[str]:
        """Resolve classifier or prefix-discovered vector columns deterministically."""
        if not self.feature_prefix:
            return [str(x) for x in range(127)]

        if df_or_path is None:
            return []

        if isinstance(df_or_path, pd.DataFrame):
            columns = list(df_or_path.columns)
        else:
            columns = self._schema_columns(Path(df_or_path))

        prefix = re.escape(self.feature_prefix)
        matches = []
        for column in columns:
            match = re.fullmatch(rf"{prefix}(\d+)", str(column))
            if match:
                matches.append((int(match.group(1)), str(column)))

        if not matches:
            raise ValueError(f"No vector columns found with prefix {self.feature_prefix!r}")

        matches.sort(key=lambda item: item[0])
        suffixes = [suffix for suffix, _column in matches]
        expected_suffixes = list(range(suffixes[0], suffixes[0] + len(suffixes)))
        if suffixes != expected_suffixes or suffixes[0] != 0:
            raise ValueError(
                f"Vector columns with prefix {self.feature_prefix!r} must be contiguous from 0"
            )

        return [column for _suffix, column in matches]

    def validate_input_contracts(
        self, cities: List[str], resolution: Optional[int] = None
    ) -> List[str]:
        """Validate all city vector dimensions before pairwise computation starts."""
        if not self.feature_prefix:
            self.vector_columns = [str(x) for x in range(127)]
            return self.vector_columns

        expected_columns: Optional[List[str]] = None
        expected_city: Optional[str] = None
        self._vector_contract_by_city = {}
        missing_inputs: List[str] = []
        empty_resolution_inputs: List[str] = []

        for city in cities:
            file_path = self.build_input_path(city)
            if not file_path.exists():
                missing_inputs.append(f"{city}: {file_path}")
                continue

            columns = self.resolve_vector_columns(file_path)
            self._vector_contract_by_city[city] = columns
            if resolution is not None:
                res_values = pd.read_parquet(file_path, columns=["res"])
                if res_values[res_values["res"] == resolution].empty:
                    empty_resolution_inputs.append(
                        f"{city}: {file_path.name} has no res={resolution} rows"
                    )

            if expected_columns is None:
                expected_columns = columns
                expected_city = city
                continue

            if columns != expected_columns:
                raise ValueError(
                    "Inconsistent vector columns for city "
                    f"{city} file {file_path.name}: found {len(columns)} vector columns "
                    f"but {expected_city} has {len(expected_columns)} vector columns"
                )

        if missing_inputs:
            raise FileNotFoundError(
                "Missing input files for configured vector pipeline: "
                + "; ".join(missing_inputs)
            )

        if empty_resolution_inputs:
            raise ValueError(
                "Input files have no rows for requested resolution: "
                + "; ".join(empty_resolution_inputs)
            )

        if expected_columns is None:
            raise FileNotFoundError(
                f"No input files found under {self.source_root} using {self.input_template}"
            )

        self.vector_columns = expected_columns
        return self.vector_columns

    def similarity_keep_mask(self, similarity_block: np.ndarray) -> np.ndarray:
        """Return mask of similarities to persist for the configured threshold."""
        if self.feature_prefix and self.similarity_threshold <= -1.0:
            return similarity_block >= self.similarity_threshold
        return similarity_block > self.similarity_threshold

    def get_progress_path(self, output_dir: Path, resolution: int) -> Path:
        """Return the checkpoint path for a specific resolution."""
        return output_dir / f"_progress_res={resolution}_optimized.json"

    def get_temp_dir(self, output_dir: Path) -> Path:
        """Return the directory used for intermediate pair shards."""
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def iter_resolution_artifacts(self, output_dir: Path, resolution: int) -> List[Path]:
        """Return resolution-specific progress, temp shard, and final output paths."""
        artifacts: List[Path] = []
        progress_path = self.get_progress_path(output_dir, resolution)
        if progress_path.exists():
            artifacts.append(progress_path)

        artifacts.extend(
            sorted(output_dir.glob(f"similarity_city=*_res={resolution}_optimized.parquet"))
        )
        artifacts.extend(
            sorted(output_dir.glob(f".similarity_city=*_res={resolution}_optimized.parquet.tmp"))
        )

        temp_dir = output_dir / "temp"
        if temp_dir.exists():
            artifacts.extend(
                sorted(temp_dir.glob(f"city1=*/city2=*/part_res={resolution}.parquet"))
            )
            artifacts.extend(
                sorted(temp_dir.glob(f"city1=*/city2=*/.part_res={resolution}.parquet.tmp"))
            )

        return artifacts

    def purge_resolution_outputs(
        self,
        output_dir: Path,
        resolution: int,
        before_date: Optional[date] = None,
    ) -> Dict[str, int]:
        """Delete resolution-specific artifacts, optionally only those older than a cutoff date."""
        files_deleted = 0
        directories_deleted = 0

        for artifact_path in self.iter_resolution_artifacts(output_dir, resolution):
            if before_date is not None:
                modified_date = datetime.fromtimestamp(artifact_path.stat().st_mtime).date()
                if modified_date >= before_date:
                    continue
            artifact_path.unlink(missing_ok=True)
            files_deleted += 1

        temp_dir = output_dir / "temp"
        if temp_dir.exists():
            for city2_dir in sorted(temp_dir.glob("city1=*/city2=*"), reverse=True):
                try:
                    city2_dir.rmdir()
                    directories_deleted += 1
                except OSError:
                    pass
            for city1_dir in sorted(temp_dir.glob("city1=*"), reverse=True):
                try:
                    city1_dir.rmdir()
                    directories_deleted += 1
                except OSError:
                    pass
            try:
                temp_dir.rmdir()
                directories_deleted += 1
            except OSError:
                pass

        self.logger.info(
            "Purged %d resolution=%d artifacts%s and pruned %d directories",
            files_deleted,
            resolution,
            f" older than {before_date.isoformat()}" if before_date else "",
            directories_deleted,
        )
        return {
            "files_deleted": files_deleted,
            "directories_deleted": directories_deleted,
        }

    def discover_completed_output_cities(
        self, output_dir: Path, resolution: int
    ) -> List[str]:
        """Infer completed city1 outputs from existing final parquet files."""
        completed_cities = []
        prefix = "similarity_city="
        resolution_suffix = f"_res={resolution}_optimized.parquet"
        legacy_suffix = "_optimized.parquet"

        for output_file in sorted(output_dir.glob(f"{prefix}*{legacy_suffix}")):
            name = output_file.name
            if name.startswith(prefix) and name.endswith(resolution_suffix):
                completed_cities.append(
                    name[len(prefix) : -len(resolution_suffix)]
                )
            elif name.startswith(prefix) and name.endswith(legacy_suffix):
                completed_cities.append(name[len(prefix) : -len(legacy_suffix)])

        return completed_cities

    def read_progress(self, progress_path: Path) -> Optional[Dict[str, Any]]:
        """Load a saved checkpoint if one exists."""
        if not progress_path.exists():
            return None

        return json.loads(progress_path.read_text())

    def pair_to_key(self, city1: str, city2: str, resolution: int) -> str:
        """Return a stable checkpoint key for a processed city pair."""
        return f"{city1}__{city2}__res={resolution}"

    def create_city_pairs(self, city_meta_path: str) -> List[Tuple[str, str]]:
        """Create all unique city pairs in deterministic order."""
        city_df = pd.read_csv(city_meta_path)
        cities = sorted(city_df["City"].dropna().unique().tolist())

        return [
            (city1, city2)
            for idx, city1 in enumerate(cities)
            for city2 in cities[idx + 1 :]
        ]

    def write_progress(
        self,
        progress_path: Path,
        resolution: int,
        completed_pair_keys: List[str],
        pending_pairs: List[Tuple[str, str]],
        status: str,
        current_pair: Optional[Tuple[str, str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist checkpoint state atomically so interrupted runs can resume."""
        payload: Dict[str, Any] = {
            "resolution": resolution,
            "completed_pair_keys": completed_pair_keys,
            "pending_pairs": [list(pair) for pair in pending_pairs],
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_file": str(self.log_file),
        }

        if current_pair:
            payload["current_pair"] = list(current_pair)
        if error:
            payload["last_error"] = error

        tmp_path = progress_path.with_name(f".{progress_path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(progress_path)

    def resolve_processing_state(
        self,
        city_pairs: List[Tuple[str, str]],
        progress_path: Path,
        resolution: int,
        output_dir: Path,
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Resolve which city pairs still need processing, preferring checkpoint state."""
        ordered_pairs = list(city_pairs)
        checkpoint = self.read_progress(progress_path) if self.resume_enabled else None

        if checkpoint:
            completed_keys = checkpoint.get("completed_pair_keys", [])
            completed_set = set(completed_keys)

            checkpoint_pending = checkpoint.get("pending_pairs", [])
            pending_seen = set()
            pending_pairs: List[Tuple[str, str]] = []

            for pair in checkpoint_pending:
                pair_tuple = tuple(pair)
                pair_key = self.pair_to_key(pair_tuple[0], pair_tuple[1], resolution)
                if pair_tuple in ordered_pairs and pair_key not in completed_set and pair_tuple not in pending_seen:
                    pending_pairs.append(pair_tuple)
                    pending_seen.add(pair_tuple)

            for pair in ordered_pairs:
                pair_key = self.pair_to_key(pair[0], pair[1], resolution)
                if pair_key not in completed_set and pair not in pending_seen:
                    pending_pairs.append(pair)

            if pending_pairs:
                self.logger.info(
                    "Resuming from checkpoint %s with %d pending city pairs",
                    progress_path,
                    len(pending_pairs),
                )
            else:
                self.logger.info("Checkpoint %s reports no pending cities", progress_path)

            return pending_pairs, list(completed_keys)

        completed_cities = (
            set(self.discover_completed_output_cities(output_dir, resolution))
            if self.resume_enabled
            else set()
        )
        completed_pair_keys = [
            self.pair_to_key(city1, city2, resolution)
            for city1, city2 in ordered_pairs
            if city1 in completed_cities
        ]
        pending_pairs = [
            pair for pair in ordered_pairs if pair[0] not in completed_cities
        ]

        if completed_cities:
            self.logger.info(
                "Recovered %d completed city outputs from existing parquet files",
                len(completed_cities),
            )

        return pending_pairs, completed_pair_keys

    def setup_duckdb_optimizations(self) -> None:
        """Configure DuckDB for optimal performance."""
        # Set memory limit
        memory_limit = self.config.get("memory_limit", "8GB")
        self.conn.execute(f"SET memory_limit='{memory_limit}'")

        # Enable parallel execution
        self.conn.execute("SET threads TO 4")

        # Enable progress bar
        self.conn.execute("SET enable_progress_bar=true")

        self.logger.debug("DuckDB optimizations configured")

    def load_features_optimized(
        self, cities: List[str], resolution: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load features for multiple cities in a single optimized query.

        Args:
            cities: List of city names
            resolution: H3 resolution level

        Returns:
            Tuple of (feature DataFrame, city mapping)
        """
        self.logger.info(f"Loading features for {len(cities)} cities")

        self.validate_input_contracts(cities, resolution=resolution)

        # Build optimized query using DuckDB
        file_patterns = []
        for city in cities:
            file_path = self.build_input_path(city)
            if file_path.exists():
                file_patterns.append((city, str(file_path)))

        if not file_patterns:
            return pd.DataFrame(), {}

        # Create UNION query for all cities
        query_parts = []
        for city, file_path in file_patterns:
            cols = ", ".join([f'"{col}"' for col in self.vector_columns])
            query_part = f"""
                SELECT 
                    hex_id,
                    '{city}' as city,
                    {cols}
                FROM read_parquet('{file_path}')
                WHERE res = {resolution}
            """
            query_parts.append(query_part)

        union_query = " UNION ALL ".join(query_parts)
        final_query = f"""
            WITH all_features AS ({union_query})
            SELECT DISTINCT * FROM all_features
            ORDER BY city, hex_id
        """

        df = self.conn.execute(final_query).fetchdf()

        # Create city mapping
        city_map = df.groupby("city")["hex_id"].apply(list).to_dict()

        self.logger.info(f"Loaded {len(df)} total features")
        return df, city_map

    def convert_to_sparse(
        self, features: np.ndarray, threshold: float = 0.01
    ) -> sparse.csr_matrix:
        """
        Convert dense features to sparse matrix format.

        Args:
            features: Dense feature matrix
            threshold: Values below this are set to zero

        Returns:
            Sparse CSR matrix
        """
        # Set small values to zero for sparsity
        features[np.abs(features) < threshold] = 0
        sparse_features = sparse.csr_matrix(features)

        density = sparse_features.nnz / (
            sparse_features.shape[0] * sparse_features.shape[1]
        )
        self.logger.debug(f"Sparse matrix density: {density:.4f}")

        return sparse_features

    def apply_dimensionality_reduction(
        self, features: np.ndarray, n_components: int = 64
    ) -> np.ndarray:
        """
        Apply dimensionality reduction for faster computation.

        Args:
            features: Original feature matrix
            n_components: Number of components to keep

        Returns:
            Reduced feature matrix
        """
        self.logger.info(
            f"Applying dimensionality reduction: {features.shape[1]} -> {n_components}"
        )

        if self.use_sparse and sparse.issparse(features):
            # Use TruncatedSVD for sparse matrices
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            # Use random projection for dense matrices
            reducer = GaussianRandomProjection(
                n_components=n_components, random_state=42
            )

        reduced_features = reducer.fit_transform(features)

        # Calculate variance explained (if using SVD)
        if hasattr(reducer, "explained_variance_ratio_"):
            variance_explained = reducer.explained_variance_ratio_.sum()
            self.logger.info(f"Variance explained: {variance_explained:.4f}")

        return reduced_features

    def compute_similarity_gpu(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity using GPU acceleration.

        Args:
            features1: First feature matrix
            features2: Second feature matrix

        Returns:
            Similarity matrix
        """
        if not self.use_gpu:
            return cosine_similarity(features1, features2)

        self.logger.info("Using GPU for similarity computation")

        # Transfer to GPU
        gpu_features1 = cp.asarray(features1)
        gpu_features2 = cp.asarray(features2)

        # Normalize for cosine similarity
        norm1 = cp.linalg.norm(gpu_features1, axis=1, keepdims=True)
        norm2 = cp.linalg.norm(gpu_features2, axis=1, keepdims=True)

        gpu_features1 = gpu_features1 / (norm1 + 1e-10)
        gpu_features2 = gpu_features2 / (norm2 + 1e-10)

        # Compute dot product (cosine similarity)
        similarity = cp.dot(gpu_features1, gpu_features2.T)

        # Transfer back to CPU
        return cp.asnumpy(similarity)

    def compute_similarity_blocked(
        self, features: np.ndarray, block_size: int = 1000
    ) -> sparse.csr_matrix:
        """
        Compute similarity matrix in blocks to manage memory.

        Args:
            features: Feature matrix
            block_size: Size of blocks for computation

        Returns:
            Sparse similarity matrix
        """
        n = features.shape[0]
        self.logger.info(f"Computing blocked similarity for {n} samples")

        # Initialize sparse matrix builder
        row_indices = []
        col_indices = []
        data_values = []

        # Process in blocks
        for i in tqdm(range(0, n, block_size), desc="Computing similarity blocks"):
            end_i = min(i + block_size, n)
            block_i = features[i:end_i]

            for j in range(i, n, block_size):  # Start from i to get upper triangle
                end_j = min(j + block_size, n)
                block_j = features[j:end_j]

                # Compute similarity for this block
                if self.use_gpu:
                    sim_block = self.compute_similarity_gpu(block_i, block_j)
                else:
                    sim_block = cosine_similarity(block_i, block_j)

                # Keep only values above threshold
                if i == j:
                    # For diagonal blocks, keep upper triangle only
                    sim_block = sim_block.copy()
                    lower_rows, lower_cols = np.tril_indices_from(sim_block, k=0)
                    sim_block[lower_rows, lower_cols] = -np.inf

                # Find non-zero values above threshold
                rows, cols = np.where(self.similarity_keep_mask(sim_block))

                # Adjust indices to global coordinates
                global_rows = rows + i
                global_cols = cols + j

                # Store sparse representation
                row_indices.extend(global_rows)
                col_indices.extend(global_cols)
                data_values.extend(sim_block[rows, cols])

                # Clean up memory
                del sim_block
                gc.collect()

        # Create sparse matrix
        similarity_matrix = sparse.csr_matrix(
            (data_values, (row_indices, col_indices)), shape=(n, n)
        )

        self.logger.info(
            f"Similarity matrix sparsity: {similarity_matrix.nnz / (n * n):.4f}"
        )
        return similarity_matrix

    def process_city_group(self, cities: List[str], resolution: int) -> pd.DataFrame:
        """
        Process a group of cities together for efficiency.

        Args:
            cities: List of city names
            resolution: H3 resolution level

        Returns:
            DataFrame with all similarity pairs
        """
        self.logger.info(f"Processing city group: {cities}")

        # Load all features at once
        df_features, city_map = self.load_features_optimized(cities, resolution)

        if df_features.empty:
            return pd.DataFrame()

        # Extract feature matrix
        feature_matrix = df_features[self.vector_columns].to_numpy(copy=True)
        hex_ids = df_features["hex_id"].values
        city_labels = df_features["city"].values

        # Apply optimizations
        if self.use_approximation and feature_matrix.shape[1] > self.approximation_dims:
            feature_matrix = self.apply_dimensionality_reduction(
                feature_matrix, self.approximation_dims
            )

        if self.use_sparse:
            feature_matrix = self.convert_to_sparse(feature_matrix)

        # Compute similarity
        if feature_matrix.shape[0] > self.batch_size:
            similarity_matrix = self.compute_similarity_blocked(
                feature_matrix, self.batch_size
            )
        else:
            if sparse.issparse(feature_matrix):
                feature_matrix = feature_matrix.toarray()

            if self.use_gpu:
                similarity_matrix = self.compute_similarity_gpu(
                    feature_matrix, feature_matrix
                )
            else:
                similarity_matrix = cosine_similarity(feature_matrix)

            # Convert to sparse and keep upper triangle
            similarity_matrix = np.triu(similarity_matrix, k=1)
            similarity_matrix = sparse.csr_matrix(similarity_matrix)

        # Extract non-zero similarities
        rows, cols = similarity_matrix.nonzero()
        similarities = similarity_matrix.data

        # Create result DataFrame
        results = []
        for idx in range(len(rows)):
            if self.similarity_keep_mask(np.asarray([similarities[idx]])).item():
                results.append(
                    {
                        "hex_id1": hex_ids[rows[idx]],
                        "hex_id2": hex_ids[cols[idx]],
                        "city1": city_labels[rows[idx]],
                        "city2": city_labels[cols[idx]],
                        "similarity": similarities[idx],
                    }
                )

        result_df = pd.DataFrame(results)

        # Clean up memory
        del feature_matrix
        del similarity_matrix
        gc.collect()

        self.logger.info(f"Generated {len(result_df)} similarity pairs")
        return result_df

    def load_city_features(
        self, city_name: str, resolution: int
    ) -> pd.DataFrame:
        """Load feature vectors for a specific city."""
        file_path = self.build_input_path(city_name)

        if not file_path.exists():
            self.logger.warning("File not found: %s", file_path)
            return pd.DataFrame()

        if self.feature_prefix and city_name not in self._vector_contract_by_city:
            columns = self.validate_input_contracts([city_name], resolution=resolution)
        else:
            columns = self._vector_contract_by_city.get(city_name, self.vector_columns)
            if self.feature_prefix and columns != self.vector_columns:
                raise ValueError(
                    f"Inconsistent vector columns for city {city_name} file {file_path.name}"
                )

        read_columns = ["hex_id", "res", *columns]
        df = pd.read_parquet(file_path, columns=read_columns)
        df = df[df["res"] == resolution].drop(columns=["res"]).reset_index(drop=True)
        df["city"] = city_name
        return df

    def compute_city_pair_similarity(
        self, city1: str, city2: str, resolution: int
    ) -> pd.DataFrame:
        """Compute exact similarity for a single global city pair."""
        return self.compute_city_pair_similarity_blocked(
            city1, city2, resolution, self.row_block_size
        )

    def process_city_pair(
        self, city1: str, city2: str, resolution: int
    ) -> pd.DataFrame:
        """Wrapper hook for one exact global city pair."""
        return self.compute_city_pair_similarity(city1, city2, resolution)

    def compute_city_pair_similarity_blocked(
        self, city1: str, city2: str, resolution: int, row_block_size: int
    ) -> pd.DataFrame:
        """Compute one exact city pair in bounded-memory row blocks."""
        df1 = self.load_city_features(city1, resolution)
        df2 = self.load_city_features(city2, resolution)

        if df1.empty or df2.empty:
            self.logger.warning("Missing data for %s or %s", city1, city2)
            return pd.DataFrame()

        if city1 == city2:
            df_combined = df1.reset_index(drop=True)
        else:
            df_combined = (
                pd.concat([df1, df2], ignore_index=True)
                .drop_duplicates("hex_id")
                .reset_index(drop=True)
            )

        features = df_combined[self.vector_columns].to_numpy(copy=True)
        hex_ids = df_combined["hex_id"].to_numpy()
        city_labels = df_combined["city"].to_numpy()

        results = []
        n_rows = len(df_combined)

        for block_start_i in range(0, n_rows, row_block_size):
            block_end_i = min(block_start_i + row_block_size, n_rows)
            block_i = features[block_start_i:block_end_i]

            for block_start_j in range(block_start_i, n_rows, row_block_size):
                block_end_j = min(block_start_j + row_block_size, n_rows)
                block_j = features[block_start_j:block_end_j]
                similarity_block = cosine_similarity(block_i, block_j)

                if block_start_i == block_start_j:
                    similarity_block = similarity_block.copy()
                    lower_rows, lower_cols = np.tril_indices_from(similarity_block, k=0)
                    similarity_block[lower_rows, lower_cols] = -np.inf

                rows, cols = np.where(self.similarity_keep_mask(similarity_block))
                for row_idx, col_idx in zip(rows, cols):
                    global_row = block_start_i + row_idx
                    global_col = block_start_j + col_idx
                    if city1 != city2 and city_labels[global_row] == city_labels[global_col]:
                        continue
                    results.append(
                        {
                            "hex_id1": hex_ids[global_row],
                            "hex_id2": hex_ids[global_col],
                            "city1": city_labels[global_row],
                            "city2": city_labels[global_col],
                            "similarity": similarity_block[row_idx, col_idx],
                        }
                    )

        return pd.DataFrame(results)

    def get_pair_temp_dir(self, output_dir: Path, city1: str, city2: str) -> Path:
        """Return the temp directory for one city pair."""
        pair_dir = self.get_temp_dir(output_dir) / f"city1={city1}" / f"city2={city2}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        return pair_dir

    def save_pair_results(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        city1: str,
        city2: str,
        resolution: int,
    ) -> None:
        """Persist one city pair result shard atomically."""
        if df.empty:
            return

        if self.metric_label and "metric" not in df.columns:
            df = df.copy()
            df["metric"] = self.metric_label

        pair_dir = self.get_pair_temp_dir(output_dir, city1, city2)
        shard_file = pair_dir / f"part_res={resolution}.parquet"
        tmp_file = pair_dir / f".part_res={resolution}.parquet.tmp"
        df.to_parquet(tmp_file, index=False, compression="snappy", engine="pyarrow")
        tmp_file.replace(shard_file)

    def merge_city_results(self, output_dir: Path, resolution: int) -> None:
        """Merge all temp pair outputs into final per-city parquet files."""
        temp_dir = self.get_temp_dir(output_dir)
        for city1_dir in sorted(temp_dir.glob("city1=*")):
            city = city1_dir.name.split("=", 1)[1]
            shard_files = sorted(city1_dir.glob(f"city2=*/part_res={resolution}.parquet"))
            if not shard_files:
                continue

            city_frames = [pd.read_parquet(shard_file) for shard_file in shard_files]
            city_df = pd.concat(city_frames, ignore_index=True)
            city_df = (
                city_df.sort_values("similarity", ascending=False)
                .drop_duplicates(["hex_id1", "hex_id2"], keep="first")
                .reset_index(drop=True)
            )

            output_file = (
                output_dir / f"similarity_city={city}_res={resolution}_optimized.parquet"
            )
            tmp_output_file = (
                output_dir
                / f".similarity_city={city}_res={resolution}_optimized.parquet.tmp"
            )
            city_df.to_parquet(
                tmp_output_file, index=False, compression="snappy", engine="pyarrow"
            )
            tmp_output_file.replace(output_file)

    def cleanup_temp_outputs(self, output_dir: Path) -> None:
        """Remove temp outputs after a successful merge."""
        temp_dir = output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def save_results_partitioned(
        self, df: pd.DataFrame, output_dir: Path, resolution: int
    ) -> None:
        """
        Save results partitioned by city for efficient access.

        Args:
            df: DataFrame with similarity results
            output_dir: Output directory
        """
        if df.empty:
            return

        # Partition by city1
        for city in df["city1"].unique():
            city_df = df[df["city1"] == city].copy()

            output_file = (
                output_dir / f"similarity_city={city}_res={resolution}_optimized.parquet"
            )
            tmp_output_file = (
                output_dir
                / f".similarity_city={city}_res={resolution}_optimized.parquet.tmp"
            )

            # Save with compression
            city_df.to_parquet(
                tmp_output_file, index=False, compression="snappy", engine="pyarrow"
            )
            tmp_output_file.replace(output_file)

            self.logger.debug(f"Saved {len(city_df)} pairs for {city}")

    def run(
        self,
        city_meta_path: str,
        resolution: int = 6,
        group_size: int = 10,
    ) -> None:
        """
        Run optimized similarity processing.

        Args:
            city_meta_path: Path to city metadata
            resolution: H3 resolution level
            group_size: Number of cities to process together
        """
        self.logger.info("Starting optimized similarity processing")

        try:
            # Load city list
            city_pairs = self.create_city_pairs(city_meta_path)
            input_cities = sorted({city for pair in city_pairs for city in pair})
            self.validate_input_contracts(input_cities, resolution=resolution)
            self.logger.info("Processing %d city pairs", len(city_pairs))

            # Create output directory
            output_dir = self.get_output_dir()
            purge_all_for_resolution = self.config.get("purge_all_for_resolution", False)
            purge_before_date = self.config.get("purge_before_date")
            if purge_all_for_resolution or purge_before_date:
                self.purge_resolution_outputs(
                    output_dir=output_dir,
                    resolution=resolution,
                    before_date=purge_before_date,
                )
            progress_path = self.get_progress_path(output_dir, resolution)
            pending_pairs, completed_pair_keys = self.resolve_processing_state(
                city_pairs, progress_path, resolution, output_dir
            )

            self.write_progress(
                progress_path=progress_path,
                resolution=resolution,
                completed_pair_keys=completed_pair_keys,
                pending_pairs=pending_pairs,
                status="in_progress",
            )

            for idx, (city1, city2) in enumerate(
                tqdm(pending_pairs, desc="Processing city pairs")
            ):
                remaining_pairs = pending_pairs[idx + 1 :]
                try:
                    pair_results = self.process_city_pair(city1, city2, resolution)
                    self.save_pair_results(
                        pair_results, output_dir, city1, city2, resolution
                    )

                    completed_pair_keys.append(
                        self.pair_to_key(city1, city2, resolution)
                    )
                    self.write_progress(
                        progress_path=progress_path,
                        resolution=resolution,
                        completed_pair_keys=completed_pair_keys,
                        pending_pairs=remaining_pairs,
                        status="in_progress",
                    )
                    gc.collect()

                except Exception as group_error:
                    failed_pending_pairs = [(city1, city2)] + remaining_pairs
                    self.write_progress(
                        progress_path=progress_path,
                        resolution=resolution,
                        completed_pair_keys=completed_pair_keys,
                        pending_pairs=failed_pending_pairs,
                        status="failed",
                        current_pair=(city1, city2),
                        error=str(group_error),
                    )
                    raise

            self.merge_city_results(output_dir, resolution)
            if not self.keep_temp:
                self.cleanup_temp_outputs(output_dir)
            self.write_progress(
                progress_path=progress_path,
                resolution=resolution,
                completed_pair_keys=completed_pair_keys,
                pending_pairs=[],
                status="completed",
            )
            self.logger.info("Optimized processing completed successfully")

        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            raise

        finally:
            self.close()
            gc.collect()

    def close(self) -> None:
        """Release open resources for one-shot script execution."""
        if hasattr(self, "conn") and self.conn is not None:
            self.conn.close()
            self.conn = None

        if hasattr(self, "logger"):
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Optimized City Similarity Processor")

    parser.add_argument("--resolution", type=int, default=6, help="H3 resolution")
    parser.add_argument(
        "--city-meta", default="/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/city_meta.csv"
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--use-approximation", action="store_true", help="Use approximation algorithms"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000, help="Batch size for processing"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=10,
        help="Deprecated compatibility flag; exact processing now batches inside each city pair instead",
    )
    parser.add_argument(
        "--row-block-size",
        type=int,
        default=1000,
        help="Rows per block when computing exact similarity inside one city pair",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Similarity threshold; use -1.0 for all DINOv3 cosine pairs",
    )
    parser.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore saved progress and recompute all cities from scratch",
    )
    parser.add_argument(
        "--purge-all-for-resolution",
        action="store_true",
        help="Delete all existing temp shards, final outputs, and checkpoint files for this resolution before starting.",
    )
    parser.add_argument(
        "--purge-before-date",
        default=None,
        help="Delete existing temp shards, final outputs, and checkpoint files for this resolution last modified before YYYY-MM-DD before starting.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for local log files; defaults to a logs folder next to this script",
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_CURATE_FOLDER_SOURCE,
        help="Root folder containing per-city H3 vector parquet inputs",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_CURATE_FOLDER_EXPORT,
        help="Root folder where optimized pairwise outputs are written",
    )
    parser.add_argument(
        "--input-template",
        default=DEFAULT_INPUT_TEMPLATE,
        help="Input filename template with {city} and {res_exclude} placeholders",
    )
    parser.add_argument(
        "--feature-prefix",
        default=None,
        help="Discover vector columns by prefix, e.g. e_; omit for classifier columns 0..126",
    )
    parser.add_argument(
        "--metric-label",
        default="cosine",
        help="Metric label added to configurable-vector pairwise shards",
    )
    parser.add_argument(
        "--res-exclude",
        type=int,
        default=11,
        help="Exclusion resolution used in input filenames",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep optimized/temp pair shards after successful merge",
    )

    args = parser.parse_args()

    # Configuration
    purge_before_date = None
    if args.purge_before_date:
        purge_before_date = datetime.strptime(args.purge_before_date, "%Y-%m-%d").date()

    config = {
        "CURATE_FOLDER_SOURCE": args.source_root,
        "CURATE_FOLDER_EXPORT": args.output_root,
        "RES_EXCLUDE": args.res_exclude,
        "INPUT_TEMPLATE": args.input_template,
        "FEATURE_PREFIX": args.feature_prefix,
        "METRIC_LABEL": args.metric_label,
        "keep_temp": args.keep_temp,
        "use_gpu": args.use_gpu,
        "use_sparse": True,
        "use_approximation": args.use_approximation,
        "batch_size": args.batch_size,
        "row_block_size": args.row_block_size,
        "similarity_threshold": args.threshold,
        "memory_limit": args.memory_limit,
        "resume": not args.fresh_start,
        "purge_all_for_resolution": args.purge_all_for_resolution,
        "purge_before_date": purge_before_date,
        "log_dir": args.log_dir,
    }

    # Run processor
    processor = OptimizedSimilarityProcessor(config)
    processor.run(args.city_meta, args.resolution, args.group_size)


if __name__ == "__main__":
    main()
