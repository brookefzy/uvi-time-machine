#!/usr/bin/env python
"""
Optimized City Similarity Processor
Advanced implementation with sparse matrices, approximate algorithms, and GPU support.
"""

import os
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import gc
import json

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


class OptimizedSimilarityProcessor:
    """Optimized similarity computation with multiple algorithmic improvements."""

    def __init__(self, config: Dict[str, any], log_level: str = "INFO"):
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
        self.vector_columns = [str(x) for x in range(127)]

        # Optimization parameters
        self.use_gpu = config.get("use_gpu", False) and GPU_AVAILABLE
        self.use_sparse = config.get("use_sparse", True)
        self.use_approximation = config.get("use_approximation", False)
        self.approximation_dims = config.get("approximation_dims", 64)
        self.batch_size = config.get("batch_size", 5000)
        self.similarity_threshold = config.get("similarity_threshold", 0.1)

        self.logger.info(
            f"Optimization settings: GPU={self.use_gpu}, Sparse={self.use_sparse}, Approx={self.use_approximation}"
        )

    def setup_logging(self, log_level: str) -> None:
        """Configure optimized logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"optimized_similarity_{timestamp}.log"

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

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

        # Build optimized query using DuckDB
        file_patterns = []
        for city in cities:
            pattern = (
                f"prob_city={city}_res_exclude={self.config['RES_EXCLUDE']}.parquet"
            )
            file_path = Path(self.config["CURATE_FOLDER_SOURCE"]) / pattern
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
                    sim_block = np.triu(sim_block, k=1)

                # Find non-zero values above threshold
                rows, cols = np.where(sim_block > self.similarity_threshold)

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
        feature_matrix = df_features[self.vector_columns].values
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
            if similarities[idx] > self.similarity_threshold:
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

    def save_results_partitioned(self, df: pd.DataFrame, output_dir: Path) -> None:
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

            output_file = output_dir / f"similarity_city={city}_optimized.parquet"

            # Save with compression
            city_df.to_parquet(
                output_file, index=False, compression="snappy", engine="pyarrow"
            )

            self.logger.debug(f"Saved {len(city_df)} pairs for {city}")

    def run(
        self, city_meta_path: str, resolution: int = 6, group_size: int = 10
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
            city_df = pd.read_csv(city_meta_path)
            cities = city_df["City"].unique().tolist()
            self.logger.info(f"Processing {len(cities)} cities")

            # Create output directory
            output_dir = Path(self.config["CURATE_FOLDER_EXPORT"]) / "optimized"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process cities in groups for efficiency
            all_results = []

            for i in tqdm(
                range(0, len(cities), group_size), desc="Processing city groups"
            ):
                city_group = cities[i : i + group_size]

                group_results = self.process_city_group(city_group, resolution)

                if not group_results.empty:
                    all_results.append(group_results)

                # Save intermediate results
                if len(all_results) >= 5:  # Save every 5 groups
                    combined = pd.concat(all_results, ignore_index=True)
                    self.save_results_partitioned(combined, output_dir)
                    all_results = []
                    gc.collect()

            # Save final results
            if all_results:
                combined = pd.concat(all_results, ignore_index=True)
                self.save_results_partitioned(combined, output_dir)

            self.logger.info("Optimized processing completed successfully")

        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            raise

        finally:
            self.conn.close()
            gc.collect()


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Optimized City Similarity Processor")

    parser.add_argument("--resolution", type=int, default=6, help="H3 resolution")
    parser.add_argument(
        "--city-meta", default="/home/yuanzf/uvi-time-machine/_script/city_meta.csv"
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--use-approximation", action="store_true", help="Use approximation algorithms"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000, help="Batch size for processing"
    )
    parser.add_argument("--group-size", type=int, default=10, help="Cities per group")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Similarity threshold"
    )
    parser.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit")

    args = parser.parse_args()

    # Configuration
    config = {
        "CURATE_FOLDER_SOURCE": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary",
        "CURATE_FOLDER_EXPORT": "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
        "RES_EXCLUDE": 11,
        "use_gpu": args.use_gpu,
        "use_sparse": True,
        "use_approximation": args.use_approximation,
        "batch_size": args.batch_size,
        "similarity_threshold": args.threshold,
        "memory_limit": args.memory_limit,
    }

    # Run processor
    processor = OptimizedSimilarityProcessor(config)
    processor.run(args.city_meta, args.resolution, args.group_size)


if __name__ == "__main__":
    main()
