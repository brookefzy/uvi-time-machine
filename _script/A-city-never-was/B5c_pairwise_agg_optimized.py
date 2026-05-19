#!/usr/bin/env python
"""
Urban Visual Similarity Processor for optimized temp shards.
Processes pairwise similarity shard files from
B5b_compute_similarity_pairwise-optimized.py and exports the same
downstream-friendly inter-city aggregation outputs as B5c_pairwise_agg.py.
"""

import argparse
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm


class OptimizedUrbanSimilarityProcessor:
    """Aggregate optimized pairwise temp shards with DuckDB."""

    def __init__(self, config: Dict[str, Any], log_level: str = "INFO"):
        self.config = config
        self.setup_logging(log_level)
        self.conn = duckdb.connect(":memory:")
        self.configure_duckdb()
        self.setup_directories()

    def setup_logging(self, log_level: str) -> None:
        """Configure local logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"urban_similarity_optimized_{timestamp}.log"

        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        self.logger.info("Logging initialized. Log file: %s", log_file)

    def setup_directories(self) -> None:
        """Create necessary directories."""
        Path(self.config["EXPORT_FOLDER"]).mkdir(parents=True, exist_ok=True)

    def configure_duckdb(self) -> None:
        """Configure DuckDB spill-to-disk settings for large cities."""
        memory_limit = self.config.get("DUCKDB_MEMORY_LIMIT")
        temp_directory = self.config.get("DUCKDB_TEMP_DIR")
        threads = self.config.get("DUCKDB_THREADS")

        if memory_limit:
            self.conn.execute(f"SET memory_limit='{memory_limit}'")
        if temp_directory:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
            self.conn.execute(f"SET temp_directory='{temp_directory}'")
        if threads:
            self.conn.execute(f"SET threads TO {int(threads)}")

    def get_temp_root(self) -> Path:
        """Return the optimized temp shard root."""
        return Path(self.config["CURATE_FOLDER_EXPORT2"]) / "optimized" / "temp"

    def get_progress_path(self) -> Optional[Path]:
        """Return optional upstream pairwise progress file path."""
        progress_path = self.config.get("PROGRESS_PATH")
        return Path(progress_path) if progress_path else None

    def get_agg_progress_path(self) -> Optional[Path]:
        """Return optional aggregation progress file path."""
        progress_path = self.config.get("AGG_PROGRESS_PATH")
        return Path(progress_path) if progress_path else None

    def get_output_file(self, city: str) -> Path:
        """Return the final aggregated output file for one city."""
        return (
            Path(self.config["EXPORT_FOLDER"])
            / f"similarity_intracity_city={city}_res={self.config['RES_SEL']}.parquet"
        )

    def read_progress(self) -> Optional[Dict[str, Any]]:
        """Load a saved aggregation progress file if present."""
        progress_path = self.get_agg_progress_path()
        if not progress_path or not progress_path.exists():
            return None
        return json.loads(progress_path.read_text())

    def write_progress(
        self,
        completed_cities: List[str],
        pending_cities: List[str],
        status: str,
    ) -> None:
        """Persist city-level aggregation progress."""
        progress_path = self.get_agg_progress_path()
        if not progress_path:
            return

        payload = {
            "resolution": self.config["RES_SEL"],
            "completed_cities": completed_cities,
            "pending_cities": pending_cities,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_file": next(
                (
                    handler.baseFilename
                    for handler in self.logger.handlers
                    if isinstance(handler, logging.FileHandler)
                ),
                None,
            ),
        }
        progress_path.write_text(json.dumps(payload, indent=2))

    def resolve_cities_to_process(self, cities: List[str]) -> Tuple[List[str], List[str]]:
        """Resolve pending cities from progress and existing output files."""
        ordered_cities = list(dict.fromkeys(cities))
        if not self.config.get("RESUME", True):
            return ordered_cities, []

        progress = self.read_progress()
        if progress:
            completed_set = set(progress.get("completed_cities", []))
            completed = [city for city in ordered_cities if city in completed_set]
            pending = [city for city in ordered_cities if city not in completed_set]
            return pending, completed

        completed = [
            city for city in ordered_cities if self.get_output_file(city).exists()
        ]
        pending = [city for city in ordered_cities if city not in set(completed)]
        return pending, completed

    def warn_if_pairwise_not_finished(self) -> None:
        """Warn if the upstream optimized pairwise run looks incomplete."""
        progress_path = self.get_progress_path()
        if not progress_path or not progress_path.exists():
            return

        progress = json.loads(progress_path.read_text())
        pending_pairs = progress.get("pending_pairs", [])
        status = progress.get("status", "unknown")

        if pending_pairs:
            self.logger.warning(
                "Progress file %s still has %d pending pairs; aggregation may be incomplete",
                progress_path,
                len(pending_pairs),
            )
        elif status != "completed":
            self.logger.warning(
                "Progress file %s reports status=%s but no pending pairs; proceeding with aggregation",
                progress_path,
                status,
            )

    def get_city_shard_files(self, city: str) -> List[Tuple[Path, str]]:
        """Return shard files for one city1, along with the city2 directory name."""
        city_dir = self.get_temp_root() / f"city1={city}"
        if not city_dir.exists():
            return []

        shard_files: List[Tuple[Path, str]] = []
        pattern = f"part_res={self.config['RES_SEL']}.parquet"
        for city2_dir in sorted(city_dir.glob("city2=*")):
            shard_file = city2_dir / pattern
            if shard_file.exists():
                city2 = city2_dir.name.split("=", 1)[1]
                shard_files.append((shard_file, city2))

        return shard_files

    def process_city_similarity(self, city: str) -> Tuple[int, int]:
        """Aggregate similarity data for one city1 from optimized temp shards."""
        self.logger.info("Processing similarity data for city: %s", city)
        shard_files = self.get_city_shard_files(city)

        if not shard_files:
            self.logger.warning("No temp shard files found for city: %s", city)
            return 0, 0

        query_parts = []
        for shard_file, city2 in shard_files:
            query_parts.append(
                f"""
                SELECT
                    hex_id1,
                    hex_id2,
                    similarity,
                    '{city}' AS shard_city1,
                    '{city2}' AS shard_city2
                FROM read_parquet('{shard_file}')
                """
            )

        union_query = " UNION ALL ".join(query_parts)
        base_query = f"""
            WITH shard_union AS ({union_query}),
            deduped AS (
                SELECT
                    LEAST(hex_id1, hex_id2) AS hex_id1,
                    GREATEST(hex_id1, hex_id2) AS hex_id2,
                    MAX(similarity) AS similarity,
                    shard_city1 AS city_1,
                    shard_city2 AS city_2
                FROM shard_union
                GROUP BY
                    LEAST(hex_id1, hex_id2),
                    GREATEST(hex_id1, hex_id2),
                    shard_city1,
                    shard_city2
            )
        """
        inner_count_query = f"""
            {base_query}
            SELECT COUNT(*) AS count
            FROM deduped
            WHERE city_1 = city_2
        """
        inter_count_query = f"""
            {base_query}
            SELECT COUNT(*) AS count
            FROM deduped
            WHERE city_1 != city_2
        """
        export_query = f"""
            {base_query}
            SELECT hex_id1, hex_id2, similarity, city_1, city_2
            FROM deduped
            WHERE city_1 != city_2
            ORDER BY similarity DESC, hex_id1, hex_id2
        """

        inner_count = int(self.conn.execute(inner_count_query).fetchone()[0])
        inter_count = int(self.conn.execute(inter_count_query).fetchone()[0])

        self.logger.info(
            "City %s: %d inner-city pairs, %d inter-city pairs",
            city,
            inner_count,
            inter_count,
        )

        if inter_count > 0:
            output_file = self.get_output_file(city)
            self.conn.execute(f"COPY ({export_query}) TO '{output_file}'")
            self.logger.debug("Saved inter-city results to: %s", output_file)

        gc.collect()
        return inner_count, inter_count

    def run(self, city_meta_path: str) -> None:
        """Run aggregation across all cities in metadata."""
        self.logger.info("Starting optimized urban similarity aggregation")

        try:
            self.warn_if_pairwise_not_finished()

            city_meta = pd.read_csv(city_meta_path)
            cities = city_meta["City"].dropna().tolist()
            pending_cities, completed_cities = self.resolve_cities_to_process(cities)
            self.logger.info(
                "Processing %d cities (%d already completed)",
                len(pending_cities),
                len(completed_cities),
            )
            self.write_progress(completed_cities, pending_cities, "in_progress")

            total_inner = 0
            total_inter = 0
            for idx, city in enumerate(tqdm(pending_cities, desc="Processing cities")):
                try:
                    inner_count, inter_count = self.process_city_similarity(city)
                    total_inner += inner_count
                    total_inter += inter_count
                    completed_cities.append(city)
                    self.write_progress(
                        completed_cities,
                        pending_cities[idx + 1 :],
                        "in_progress",
                    )
                except Exception:
                    self.write_progress(
                        completed_cities,
                        pending_cities[idx:],
                        "failed",
                    )
                    raise

            self.logger.info(
                "Processing complete. Total: %d inner-city pairs, %d inter-city pairs",
                total_inner,
                total_inter,
            )
            self.write_progress(completed_cities, [], "completed")
        finally:
            self.close()
            gc.collect()

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "conn") and self.conn is not None:
            self.conn.close()
            self.conn = None

        if hasattr(self, "logger"):
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate optimized pairwise similarity temp shards"
    )
    parser.add_argument(
        "--city-meta",
        default="../city_meta.csv",
        help="Path to city metadata CSV",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=8,
        help="H3 resolution level to aggregate",
    )
    parser.add_argument(
        "--pairwise-root",
        default="/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
        help="Root folder containing the optimized temp output tree",
    )
    parser.add_argument(
        "--export-folder",
        default=None,
        help="Output folder for aggregated parquet files",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional optimized pairwise progress JSON; warns if pending pairs remain",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cities that already have aggregated outputs or are marked completed in the aggregation progress file",
    )
    parser.add_argument(
        "--agg-progress-file",
        default=None,
        help="Optional city-level aggregation progress JSON for resume support",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default=None,
        help="Optional DuckDB memory limit, for example 8GB",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=None,
        help="Optional DuckDB temp spill directory for large city aggregations",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=None,
        help="Optional DuckDB thread count",
    )
    args = parser.parse_args()

    today = datetime.now().strftime("%Y%m%d")
    export_folder = args.export_folder or (
        f"/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_{today}"
    )

    config = {
        "CURATE_FOLDER_EXPORT2": args.pairwise_root,
        "EXPORT_FOLDER": export_folder,
        "RES_SEL": args.resolution,
        "PROGRESS_PATH": args.progress_file,
        "RESUME": args.resume,
        "AGG_PROGRESS_PATH": args.agg_progress_file,
        "DUCKDB_MEMORY_LIMIT": args.duckdb_memory_limit,
        "DUCKDB_TEMP_DIR": args.duckdb_temp_dir,
        "DUCKDB_THREADS": args.duckdb_threads,
    }
    processor = OptimizedUrbanSimilarityProcessor(config, log_level="INFO")
    processor.run(args.city_meta)


if __name__ == "__main__":
    main()
