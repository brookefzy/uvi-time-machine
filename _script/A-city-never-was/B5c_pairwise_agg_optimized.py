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
from datetime import datetime
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

    def get_temp_root(self) -> Path:
        """Return the optimized temp shard root."""
        return Path(self.config["CURATE_FOLDER_EXPORT2"]) / "optimized" / "temp"

    def get_progress_path(self) -> Optional[Path]:
        """Return optional optimized progress file path."""
        progress_path = self.config.get("PROGRESS_PATH")
        return Path(progress_path) if progress_path else None

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
        query = f"""
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
            SELECT *
            FROM deduped
            ORDER BY similarity DESC, hex_id1, hex_id2
        """

        result_df = self.conn.execute(query).fetchdf()
        inner_city = result_df[result_df["city_1"] == result_df["city_2"]]
        inter_city = result_df[result_df["city_1"] != result_df["city_2"]]

        inner_count = len(inner_city)
        inter_count = len(inter_city)

        self.logger.info(
            "City %s: %d inner-city pairs, %d inter-city pairs",
            city,
            inner_count,
            inter_count,
        )

        if inter_count > 0:
            output_file = (
                Path(self.config["EXPORT_FOLDER"])
                / f"similarity_intracity_city={city}_res={self.config['RES_SEL']}.parquet"
            )
            inter_city.to_parquet(output_file, index=False)
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
            self.logger.info("Processing %d cities", len(cities))

            total_inner = 0
            total_inter = 0
            for city in tqdm(cities, desc="Processing cities"):
                inner_count, inter_count = self.process_city_similarity(city)
                total_inner += inner_count
                total_inter += inter_count

            self.logger.info(
                "Processing complete. Total: %d inner-city pairs, %d inter-city pairs",
                total_inner,
                total_inter,
            )
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
    }
    processor = OptimizedUrbanSimilarityProcessor(config, log_level="INFO")
    processor.run(args.city_meta)


if __name__ == "__main__":
    main()
