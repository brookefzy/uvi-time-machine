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
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm

from dinov3_utils import normalize_city_name


AUDIT_STATUSES = ("direct", "reversed_fixed", "unresolved", "missing_city_h3")
VALIDATION_CONTRACT = "h3-city-membership-v1"


def sql_quote(value: object) -> str:
    """Escape a value embedded in a DuckDB string literal."""
    return str(value).replace("'", "''")


class OptimizedUrbanSimilarityProcessor:
    """Aggregate optimized pairwise temp shards with DuckDB."""

    def __init__(self, config: Dict[str, Any], log_level: str = "INFO"):
        self.config = config
        self.setup_logging(log_level)
        self.conn = duckdb.connect(":memory:")
        self.configure_duckdb()
        self.setup_directories()
        self.audit = {
            "validation_contract": VALIDATION_CONTRACT,
            "resolution": int(self.config["RES_SEL"]),
            "totals": {
                **{status: 0 for status in AUDIT_STATUSES},
                "input_rows": 0,
                "emitted_rows": 0,
            },
            "by_city": {},
            "examples": {status: [] for status in AUDIT_STATUSES},
        }
        self.validated_cities: set[str] = set()

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
        self.get_unresolved_folder().mkdir(parents=True, exist_ok=True)

    def get_unresolved_folder(self) -> Path:
        configured = self.config.get("UNRESOLVED_FOLDER")
        return (
            Path(configured)
            if configured
            else Path(self.config["EXPORT_FOLDER"]) / "_unresolved"
        )

    def get_audit_report_path(self) -> Path:
        configured = self.config.get("AUDIT_REPORT_PATH")
        return (
            Path(configured)
            if configured
            else Path(self.config["EXPORT_FOLDER"])
            / f"_audit_res={self.config['RES_SEL']}.json"
        )

    def get_unresolved_file(self, city: str) -> Path:
        return self.get_unresolved_folder() / (
            f"unresolved_city={city}_res={self.config['RES_SEL']}.parquet"
        )

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

    def get_membership_path(self, city: str) -> Path:
        """Return the authoritative H3 summary path for one metadata city."""
        root = self.config.get("H3_MEMBERSHIP_ROOT")
        template = self.config.get(
            "H3_INPUT_TEMPLATE", "dinov3_city={city}_res_exclude=None.parquet"
        )
        if not root:
            raise ValueError("H3_MEMBERSHIP_ROOT is required for B5c validation")
        root_path = Path(root)
        exact_path = root_path / template.format(city=city)
        if exact_path.exists() or "{city}" not in template:
            return exact_path

        prefix, suffix = template.split("{city}", 1)
        normalized_city = normalize_city_name(city)
        matches = []
        for candidate in root_path.glob(f"{prefix}*{suffix}"):
            candidate_name = candidate.name
            candidate_city = candidate_name[
                len(prefix) : len(candidate_name) - len(suffix) if suffix else None
            ]
            if normalize_city_name(candidate_city) == normalized_city:
                matches.append(candidate)
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous normalized H3 membership files for {city!r}: {matches}"
            )
        return matches[0] if matches else exact_path

    def load_h3_membership(self, cities: List[str]) -> pd.DataFrame:
        """Load and register authoritative resolution-specific H3 membership."""
        frames = []
        missing_files = []
        resolution = int(self.config["RES_SEL"])
        for city in cities:
            membership_path = self.get_membership_path(city)
            if not membership_path.exists():
                missing_files.append(str(membership_path))
                continue
            frame = pd.read_parquet(membership_path, columns=["hex_id", "res"])
            frame = frame.loc[frame["res"] == resolution, ["hex_id"]].copy()
            frame["membership_city"] = city
            frame["city_norm"] = normalize_city_name(city)
            frames.append(frame)

        self.audit["missing_membership_files"] = missing_files
        if missing_files:
            self.logger.warning(
                "Missing %d authoritative H3 membership files; affected rows will be quarantined",
                len(missing_files),
            )
        if not frames:
            raise ValueError(f"No H3 membership rows found for resolution {resolution}")

        membership = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=["hex_id"])
            .drop_duplicates(["hex_id", "city_norm"])
            .reset_index(drop=True)
        )
        if (membership["city_norm"] == "").any():
            raise ValueError("H3 membership contains an empty normalized city name")

        known_cities = membership[["city_norm"]].drop_duplicates().reset_index(drop=True)
        known_hex = membership[["hex_id"]].drop_duplicates().reset_index(drop=True)
        for relation_name in (
            "h3_membership",
            "known_membership_cities",
            "known_membership_hex",
        ):
            try:
                self.conn.unregister(relation_name)
            except Exception:
                pass
        self.conn.register("h3_membership", membership)
        self.conn.register("known_membership_cities", known_cities)
        self.conn.register("known_membership_hex", known_hex)
        self.logger.info(
            "Loaded %d H3-to-city membership rows for resolution %d",
            len(membership),
            resolution,
        )
        return membership

    def _parquet_columns(self, path: Path) -> set[str]:
        """Return a parquet file's columns without materializing its rows."""
        try:
            import pyarrow.parquet as pq

            return set(pq.read_schema(path).names)
        except Exception:
            return set(pd.read_parquet(path).columns)

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
        """Return the final aggregated output path for one city."""
        return (
            Path(self.config["EXPORT_FOLDER"])
            / f"similarity_intracity_city={city}_res={self.config['RES_SEL']}.parquet"
        )

    def output_exists(self, city: str) -> bool:
        """Return true when a city already has a usable aggregated export."""
        output_path = self.get_output_file(city)
        if output_path.is_file():
            return True
        if output_path.is_dir():
            return any(output_path.glob("*.parquet"))
        return False

    def remove_output_path(self, output_path: Path) -> None:
        """Remove an output file or dataset directory."""
        if output_path.is_dir():
            shutil.rmtree(output_path)
        elif output_path.exists():
            output_path.unlink()

    def export_intercity_dataset(self, city: str, export_query: str) -> Path:
        """Write one city's inter-city result as a parquet file or partitioned dataset."""
        output_path = self.get_output_file(city)
        temp_output_path = output_path.with_name(f"{output_path.name}.tmp")
        parquet_file_size = str(
            self.config.get("PARQUET_FILE_SIZE_BYTES", "512MB")
        ).strip()

        self.remove_output_path(temp_output_path)
        if output_path.exists():
            self.remove_output_path(output_path)

        try:
            if parquet_file_size and parquet_file_size.lower() not in {"0", "none", "false"}:
                copy_options = (
                    "FORMAT parquet, "
                    "PER_THREAD_OUTPUT true, "
                    f"FILE_SIZE_BYTES '{parquet_file_size}', "
                    "FILENAME_PATTERN 'part_{i}'"
                )
            else:
                copy_options = "FORMAT parquet"

            self.conn.execute(
                f"COPY ({export_query}) TO '{sql_quote(temp_output_path)}' ({copy_options})"
            )
            temp_output_path.rename(output_path)
            return output_path
        except Exception:
            self.remove_output_path(temp_output_path)
            raise

    def read_progress(self) -> Optional[Dict[str, Any]]:
        """Load a saved aggregation progress file if present."""
        progress_path = self.get_agg_progress_path()
        if not progress_path or not progress_path.exists():
            return None
        return json.loads(progress_path.read_text())

    def load_existing_audit(self) -> None:
        """Restore only audit state produced by the current validation contract."""
        audit_path = self.get_audit_report_path()
        self.validated_cities = set()
        if not audit_path.exists():
            return
        existing = json.loads(audit_path.read_text())
        if (
            existing.get("validation_contract") != VALIDATION_CONTRACT
            or int(existing.get("resolution", -1)) != int(self.config["RES_SEL"])
        ):
            self.logger.warning(
                "Ignoring legacy or incompatible B5c audit at %s", audit_path
            )
            return

        existing_by_city = existing.get("by_city", {})
        self.validated_cities = set(existing_by_city)
        self.audit["by_city"].update(existing_by_city)
        for key in self.audit["totals"]:
            self.audit["totals"][key] = int(existing.get("totals", {}).get(key, 0))
        for status in AUDIT_STATUSES:
            self.audit["examples"][status] = list(
                existing.get("examples", {}).get(status, [])
            )

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
            "validation_contract": VALIDATION_CONTRACT,
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

        self.load_existing_audit()
        progress = self.read_progress()
        if progress and progress.get("validation_contract") == VALIDATION_CONTRACT:
            completed_set = set(progress.get("completed_cities", []))
            completed = [
                city
                for city in ordered_cities
                if city in completed_set
                and city in self.validated_cities
                and self.output_exists(city)
            ]
            completed_actual = set(completed)
            pending = [city for city in ordered_cities if city not in completed_actual]
            return pending, completed
        if progress:
            self.logger.warning(
                "Ignoring legacy aggregation progress without validation contract"
            )

        completed = [
            city
            for city in ordered_cities
            if city in self.validated_cities and self.output_exists(city)
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
        temp_root = self.get_temp_root()
        city_dir = temp_root / f"city1={city}"
        if not city_dir.exists():
            normalized_city = normalize_city_name(city)
            matches = [
                path
                for path in temp_root.glob("city1=*")
                if path.is_dir()
                and normalize_city_name(path.name.split("=", 1)[1]) == normalized_city
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous normalized pairwise city directories for {city!r}: {matches}"
                )
            if matches:
                city_dir = matches[0]
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

    def get_merged_city_file(self, city: str) -> Path:
        """Return the merged optimized pairwise file for one city1."""
        output_dir = Path(self.config["CURATE_FOLDER_EXPORT2"]) / "optimized"
        exact_path = output_dir / (
            f"similarity_city={city}_res={self.config['RES_SEL']}_optimized.parquet"
        )
        if exact_path.exists():
            return exact_path
        suffix = f"_res={self.config['RES_SEL']}_optimized.parquet"
        normalized_city = normalize_city_name(city)
        matches = []
        for candidate in output_dir.glob(f"similarity_city=*{suffix}"):
            candidate_city = candidate.name[len("similarity_city=") : -len(suffix)]
            if normalize_city_name(candidate_city) == normalized_city:
                matches.append(candidate)
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous normalized merged city files for {city!r}: {matches}"
            )
        return matches[0] if matches else exact_path

    def get_city_source_queries(self, city: str) -> List[str]:
        """Return DuckDB source queries for one city from temp shards or merged output."""
        query_parts = []
        for shard_file, city2 in self.get_city_shard_files(city):
            columns = self._parquet_columns(shard_file)
            row_city1 = "city1" if "city1" in columns else f"'{sql_quote(city)}'"
            row_city2 = "city2" if "city2" in columns else f"'{sql_quote(city2)}'"
            query_parts.append(
                f"""
                SELECT
                    hex_id1,
                    hex_id2,
                    similarity,
                    {row_city1} AS row_city1,
                    {row_city2} AS row_city2,
                    '{sql_quote(city)}' AS shard_city1,
                    '{sql_quote(city2)}' AS shard_city2
                FROM read_parquet('{sql_quote(shard_file)}', hive_partitioning=false)
                """
            )

        if query_parts:
            return query_parts

        merged_city_file = self.get_merged_city_file(city)
        if merged_city_file.exists():
            columns = self._parquet_columns(merged_city_file)
            row_city1 = "city1" if "city1" in columns else f"'{sql_quote(city)}'"
            row_city2 = "city2" if "city2" in columns else "city2"
            if "city2" not in columns:
                raise ValueError(
                    f"Merged file lacks city2 endpoint labels: {merged_city_file}"
                )
            query_parts.append(
                f"""
                SELECT
                    hex_id1,
                    hex_id2,
                    similarity,
                    {row_city1} AS row_city1,
                    {row_city2} AS row_city2,
                    {row_city1} AS shard_city1,
                    {row_city2} AS shard_city2
                FROM read_parquet('{sql_quote(merged_city_file)}', hive_partitioning=false)
                """
            )

        return query_parts

    def _register_city_aliases(self) -> None:
        """Normalize the distinct row labels once, then join by a tiny alias table."""
        raw_aliases = self.conn.execute(
            """
            SELECT DISTINCT raw_city
            FROM (
                SELECT row_city1 AS raw_city FROM raw_city_similarity
                UNION
                SELECT row_city2 AS raw_city FROM raw_city_similarity
            ) aliases
            """
        ).fetchdf()
        raw_aliases["city_norm"] = raw_aliases["raw_city"].map(normalize_city_name)
        try:
            self.conn.unregister("city_aliases")
        except Exception:
            pass
        self.conn.register("city_aliases", raw_aliases)

    def _create_classified_table(self, union_query: str) -> None:
        """Materialize one city's rows and classify their endpoint membership."""
        self.conn.execute(
            f"CREATE OR REPLACE TEMP TABLE raw_city_similarity AS {union_query}"
        )
        self._register_city_aliases()
        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE classified_city_similarity AS
            WITH normalized AS (
                SELECT
                    raw.*,
                    a1.city_norm AS city1_norm,
                    a2.city_norm AS city2_norm,
                    EXISTS (
                        SELECT 1 FROM known_membership_hex h
                        WHERE h.hex_id = raw.hex_id1
                    ) AS has_hex1,
                    EXISTS (
                        SELECT 1 FROM known_membership_hex h
                        WHERE h.hex_id = raw.hex_id2
                    ) AS has_hex2,
                    EXISTS (
                        SELECT 1 FROM known_membership_cities c
                        WHERE c.city_norm = a1.city_norm
                    ) AS has_city1,
                    EXISTS (
                        SELECT 1 FROM known_membership_cities c
                        WHERE c.city_norm = a2.city_norm
                    ) AS has_city2,
                    EXISTS (
                        SELECT 1 FROM h3_membership m
                        WHERE m.hex_id = raw.hex_id1 AND m.city_norm = a1.city_norm
                    ) AND EXISTS (
                        SELECT 1 FROM h3_membership m
                        WHERE m.hex_id = raw.hex_id2 AND m.city_norm = a2.city_norm
                    ) AS is_direct,
                    EXISTS (
                        SELECT 1 FROM h3_membership m
                        WHERE m.hex_id = raw.hex_id1 AND m.city_norm = a2.city_norm
                    ) AND EXISTS (
                        SELECT 1 FROM h3_membership m
                        WHERE m.hex_id = raw.hex_id2 AND m.city_norm = a1.city_norm
                    ) AS is_reverse
                FROM raw_city_similarity raw
                LEFT JOIN city_aliases a1 ON raw.row_city1 = a1.raw_city
                LEFT JOIN city_aliases a2 ON raw.row_city2 = a2.raw_city
            )
            SELECT
                *,
                CASE
                    WHEN is_direct THEN 'direct'
                    WHEN is_reverse THEN 'reversed_fixed'
                    WHEN NOT has_hex1 OR NOT has_hex2
                      OR NOT has_city1 OR NOT has_city2
                      OR city1_norm = '' OR city2_norm = ''
                        THEN 'missing_city_h3'
                    ELSE 'unresolved'
                END AS validation_status
            FROM normalized
            """
        )

    def _resolved_base_query(self) -> str:
        """Return validated, fully oriented, and safely deduplicated rows."""
        return """
            WITH oriented AS (
                SELECT
                    CASE WHEN validation_status = 'reversed_fixed' THEN hex_id2 ELSE hex_id1 END AS oriented_hex1,
                    CASE WHEN validation_status = 'reversed_fixed' THEN hex_id1 ELSE hex_id2 END AS oriented_hex2,
                    row_city1 AS oriented_city1,
                    row_city2 AS oriented_city2,
                    city1_norm AS oriented_city1_norm,
                    city2_norm AS oriented_city2_norm,
                    similarity
                FROM classified_city_similarity
                WHERE validation_status IN ('direct', 'reversed_fixed')
            ),
            canonicalized AS (
                SELECT
                    CASE WHEN oriented_hex1 <= oriented_hex2 THEN oriented_hex1 ELSE oriented_hex2 END AS hex_id1,
                    CASE WHEN oriented_hex1 <= oriented_hex2 THEN oriented_hex2 ELSE oriented_hex1 END AS hex_id2,
                    CASE WHEN oriented_hex1 <= oriented_hex2 THEN oriented_city1 ELSE oriented_city2 END AS city_1,
                    CASE WHEN oriented_hex1 <= oriented_hex2 THEN oriented_city2 ELSE oriented_city1 END AS city_2,
                    CASE WHEN oriented_hex1 <= oriented_hex2 THEN oriented_city1_norm ELSE oriented_city2_norm END AS city_1_norm,
                    CASE WHEN oriented_hex1 <= oriented_hex2 THEN oriented_city2_norm ELSE oriented_city1_norm END AS city_2_norm,
                    similarity
                FROM oriented
            ),
            deduped AS (
                SELECT
                    hex_id1,
                    hex_id2,
                    city_1,
                    city_2,
                    city_1_norm,
                    city_2_norm,
                    MAX(similarity) AS similarity
                FROM canonicalized
                GROUP BY ALL
            )
        """

    def _write_unresolved_rows(self, city: str, unresolved_count: int) -> None:
        output_path = self.get_unresolved_file(city)
        self.remove_output_path(output_path)
        if unresolved_count == 0:
            return
        temp_path = output_path.with_name(f".{output_path.name}.tmp")
        self.remove_output_path(temp_path)
        query = """
            SELECT
                hex_id1,
                hex_id2,
                similarity,
                row_city1 AS city_1,
                row_city2 AS city_2,
                shard_city1,
                shard_city2,
                validation_status,
                CASE
                    WHEN NOT has_hex1 OR NOT has_hex2 THEN 'missing_h3'
                    WHEN NOT has_city1 OR NOT has_city2 OR city1_norm = '' OR city2_norm = '' THEN 'missing_city'
                    ELSE 'membership_mismatch'
                END AS unresolved_reason
            FROM classified_city_similarity
            WHERE validation_status IN ('unresolved', 'missing_city_h3')
        """
        try:
            self.conn.execute(
                f"COPY ({query}) TO '{sql_quote(temp_path)}' (FORMAT parquet)"
            )
            temp_path.replace(output_path)
        except Exception:
            self.remove_output_path(temp_path)
            raise

    def _record_city_audit(self, city: str, emitted_rows: int) -> None:
        grouped = {
            row[0]: int(row[1])
            for row in self.conn.execute(
                """
                SELECT validation_status, COUNT(*)
                FROM classified_city_similarity
                GROUP BY validation_status
                """
            ).fetchall()
        }
        city_counts = {status: grouped.get(status, 0) for status in AUDIT_STATUSES}
        city_counts["input_rows"] = sum(city_counts.values())
        city_counts["emitted_rows"] = int(emitted_rows)
        previous_counts = self.audit["by_city"].get(city, {})
        for key in self.audit["totals"]:
            self.audit["totals"][key] -= int(previous_counts.get(key, 0))
        self.audit["by_city"][city] = city_counts
        for key, count in city_counts.items():
            self.audit["totals"][key] += count

        example_limit = int(self.config.get("AUDIT_EXAMPLE_LIMIT", 5))
        for status in AUDIT_STATUSES:
            remaining = example_limit - len(self.audit["examples"][status])
            if remaining <= 0 or city_counts[status] == 0:
                continue
            examples = self.conn.execute(
                """
                SELECT hex_id1, hex_id2, row_city1 AS city_1, row_city2 AS city_2,
                       shard_city1, shard_city2, similarity
                FROM classified_city_similarity
                WHERE validation_status = ?
                LIMIT ?
                """,
                [status, remaining],
            ).fetchdf()
            self.audit["examples"][status].extend(examples.to_dict("records"))
        self.validated_cities.add(city)

    def _record_empty_city_audit(self, city: str) -> None:
        """Mark a city with no source rows as checked under this contract."""
        city_counts = {
            **{status: 0 for status in AUDIT_STATUSES},
            "input_rows": 0,
            "emitted_rows": 0,
        }
        previous_counts = self.audit["by_city"].get(city, {})
        for key in self.audit["totals"]:
            self.audit["totals"][key] -= int(previous_counts.get(key, 0))
        self.audit["by_city"][city] = city_counts
        self.validated_cities.add(city)

    def write_audit_report(self) -> Path:
        """Atomically persist the cumulative validation audit report."""
        output_path = self.get_audit_report_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.audit)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload["unresolved_folder"] = str(self.get_unresolved_folder())
        temp_path = output_path.with_name(f".{output_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, default=str))
        temp_path.replace(output_path)
        return output_path

    def process_city_similarity(self, city: str) -> Tuple[int, int]:
        """Validate, orient, deduplicate, and export one city's similarity rows."""
        self.logger.info("Processing similarity data for city: %s", city)
        query_parts = self.get_city_source_queries(city)

        if not query_parts:
            self.logger.warning(
                "No optimized pairwise shard or merged files found for city: %s", city
            )
            self._record_empty_city_audit(city)
            self.write_audit_report()
            return 0, 0

        self._create_classified_table(" UNION ALL ".join(query_parts))
        base_query = self._resolved_base_query()
        inner_count = int(
            self.conn.execute(
                f"{base_query} SELECT COUNT(*) FROM deduped WHERE city_1_norm = city_2_norm"
            ).fetchone()[0]
        )
        inter_count = int(
            self.conn.execute(
                f"{base_query} SELECT COUNT(*) FROM deduped WHERE city_1_norm != city_2_norm"
            ).fetchone()[0]
        )
        membership_violations = int(
            self.conn.execute(
                f"""
                {base_query}
                SELECT COUNT(*)
                FROM deduped d
                WHERE NOT EXISTS (
                    SELECT 1 FROM h3_membership m
                    WHERE m.hex_id = d.hex_id1 AND m.city_norm = d.city_1_norm
                ) OR NOT EXISTS (
                    SELECT 1 FROM h3_membership m
                    WHERE m.hex_id = d.hex_id2 AND m.city_norm = d.city_2_norm
                )
                """
            ).fetchone()[0]
        )
        if membership_violations:
            raise RuntimeError(
                f"Refusing to write {membership_violations} rows that fail direct H3 membership"
            )
        unresolved_count = int(
            self.conn.execute(
                """
                SELECT COUNT(*) FROM classified_city_similarity
                WHERE validation_status IN ('unresolved', 'missing_city_h3')
                """
            ).fetchone()[0]
        )

        self.logger.info(
            "City %s: %d validated inner-city, %d validated inter-city, %d unresolved",
            city,
            inner_count,
            inter_count,
            unresolved_count,
        )

        output_file = self.get_output_file(city)
        if inter_count > 0:
            export_query = f"""
                {base_query}
                SELECT hex_id1, hex_id2, similarity, city_1, city_2
                FROM deduped
                WHERE city_1_norm != city_2_norm
            """
            output_file = self.export_intercity_dataset(city, export_query)
            self.logger.debug("Saved validated inter-city results to: %s", output_file)
        else:
            self.remove_output_path(output_file)

        self._write_unresolved_rows(city, unresolved_count)
        self._record_city_audit(city, inner_count + inter_count)
        self.write_audit_report()
        gc.collect()
        return inner_count, inter_count

    def run(self, city_meta_path: str) -> None:
        """Run aggregation across all cities in metadata."""
        self.logger.info("Starting optimized urban similarity aggregation")

        try:
            self.warn_if_pairwise_not_finished()

            city_meta = pd.read_csv(city_meta_path)
            cities = city_meta["City"].dropna().tolist()
            self.load_h3_membership(cities)
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
            audit_path = self.write_audit_report()
            self.logger.info("Membership audit written to %s", audit_path)
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
        default=7,
        help="H3 resolution level to aggregate",
    )
    parser.add_argument(
        "--pairwise-root",
        default="/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
        help="Root folder containing the optimized temp output tree",
    )
    parser.add_argument(
        "--h3-membership-root",
        default="/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_hex_summary",
        help="Root containing authoritative per-city H3 summary parquet files",
    )
    parser.add_argument(
        "--h3-input-template",
        default="dinov3_city={city}_res_exclude=None.parquet",
        help="Membership filename template containing a {city} placeholder",
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
    parser.add_argument(
        "--parquet-file-size",
        default="512MB",
        help="Approximate max size for each parquet part file, e.g. 512MB; set to 0 to write a single parquet file",
    )
    parser.add_argument(
        "--audit-report",
        default=None,
        help="Validation audit JSON path; defaults inside the export folder",
    )
    parser.add_argument(
        "--unresolved-folder",
        default=None,
        help="Folder for explicitly quarantined unresolved parquet rows",
    )
    parser.add_argument(
        "--audit-example-limit",
        type=int,
        default=5,
        help="Maximum examples retained for each validation status",
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
        "PARQUET_FILE_SIZE_BYTES": args.parquet_file_size,
        "H3_MEMBERSHIP_ROOT": args.h3_membership_root,
        "H3_INPUT_TEMPLATE": args.h3_input_template,
        "AUDIT_REPORT_PATH": args.audit_report,
        "UNRESOLVED_FOLDER": args.unresolved_folder,
        "AUDIT_EXAMPLE_LIMIT": args.audit_example_limit,
    }
    processor = OptimizedUrbanSimilarityProcessor(config, log_level="INFO")
    processor.run(args.city_meta)


if __name__ == "__main__":
    main()
