#!/usr/bin/env python3
"""
Summarize inter-city similarity by landuse bucket from optimized pairwise shards.
"""

import argparse
import logging
import os
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


DEFAULT_PAIRWISE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_similarity_by_pair"
)
DEFAULT_STAGE2_LANDUSE_ROOT = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/landuse"
DEFAULT_STAGE3_LANDUSE_ROOT = (
    "/lustre1/g/geog_pyloo/05_timemachine/_transformed/landuse_poi_res=8"
)
DEFAULT_EXPORT_FOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_by_landuse"
)
DEFAULT_DUCKDB_TEMP_DIR = (
    "/lustre1/g/geog_pyloo/05_timemachine/_tmp/duckdb_city_similarity_by_landuse"
)

US_STATE_CODES = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
}

US_STATE_CODES_SQL = (
    "al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|"
    "mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|"
    "vt|va|wa|wv|wi|wy"
)

NORMALIZE_CITY_SQL = f"""
CASE
    WHEN {{col}} LIKE '%,%'
         AND length(replace(replace(lower({{col}}), ',', ''), ' ', '')) > 4
         AND replace(replace(lower({{col}}), ',', ''), ' ', '') ~ '({US_STATE_CODES_SQL})$'
    THEN regexp_replace(
        replace(replace(lower({{col}}), ',', ''), ' ', ''),
        '({US_STATE_CODES_SQL})$',
        ''
    )
    ELSE replace(replace(lower({{col}}), ',', ''), ' ', '')
END
"""

LANDUSE_TYPES = {
    "cbd": {
        "primary": ["cbd_high_confidence"],
        "fallback_pattern": "cbd",
        "exclude_pattern": None,
    },
    "tourism": {
        "primary": [
            "tourism_high_confidence",
            "mixed_cbd_tourism",
            "mixed_tourism_cbd",
        ],
        "fallback_pattern": "tourism",
        "exclude_pattern": None,
    },
    "residential": {
        "primary": [
            "residential_high_confidence",
            "mixed_residential_commercial",
            "mixed_cbd_residential",
            "mixed_residential_cbd",
        ],
        "fallback_pattern": "residential",
        "exclude_pattern": None,
    },
    "highdensity": {
        "primary": None,
        "fallback_pattern": None,
        "exclude_pattern": [
            "low_density",
            "industrial_high_confidence",
            "industrial_medium_confidence",
        ],
    },
}


@dataclass(frozen=True)
class LanduseSourceConfig:
    source: str
    root: str
    file_pattern: str
    city_regex: str
    label_column: str
    resolution_as_str: bool


@dataclass(frozen=True)
class ResolutionCoverageReport:
    resolution: int
    city_names: list[str]
    city_count: int
    is_sparse: bool


def normalize_city_name(city: str) -> str:
    """Normalize city name for cross-source joins."""
    normalized = unicodedata.normalize("NFD", city)
    cleaned = normalized.lower().replace(",", "").replace(" ", "")
    if "," in city and len(cleaned) > 4 and cleaned[-2:] in US_STATE_CODES:
        return cleaned[:-2]
    return cleaned


def setup_logging(log_level: str) -> logging.Logger:
    """Configure a local logger for B7 runs."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"b7_similarity_by_landuse_{timestamp}.log"

    logger = logging.getLogger(f"{__name__}.{timestamp}")
    logger.setLevel(getattr(logging, log_level))
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized. Log file: %s", log_file)
    return logger


def create_connection(
    memory_limit: str | None = None,
    temp_dir: str | None = None,
    threads: int | None = None,
):
    """Create and optionally configure a DuckDB connection."""
    con = duckdb.connect(":memory:")
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
    if threads:
        con.execute(f"SET threads TO {int(threads)}")
    return con


def get_landuse_source_config(
    source: str,
    stage2_root: str = DEFAULT_STAGE2_LANDUSE_ROOT,
    stage3_root: str = DEFAULT_STAGE3_LANDUSE_ROOT,
    tier_method: str = "pct",
) -> LanduseSourceConfig:
    """Return source-specific parquet location metadata."""
    if source == "stage2":
        return LanduseSourceConfig(
            source=source,
            root=stage2_root,
            file_pattern="{city_lower}_h3_landuse_enhanced.parquet",
            city_regex=r"([^/]+)_h3_landuse_enhanced\.parquet$",
            label_column="landuse_type",
            resolution_as_str=True,
        )

    if source == "stage3":
        label_column = "tier_pct" if tier_method == "pct" else "tier_z"
        return LanduseSourceConfig(
            source=source,
            root=stage3_root,
            file_pattern="{city_lower}_h3_poi_tiers.parquet",
            city_regex=r"([^/]+)_h3_poi_tiers\.parquet$",
            label_column=label_column,
            resolution_as_str=False,
        )

    raise ValueError(f"Unsupported landuse source: {source}")


def get_landuse_keys(source: str) -> list[str]:
    """Return supported landuse buckets for one source."""
    if source == "stage2":
        return ["cbd", "tourism", "residential", "highdensity"]
    if source == "stage3":
        return ["core", "suburban", "rural", "built"]
    raise ValueError(f"Unsupported landuse source: {source}")


def resolve_landuse_keys(landuse_key: str | None, landuse_source_name: str) -> list[str]:
    """Resolve the requested landuse selection."""
    if landuse_key == "allhex":
        return ["all"]

    if landuse_key in (None, "all"):
        return [*get_landuse_keys(landuse_source_name), "all"]

    return [landuse_key]


def build_summary_output_path(
    export_folder: str,
    landuse_type: str,
    res: int,
    zero_fill_avg: bool,
    landuse_source_name: str = "stage3",
    tier_method: str = "pct",
) -> str:
    """Return the output CSV path for one landuse bucket."""
    suffix = "_zerofill" if zero_fill_avg else ""
    if landuse_source_name == "stage3":
        suffix = f"_{tier_method}{suffix}"
    return os.path.join(
        export_folder, f"similarity_summary_{landuse_type}_city_res={res}{suffix}.csv"
    )


def summarize_resolution_coverage(
    records: Iterable[tuple[str, list[int]]],
    resolution: int,
) -> ResolutionCoverageReport:
    """Summarize which landuse files contain a requested resolution."""
    city_names = sorted(
        normalize_city_name(city)
        for city, resolutions in records
        if resolution in {int(value) for value in resolutions}
    )
    return ResolutionCoverageReport(
        resolution=resolution,
        city_names=city_names,
        city_count=len(city_names),
        is_sparse=len(city_names) < 2,
    )


def inspect_landuse_resolution_coverage(
    con,
    config: LanduseSourceConfig,
) -> ResolutionCoverageReport:
    """Inspect which landuse parquet files contain each resolution."""
    pattern = os.path.join(config.root, config.file_pattern.format(city_lower="*"))
    query = f"""
    SELECT
        regexp_extract(filename, '{config.city_regex}', 1) AS city_name,
        list(DISTINCT CAST(resolution AS INTEGER) ORDER BY CAST(resolution AS INTEGER)) AS resolutions
    FROM read_parquet('{pattern}', filename=true)
    GROUP BY 1
    ORDER BY 1
    """
    records = con.execute(query).fetchall()
    return summarize_resolution_coverage(records, resolution=-1)


def get_pairwise_temp_pattern(pairwise_root: str, res: int) -> str:
    """Return the glob for optimized pairwise temp shards."""
    return str(
        Path(pairwise_root)
        / "optimized"
        / "temp"
        / "city1=*"
        / "city2=*"
        / f"part_res={res}.parquet"
    )


def load_similarity_hex_ids(con, pairwise_root: str, res: int) -> pd.DataFrame:
    """Derive the set of similarity-covered hex IDs by city from temp shards."""
    pattern = get_pairwise_temp_pattern(pairwise_root, res)
    norm_city1 = NORMALIZE_CITY_SQL.format(col="city1")
    norm_city2 = NORMALIZE_CITY_SQL.format(col="city2")
    query = f"""
    SELECT DISTINCT city_lower, hex_id
    FROM (
        SELECT {norm_city1} AS city_lower, CAST(hex_id1 AS VARCHAR) AS hex_id
        FROM read_parquet('{pattern}')
        UNION ALL
        SELECT {norm_city2} AS city_lower, CAST(hex_id2 AS VARCHAR) AS hex_id
        FROM read_parquet('{pattern}')
    )
    ORDER BY city_lower, hex_id
    """
    return con.execute(query).df()


def load_landuse_hex_ids(
    con,
    landuse_type: str,
    similarity_hex_df: pd.DataFrame,
    res: int,
    config: LanduseSourceConfig,
) -> pd.DataFrame:
    """Load landuse-filtered hex IDs that also appear in similarity coverage."""
    if landuse_type == "all":
        return similarity_hex_df.copy()

    con.register("sim_hexes", similarity_hex_df)
    pattern = os.path.join(config.root, config.file_pattern.format(city_lower="*"))
    city_extract_sql = f"regexp_extract(filename, '{config.city_regex}', 1)"
    res_value = f"'{res}'" if config.resolution_as_str else f"{res}"

    if config.source == "stage3":
        allowed = {"core", "suburban", "rural", "built"}
        if landuse_type not in allowed:
            raise ValueError(
                f"Unknown landuse tier for stage3: {landuse_type}. Allowed: {sorted(allowed)}"
            )

        if landuse_type == "built":
            query = f"""
            SELECT DISTINCT l.h3_index AS hex_id, {city_extract_sql} AS city_lower
            FROM read_parquet('{pattern}', filename=true) l
            INNER JOIN sim_hexes s
                ON {city_extract_sql} = s.city_lower
               AND CAST(l.h3_index AS VARCHAR) = CAST(s.hex_id AS VARCHAR)
            WHERE l.resolution = {res_value}
              AND l.poi_count > 0
            """
            return con.execute(query).df()

        query = f"""
        SELECT DISTINCT l.h3_index AS hex_id, {city_extract_sql} AS city_lower
        FROM read_parquet('{pattern}', filename=true) l
        INNER JOIN sim_hexes s
            ON {city_extract_sql} = s.city_lower
           AND CAST(l.h3_index AS VARCHAR) = CAST(s.hex_id AS VARCHAR)
        WHERE l.resolution = {res_value}
          AND l.{config.label_column} = '{landuse_type}'
        """
        return con.execute(query).df()

    landuse_config = LANDUSE_TYPES.get(landuse_type)
    if not landuse_config:
        raise ValueError(f"Unknown landuse type: {landuse_type}")

    primary_types = landuse_config["primary"]
    fallback_pattern = landuse_config["fallback_pattern"]
    exclude_pattern = landuse_config["exclude_pattern"]

    if exclude_pattern and primary_types is None:
        excluded = "', '".join(exclude_pattern)
        query = f"""
        SELECT DISTINCT l.h3_index AS hex_id, {city_extract_sql} AS city_lower
        FROM read_parquet('{pattern}', filename=true) l
        INNER JOIN sim_hexes s
            ON {city_extract_sql} = s.city_lower
           AND CAST(l.h3_index AS VARCHAR) = CAST(s.hex_id AS VARCHAR)
        WHERE l.resolution = {res_value}
          AND l.{config.label_column} NOT IN ('{excluded}')
        """
        return con.execute(query).df()

    primary_type_str = "', '".join(primary_types)
    query_primary = f"""
    SELECT DISTINCT l.h3_index AS hex_id, {city_extract_sql} AS city_lower
    FROM read_parquet('{pattern}', filename=true) l
    INNER JOIN sim_hexes s
        ON {city_extract_sql} = s.city_lower
       AND CAST(l.h3_index AS VARCHAR) = CAST(s.hex_id AS VARCHAR)
    WHERE l.resolution = {res_value}
      AND l.{config.label_column} IN ('{primary_type_str}')
    """
    primary_df = con.execute(query_primary).df()

    if fallback_pattern is None:
        return primary_df[["hex_id", "city_lower"]].drop_duplicates()

    all_cities = set(similarity_hex_df["city_lower"].unique())
    cities_with_primary = set(primary_df["city_lower"].unique())
    cities_needing_fallback = sorted(all_cities - cities_with_primary)
    if not cities_needing_fallback:
        return primary_df[["hex_id", "city_lower"]].drop_duplicates()

    city_list = "', '".join(cities_needing_fallback)
    query_fallback = f"""
    SELECT DISTINCT l.h3_index AS hex_id, {city_extract_sql} AS city_lower
    FROM read_parquet('{pattern}', filename=true) l
    INNER JOIN sim_hexes s
        ON {city_extract_sql} = s.city_lower
       AND CAST(l.h3_index AS VARCHAR) = CAST(s.hex_id AS VARCHAR)
    WHERE l.resolution = {res_value}
      AND {city_extract_sql} IN ('{city_list}')
      AND l.{config.label_column} LIKE '%{fallback_pattern}%'
    """
    fallback_df = con.execute(query_fallback).df()
    return (
        pd.concat([primary_df, fallback_df], ignore_index=True)[["hex_id", "city_lower"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def compute_city_hex_counts(hex_ids_df: pd.DataFrame) -> pd.DataFrame:
    """Count filtered hexes per city."""
    return (
        hex_ids_df.groupby("city_lower", as_index=False)["hex_id"]
        .nunique()
        .rename(columns={"hex_id": "hex_count"})
    )


def aggregate_similarity(
    con,
    pairwise_root: str,
    res: int,
    hex_ids_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate filtered similarities by city pair."""
    pattern = get_pairwise_temp_pattern(pairwise_root, res)
    con.register("hex_filter", hex_ids_df[["hex_id"]].drop_duplicates())
    norm_city1 = NORMALIZE_CITY_SQL.format(col="city1")
    norm_city2 = NORMALIZE_CITY_SQL.format(col="city2")
    query = f"""
    WITH filtered AS (
        SELECT
            {norm_city1} AS city1_norm,
            {norm_city2} AS city2_norm,
            CAST(hex_id1 AS VARCHAR) AS hex_id1,
            CAST(hex_id2 AS VARCHAR) AS hex_id2,
            similarity
        FROM read_parquet('{pattern}')
        WHERE CAST(hex_id1 AS VARCHAR) IN (SELECT hex_id FROM hex_filter)
          AND CAST(hex_id2 AS VARCHAR) IN (SELECT hex_id FROM hex_filter)
    ),
    normalized AS (
        SELECT
            CASE WHEN city1_norm <= city2_norm THEN city1_norm ELSE city2_norm END AS city_1,
            CASE WHEN city1_norm <= city2_norm THEN city2_norm ELSE city1_norm END AS city_2,
            CASE WHEN city1_norm <= city2_norm THEN hex_id1 ELSE hex_id2 END AS hex_id1,
            CASE WHEN city1_norm <= city2_norm THEN hex_id2 ELSE hex_id1 END AS hex_id2,
            similarity
        FROM filtered
    ),
    hex_pair_max AS (
        SELECT
            city_1,
            city_2,
            hex_id1,
            hex_id2,
            MAX(similarity) AS similarity
        FROM normalized
        WHERE city_1 != city_2
        GROUP BY city_1, city_2, hex_id1, hex_id2
    )
    SELECT
        city_1,
        city_2,
        MAX(similarity) AS "max(similarity)",
        AVG(similarity) AS "avg(similarity)",
        COUNT(*) AS pair_count
    FROM hex_pair_max
    GROUP BY city_1, city_2
    ORDER BY city_1, city_2
    """
    return con.execute(query).df()


def aggregate_similarity_zero_fill(
    con,
    pairwise_root: str,
    res: int,
    hex_ids_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate filtered similarities with zero-fill average scoring."""
    pattern = get_pairwise_temp_pattern(pairwise_root, res)
    city_hex_counts = compute_city_hex_counts(hex_ids_df)
    con.register("hex_filter", hex_ids_df[["hex_id"]].drop_duplicates())
    con.register("city_hex_counts", city_hex_counts)
    norm_city1 = NORMALIZE_CITY_SQL.format(col="city1")
    norm_city2 = NORMALIZE_CITY_SQL.format(col="city2")
    query = f"""
    WITH counts AS (
        SELECT city_lower AS city_1, hex_count AS hex_count_1 FROM city_hex_counts
    ),
    counts_2 AS (
        SELECT city_lower AS city_2, hex_count AS hex_count_2 FROM city_hex_counts
    ),
    all_pairs AS (
        SELECT
            c.city_1,
            c2.city_2,
            c.hex_count_1 * c2.hex_count_2 AS pair_count
        FROM counts c
        CROSS JOIN counts_2 c2
        WHERE c.city_1 < c2.city_2
    ),
    filtered AS (
        SELECT
            {norm_city1} AS city1_norm,
            {norm_city2} AS city2_norm,
            CAST(hex_id1 AS VARCHAR) AS hex_id1,
            CAST(hex_id2 AS VARCHAR) AS hex_id2,
            similarity
        FROM read_parquet('{pattern}')
        WHERE CAST(hex_id1 AS VARCHAR) IN (SELECT hex_id FROM hex_filter)
          AND CAST(hex_id2 AS VARCHAR) IN (SELECT hex_id FROM hex_filter)
    ),
    normalized AS (
        SELECT
            CASE WHEN city1_norm <= city2_norm THEN city1_norm ELSE city2_norm END AS city_1,
            CASE WHEN city1_norm <= city2_norm THEN city2_norm ELSE city1_norm END AS city_2,
            CASE WHEN city1_norm <= city2_norm THEN hex_id1 ELSE hex_id2 END AS hex_id1,
            CASE WHEN city1_norm <= city2_norm THEN hex_id2 ELSE hex_id1 END AS hex_id2,
            similarity
        FROM filtered
    ),
    hex_pair_max AS (
        SELECT
            city_1,
            city_2,
            hex_id1,
            hex_id2,
            MAX(similarity) AS similarity
        FROM normalized
        WHERE city_1 != city_2
        GROUP BY city_1, city_2, hex_id1, hex_id2
    ),
    city_pair_sum AS (
        SELECT
            city_1,
            city_2,
            SUM(similarity) AS sum_similarity,
            MAX(similarity) AS max_similarity
        FROM hex_pair_max
        GROUP BY city_1, city_2
    )
    SELECT
        p.city_1,
        p.city_2,
        COALESCE(s.max_similarity, 0) AS "max(similarity)",
        CASE
            WHEN p.pair_count > 0
            THEN COALESCE(s.sum_similarity, 0) / p.pair_count
            ELSE 0
        END AS "avg(similarity)",
        p.pair_count
    FROM all_pairs p
    LEFT JOIN city_pair_sum s
        ON p.city_1 = s.city_1 AND p.city_2 = s.city_2
    ORDER BY p.city_1, p.city_2
    """
    return con.execute(query).df()


def process_landuse_type(
    con,
    logger: logging.Logger,
    pairwise_root: str,
    landuse_type: str,
    similarity_hex_df: pd.DataFrame,
    landuse_config: LanduseSourceConfig,
    export_folder: str,
    res: int,
    zero_fill_avg: bool,
    tier_method: str,
) -> pd.DataFrame:
    """Run one landuse summary and write its CSV."""
    logger.info("Processing landuse bucket: %s", landuse_type)
    hex_ids_df = load_landuse_hex_ids(
        con=con,
        landuse_type=landuse_type,
        similarity_hex_df=similarity_hex_df,
        res=res,
        config=landuse_config,
    )
    logger.info("Selected %d hexes for %s", len(hex_ids_df), landuse_type)

    if hex_ids_df.empty:
        logger.warning("No hexes found for %s at resolution %s", landuse_type, res)
        return pd.DataFrame()

    if zero_fill_avg:
        summary_df = aggregate_similarity_zero_fill(
            con=con,
            pairwise_root=pairwise_root,
            res=res,
            hex_ids_df=hex_ids_df,
        )
    else:
        summary_df = aggregate_similarity(
            con=con,
            pairwise_root=pairwise_root,
            res=res,
            hex_ids_df=hex_ids_df,
        )

    output_path = build_summary_output_path(
        export_folder=export_folder,
        landuse_type=landuse_type,
        res=res,
        zero_fill_avg=zero_fill_avg,
        landuse_source_name=landuse_config.source,
        tier_method=tier_method,
    )
    summary_df.to_csv(output_path, index=False)
    logger.info("Wrote %d summary rows to %s", len(summary_df), output_path)
    return summary_df


def run_similarity_by_landuse(
    pairwise_root: str,
    landuse_source: str,
    landuse_key: str,
    export_folder: str,
    res: int,
    zero_fill_avg: bool,
    stage2_landuse_root: str,
    stage3_landuse_root: str,
    tier_method: str,
    allow_sparse_landuse: bool,
    check_only: bool,
    duckdb_memory_limit: str | None,
    duckdb_temp_dir: str | None,
    duckdb_threads: int | None,
    log_level: str,
) -> dict[str, pd.DataFrame]:
    """Run the B7 landuse similarity pipeline."""
    logger = setup_logging(log_level)
    export_path = Path(export_folder)
    export_path.mkdir(parents=True, exist_ok=True)

    con = create_connection(
        memory_limit=duckdb_memory_limit,
        temp_dir=duckdb_temp_dir,
        threads=duckdb_threads,
    )
    try:
        landuse_config = get_landuse_source_config(
            source=landuse_source,
            stage2_root=stage2_landuse_root,
            stage3_root=stage3_landuse_root,
            tier_method=tier_method,
        )

        pattern = os.path.join(
            landuse_config.root,
            landuse_config.file_pattern.format(city_lower="*"),
        )
        coverage_query = f"""
        SELECT
            regexp_extract(filename, '{landuse_config.city_regex}', 1) AS city_name,
            list(DISTINCT CAST(resolution AS INTEGER) ORDER BY CAST(resolution AS INTEGER)) AS resolutions
        FROM read_parquet('{pattern}', filename=true)
        GROUP BY 1
        ORDER BY 1
        """
        coverage_records = con.execute(coverage_query).fetchall()
        coverage = summarize_resolution_coverage(coverage_records, res)
        logger.info(
            "Landuse coverage for %s res=%s: %d cities",
            landuse_source,
            res,
            coverage.city_count,
        )
        if coverage.city_names:
            logger.info("Cities with landuse coverage: %s", ", ".join(coverage.city_names))

        if check_only:
            return {}

        if coverage.city_count == 0:
            raise RuntimeError(
                f"No {landuse_source} landuse parquet files contain resolution {res} under {landuse_config.root}"
            )

        if coverage.is_sparse and not allow_sparse_landuse:
            raise RuntimeError(
                f"Sparse {landuse_source} landuse coverage at resolution {res}: "
                f"{coverage.city_count} city ({', '.join(coverage.city_names)}). "
                "Re-run with --allow-sparse-landuse to continue anyway."
            )

        similarity_hex_df = load_similarity_hex_ids(con, pairwise_root, res)
        logger.info(
            "Loaded %d similarity-covered hex IDs across %d cities",
            len(similarity_hex_df),
            similarity_hex_df["city_lower"].nunique() if not similarity_hex_df.empty else 0,
        )

        results: dict[str, pd.DataFrame] = {}
        for current_landuse_type in resolve_landuse_keys(landuse_key, landuse_source):
            summary_df = process_landuse_type(
                con=con,
                logger=logger,
                pairwise_root=pairwise_root,
                landuse_type=current_landuse_type,
                similarity_hex_df=similarity_hex_df,
                landuse_config=landuse_config,
                export_folder=export_folder,
                res=res,
                zero_fill_avg=zero_fill_avg,
                tier_method=tier_method,
            )
            results[current_landuse_type] = summary_df

        return results
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize inter-city similarity by landuse bucket from optimized pairwise shards."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=8,
        help="H3 resolution to process.",
    )
    parser.add_argument(
        "--pairwise-root",
        default=DEFAULT_PAIRWISE_ROOT,
        help="Remote root folder containing optimized pairwise temp shards.",
    )
    parser.add_argument(
        "--landuse-source",
        choices=["stage2", "stage3"],
        default="stage3",
        help="Landuse source to use for hex filtering.",
    )
    parser.add_argument(
        "--landuse",
        default="all",
        choices=sorted(set(get_landuse_keys("stage2")) | set(get_landuse_keys("stage3")))
        + ["all", "allhex"],
        help="Landuse bucket to process, 'all' for all buckets, or 'allhex' for no landuse filter.",
    )
    parser.add_argument(
        "--stage2-landuse-root",
        default=DEFAULT_STAGE2_LANDUSE_ROOT,
        help="Remote folder containing stage2 landuse parquet files.",
    )
    parser.add_argument(
        "--stage3-landuse-root",
        default=DEFAULT_STAGE3_LANDUSE_ROOT,
        help="Remote folder containing stage3 POI-tier parquet files.",
    )
    parser.add_argument(
        "--tier-method",
        choices=["pct", "z"],
        default="pct",
        help="Stage3 tier label to use.",
    )
    parser.add_argument(
        "--export-folder",
        default=DEFAULT_EXPORT_FOLDER,
        help="Output folder for summary CSVs.",
    )
    parser.add_argument(
        "--zero-fill-avg",
        action="store_true",
        help="Use expected hex-pair counts to compute zero-fill average similarity.",
    )
    parser.add_argument(
        "--allow-sparse-landuse",
        action="store_true",
        help="Continue even when the requested landuse resolution is present for fewer than two cities.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only inspect landuse resolution coverage and exit.",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default="32GB",
        help="Optional DuckDB memory limit.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=DEFAULT_DUCKDB_TEMP_DIR,
        help="DuckDB spill directory for large runs.",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=None,
        help="Optional DuckDB thread count.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    args = parser.parse_args()

    run_similarity_by_landuse(
        pairwise_root=args.pairwise_root,
        landuse_source=args.landuse_source,
        landuse_key=args.landuse,
        export_folder=args.export_folder,
        res=args.resolution,
        zero_fill_avg=args.zero_fill_avg,
        stage2_landuse_root=args.stage2_landuse_root,
        stage3_landuse_root=args.stage3_landuse_root,
        tier_method=args.tier_method,
        allow_sparse_landuse=args.allow_sparse_landuse,
        check_only=args.check_only,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_temp_dir=args.duckdb_temp_dir,
        duckdb_threads=args.duckdb_threads,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
