#!/usr/bin/env python3
"""Command orchestrator for the DINOv3 visual-similarity pipeline."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
DEFAULT_REPO_DIR = f"{DEFAULT_ROOT}/uvi-time-machine/_script/A-city-never-was"
DEFAULT_CITY_META = f"{DEFAULT_ROOT}/uvi-time-machine/_script/city_meta.csv"
DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_VALFOLDER = f"{DEFAULT_ROOT}/_transformed/t_classifier_img_yolo8_inf_dir"
DEFAULT_TRAIN_TEST_FOLDER = f"{DEFAULT_ROOT}/_transformed/t_classifier_img_yolo8"
DEFAULT_EMBED_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_embed"
DEFAULT_SMOKE_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_embed_smoke"
DEFAULT_HEX_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_hex_summary"
DEFAULT_PAIRWISE_ROOT = f"{DEFAULT_ROOT}/_curated/c_city_dinov3_similarity_by_pair"
DEFAULT_TMP_ROOT = f"{DEFAULT_ROOT}/_tmp/duckdb_dinov3_similarity"

CITY_COLUMNS = ("City", "city", "city_name", "name")
CITY_STAGES = {"smoke", "embed", "aggregate"}
GLOBAL_STAGES = {"pairwise", "b5c", "summary"}
ALL_STAGES = ("embed", "aggregate", "pairwise", "b5c", "summary")


@dataclass(frozen=True)
class DINOv3PipelineConfig:
    repo_dir: Path = Path(DEFAULT_REPO_DIR)
    city_meta: Path = Path(DEFAULT_CITY_META)
    rootfolder: Path = Path(DEFAULT_ROOT)
    valfolder: Path = Path(DEFAULT_VALFOLDER)
    train_test_folder: Path = Path(DEFAULT_TRAIN_TEST_FOLDER)
    embed_root: Path = Path(DEFAULT_EMBED_ROOT)
    smoke_root: Path = Path(DEFAULT_SMOKE_ROOT)
    hex_root: Path = Path(DEFAULT_HEX_ROOT)
    pairwise_root: Path = Path(DEFAULT_PAIRWISE_ROOT)
    similarity_export_folder: Path = field(
        default_factory=lambda: Path(
            f"{DEFAULT_ROOT}/_curated/c_city_dinov3_similarity_"
            f"{datetime.now().strftime('%Y%m%d')}"
        )
    )
    summary_output: Path = Path(
        f"{DEFAULT_ROOT}/_curated/c_city_dinov3_similarity_summary_res=8.parquet"
    )
    model_name: str = DEFAULT_MODEL_NAME
    backend: str = "transformers"
    batch_size: int = 64
    smoke_batch_size: int = 2
    smoke_limit: int = 2
    device: str = "cuda"
    local_files_only: bool = True
    ignore_mismatched_sizes: bool = False
    res_exclude: int = 11
    resolution: int = 8
    threshold: float = -1.0
    row_block_size: int = 1000
    b5b_memory_limit: str = "16GB"
    b5c_memory_limit: str = "32GB"
    b5c_threads: int = 8
    b5c_parquet_file_size: str = "512MB"
    duckdb_temp_dir: Path = Path(DEFAULT_TMP_ROOT)
    python: str = "python"
    log_dir: Path = Path("logs/dinov3_similarity")
    log_level: str = "INFO"


def quote(value: object) -> str:
    return shlex.quote(str(value))


def command(parts: Iterable[object]) -> str:
    return " ".join(quote(part) for part in parts if part is not None and part != "")


def load_cities(
    city_meta: str | Path,
    city_index: int | None = None,
    explicit_cities: Sequence[str] | None = None,
) -> list[str]:
    """Load unique cities from metadata, optionally selecting a 1-based index."""
    if explicit_cities:
        cities = list(dict.fromkeys(str(city).strip() for city in explicit_cities if str(city).strip()))
    else:
        meta = pd.read_csv(city_meta)
        city_column = next((column for column in CITY_COLUMNS if column in meta.columns), None)
        if city_column is None:
            raise ValueError(
                f"City metadata must contain one of these columns: {', '.join(CITY_COLUMNS)}"
            )
        values = meta[city_column].dropna().astype(str).str.strip()
        cities = list(dict.fromkeys(value for value in values if value))

    if city_index is not None:
        if city_index < 1 or city_index > len(cities):
            raise IndexError(
                f"city_index={city_index} is outside 1..{len(cities)} for {city_meta}"
            )
        return [cities[city_index - 1]]

    return cities


def script_path(config: DINOv3PipelineConfig, name: str) -> Path:
    return config.repo_dir / name


def build_smoke_command(config: DINOv3PipelineConfig, city: str) -> str:
    parts: list[object] = [
        config.python,
        script_path(config, "B5d_dinov3_embed_city.py"),
        "--city",
        city,
        "--valfolder",
        config.valfolder,
        "--output-root",
        config.smoke_root,
        "--model-name",
        config.model_name,
        "--backend",
        config.backend,
        "--batch-size",
        config.smoke_batch_size,
        "--device",
        config.device,
    ]
    if config.local_files_only:
        parts.append("--local-files-only")
    if config.ignore_mismatched_sizes:
        parts.append("--ignore-mismatched-sizes")
    parts.extend(["--limit", config.smoke_limit])
    return command(parts)


def build_embed_command(config: DINOv3PipelineConfig, city: str) -> str:
    parts: list[object] = [
        config.python,
        script_path(config, "B5d_dinov3_embed_city.py"),
        "--city",
        city,
        "--valfolder",
        config.valfolder,
        "--output-root",
        config.embed_root,
        "--model-name",
        config.model_name,
        "--backend",
        config.backend,
        "--batch-size",
        config.batch_size,
        "--device",
        config.device,
    ]
    if config.local_files_only:
        parts.append("--local-files-only")
    if config.ignore_mismatched_sizes:
        parts.append("--ignore-mismatched-sizes")
    return command(parts)


def build_aggregate_command(config: DINOv3PipelineConfig, city: str) -> str:
    return command(
        [
            config.python,
            script_path(config, "B5e_dinov3_vector_summary.py"),
            "--city",
            city,
            "--rootfolder",
            config.rootfolder,
            "--input-root",
            config.embed_root,
            "--output-root",
            config.hex_root,
            "--train-test-folder",
            config.train_test_folder,
            "--res-exclude",
            config.res_exclude,
            "--log-level",
            config.log_level,
        ]
    )


def build_pairwise_command(config: DINOv3PipelineConfig) -> str:
    return command(
        [
            config.python,
            script_path(config, "B5b_compute_similarity_pairwise-optimized.py"),
            "--resolution",
            config.resolution,
            "--city-meta",
            config.city_meta,
            "--source-root",
            config.hex_root,
            "--output-root",
            config.pairwise_root,
            "--input-template",
            "dinov3_city={city}_res_exclude={res_exclude}.parquet",
            "--feature-prefix",
            "e_",
            "--threshold",
            config.threshold,
            "--row-block-size",
            config.row_block_size,
            "--memory-limit",
            config.b5b_memory_limit,
            "--res-exclude",
            config.res_exclude,
            "--log-dir",
            config.log_dir,
        ]
    )


def build_b5c_command(config: DINOv3PipelineConfig) -> str:
    return command(
        [
            config.python,
            script_path(config, "B5c_pairwise_agg_optimized.py"),
            "--resolution",
            config.resolution,
            "--city-meta",
            config.city_meta,
            "--pairwise-root",
            config.pairwise_root,
            "--export-folder",
            config.similarity_export_folder,
            "--progress-file",
            config.pairwise_root / "optimized" / f"_progress_res={config.resolution}_optimized.json",
            "--resume",
            "--agg-progress-file",
            config.rootfolder / "_curated" / f"c_city_dinov3_similarity_agg_progress_res={config.resolution}.json",
            "--duckdb-memory-limit",
            config.b5c_memory_limit,
            "--duckdb-temp-dir",
            config.duckdb_temp_dir,
            "--duckdb-threads",
            config.b5c_threads,
            "--parquet-file-size",
            config.b5c_parquet_file_size,
        ]
    )


def build_summary_command(config: DINOv3PipelineConfig) -> str:
    return command(
        [
            config.python,
            script_path(config, "B5h_summarize_dinov3_citypair_similarity.py"),
            "--input-folder",
            config.similarity_export_folder,
            "--output",
            config.summary_output,
            "--city-meta",
            config.city_meta,
        ]
    )


def build_stage_commands(
    stage: str,
    config: DINOv3PipelineConfig,
    cities: Sequence[str],
) -> list[str]:
    """Return shell commands for one pipeline stage."""
    if stage == "smoke":
        city = cities[0] if cities else "Hong Kong"
        return [build_smoke_command(config, city)]
    if stage == "embed":
        return [build_embed_command(config, city) for city in cities]
    if stage == "aggregate":
        return [build_aggregate_command(config, city) for city in cities]
    if stage == "pairwise":
        return [build_pairwise_command(config)]
    if stage == "b5c":
        return [build_b5c_command(config)]
    if stage == "summary":
        return [build_summary_command(config)]
    if stage == "all":
        commands: list[str] = []
        for substage in ALL_STAGES:
            commands.extend(build_stage_commands(substage, config, cities))
        return commands
    raise ValueError(f"Unsupported DINOv3 stage: {stage}")


def render_slurm_array_script(
    job_name: str,
    stage: str,
    config: DINOv3PipelineConfig,
    city_count: int,
    array_concurrency: int,
    time_limit: str,
    partition: str,
    gres: str | None,
    cpus_per_task: int,
    mem: str,
) -> str:
    """Render a city-array SLURM script for embed or aggregate stages."""
    if stage not in CITY_STAGES:
        raise ValueError(f"SLURM array stages must be one of {sorted(CITY_STAGES)}")
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition={partition}
{gres_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --array=1-{city_count}%{array_concurrency}

set -euo pipefail

REPO_DIR={quote(config.repo_dir)}
cd "${{REPO_DIR}}"
mkdir -p logs/slurm

{quote(config.python)} {quote(config.repo_dir / "dinov3_pipeline.py")} \\
  --stage {stage} \\
  --city-meta {quote(config.city_meta)} \\
  --city-index ${{SLURM_ARRAY_TASK_ID}} \\
  --repo-dir {quote(config.repo_dir)} \\
  --model-name {quote(config.model_name)} \\
  --backend {quote(config.backend)} \\
  --device {quote(config.device)} \\
  --batch-size {config.batch_size} \\
  --res-exclude {config.res_exclude} \\
  --resolution {config.resolution} \\
  --execute
"""


def config_from_args(args: argparse.Namespace) -> DINOv3PipelineConfig:
    return DINOv3PipelineConfig(
        repo_dir=args.repo_dir,
        city_meta=args.city_meta,
        rootfolder=args.rootfolder,
        valfolder=args.valfolder,
        train_test_folder=args.train_test_folder,
        embed_root=args.embed_root,
        smoke_root=args.smoke_root,
        hex_root=args.hex_root,
        pairwise_root=args.pairwise_root,
        similarity_export_folder=args.similarity_export_folder,
        summary_output=args.summary_output,
        model_name=args.model_name,
        backend=args.backend,
        batch_size=args.batch_size,
        smoke_batch_size=args.smoke_batch_size,
        smoke_limit=args.smoke_limit,
        device=args.device,
        local_files_only=args.local_files_only,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        res_exclude=args.res_exclude,
        resolution=args.resolution,
        threshold=args.threshold,
        row_block_size=args.row_block_size,
        b5b_memory_limit=args.b5b_memory_limit,
        b5c_memory_limit=args.b5c_memory_limit,
        b5c_threads=args.b5c_threads,
        b5c_parquet_file_size=args.b5c_parquet_file_size,
        duckdb_temp_dir=args.duckdb_temp_dir,
        python=args.python,
        log_dir=args.log_dir,
        log_level=args.log_level,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["smoke", "embed", "aggregate", "pairwise", "b5c", "summary", "all"],
    )
    parser.add_argument("--city", action="append", help="City name; may be repeated")
    parser.add_argument("--city-index", type=int, help="1-based city index for SLURM arrays")
    parser.add_argument("--city-meta", type=Path, default=Path(DEFAULT_CITY_META))
    parser.add_argument("--repo-dir", type=Path, default=Path(DEFAULT_REPO_DIR))
    parser.add_argument("--rootfolder", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--valfolder", type=Path, default=Path(DEFAULT_VALFOLDER))
    parser.add_argument("--train-test-folder", type=Path, default=Path(DEFAULT_TRAIN_TEST_FOLDER))
    parser.add_argument("--embed-root", type=Path, default=Path(DEFAULT_EMBED_ROOT))
    parser.add_argument("--smoke-root", type=Path, default=Path(DEFAULT_SMOKE_ROOT))
    parser.add_argument("--hex-root", type=Path, default=Path(DEFAULT_HEX_ROOT))
    parser.add_argument("--pairwise-root", type=Path, default=Path(DEFAULT_PAIRWISE_ROOT))
    parser.add_argument(
        "--similarity-export-folder",
        type=Path,
        default=Path(
            f"{DEFAULT_ROOT}/_curated/c_city_dinov3_similarity_{datetime.now().strftime('%Y%m%d')}"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(f"{DEFAULT_ROOT}/_curated/c_city_dinov3_similarity_summary_res=8.parquet"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help=(
            "Pass ignore_mismatched_sizes=True to transformers for staged local "
            "checkpoints with expected tensor-size mismatches"
        ),
    )
    parser.add_argument("--res-exclude", type=int, default=11)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--row-block-size", type=int, default=1000)
    parser.add_argument("--b5b-memory-limit", default="16GB")
    parser.add_argument("--b5c-memory-limit", default="32GB")
    parser.add_argument("--b5c-threads", type=int, default=8)
    parser.add_argument("--b5c-parquet-file-size", default="512MB")
    parser.add_argument("--duckdb-temp-dir", type=Path, default=Path(DEFAULT_TMP_ROOT))
    parser.add_argument("--python", default="python")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dinov3_similarity"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    cities = load_cities(config.city_meta, city_index=args.city_index, explicit_cities=args.city)
    commands = build_stage_commands(args.stage, config, cities)

    for built_command in commands:
        print(built_command, flush=True)
        if args.execute:
            subprocess.run(built_command, shell=True, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
