#!/usr/bin/env python3
"""
Purge old B5b temp shards from the optimized/temp tree only.

Default target:
    /lustre1/g/geog_pyloo/05_timemachine/_curated/
    c_city_classifiier_prob_similarity_by_pair/optimized/temp

Sample dry run:

    python3 /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/purge_old_temp_shards.py \
      --before-date 2025-05-24

Apply deletion:

    python3 /Users/yuan/Documents/GitHub/uvi-time-machine/_script/A-city-never-was/purge_old_temp_shards.py \
      --before-date 2025-05-24 \
      --apply
"""

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


DEFAULT_TEMP_ROOT = Path(
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/"
    "c_city_classifiier_prob_similarity_by_pair/optimized/temp"
)


def parse_date(value: str) -> date:
    """Parse `YYYY-MM-DD` into a date object."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def validate_temp_root(temp_root: Path) -> Path:
    """Refuse to operate outside an `optimized/temp` tree."""
    resolved = temp_root.expanduser().resolve()
    if resolved.name != "temp" or resolved.parent.name != "optimized":
        raise ValueError(f"Refusing to purge outside an optimized/temp tree: {resolved}")
    return resolved


def collect_purge_targets(temp_root: Path, before_date: date) -> list[Path]:
    """Return files under `temp_root` modified before the cutoff date."""
    root = validate_temp_root(temp_root)
    targets: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        modified_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        if modified_date < before_date:
            targets.append(path)
    return targets


def scan_stats(temp_root: Path) -> dict[str, date | int | None]:
    """Return total file count plus oldest/newest observed modification dates."""
    root = validate_temp_root(temp_root)
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        return {"total_files": 0, "oldest_date": None, "newest_date": None}

    modified_dates = [datetime.fromtimestamp(path.stat().st_mtime).date() for path in files]
    return {
        "total_files": len(files),
        "oldest_date": min(modified_dates),
        "newest_date": max(modified_dates),
    }


def prune_empty_parents(root: Path, start_dirs: Iterable[Path]) -> int:
    """Remove empty directories between each start dir and the root."""
    root = root.resolve()
    deleted = 0
    for start_dir in sorted({path.resolve() for path in start_dirs}, reverse=True):
        current = start_dir
        while current != root and root in current.parents:
            try:
                current.rmdir()
                deleted += 1
            except OSError:
                break
            current = current.parent
    return deleted


def find_temp_root_for_target(target: Path) -> Path:
    """Return the nearest ancestor that is the `optimized/temp` root."""
    current = target.resolve().parent
    while True:
        if current.name == "temp" and current.parent.name == "optimized":
            return current
        if current.parent == current:
            raise ValueError(f"Unable to find optimized/temp root for {target}")
        current = current.parent


def purge_targets(targets: list[Path], apply: bool) -> dict[str, int]:
    """Delete matched files and prune empty parent dirs when `apply` is true."""
    files_deleted = 0
    directories_deleted = 0
    parent_dirs = [target.parent for target in targets]
    roots = {find_temp_root_for_target(target) for target in targets}

    if apply:
        for target in targets:
            target.unlink(missing_ok=True)
            files_deleted += 1
        if roots:
            # All targets should share one temp root in normal usage.
            temp_root = sorted(roots)[0]
            directories_deleted = prune_empty_parents(temp_root, parent_dirs)

    return {
        "files_matched": len(targets),
        "files_deleted": files_deleted,
        "directories_deleted": directories_deleted,
    }


def format_preview(targets: list[Path], limit: int = 20) -> str:
    """Format the first few targets for human review."""
    if not targets:
        return "No matching files found."
    lines = [f"Matched files: {len(targets)}", "Preview:"]
    for path in targets[:limit]:
        lines.append(str(path))
    if len(targets) > limit:
        lines.append(f"... {len(targets) - limit} more")
    return "\n".join(lines)


def format_stats(stats: dict[str, date | int | None]) -> str:
    """Format scan stats for dry-run diagnostics."""
    if stats["total_files"] == 0:
        return "Scanned files: 0"
    return (
        f"Scanned files: {stats['total_files']}\n"
        f"Oldest mtime: {stats['oldest_date']}\n"
        f"Newest mtime: {stats['newest_date']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Purge files older than a cutoff date from the optimized/temp tree only."
    )
    parser.add_argument(
        "--temp-root",
        default=str(DEFAULT_TEMP_ROOT),
        help="Path to the optimized/temp tree to inspect.",
    )
    parser.add_argument(
        "--before-date",
        required=True,
        type=parse_date,
        help="Delete files last modified before this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched files. Without this flag, the script only prints a dry-run preview.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    temp_root = validate_temp_root(Path(args.temp_root))
    if not temp_root.exists():
        print(f"ERROR: temp root not found: {temp_root}")
        return 2

    stats = scan_stats(temp_root)
    print(format_stats(stats))
    targets = collect_purge_targets(temp_root, args.before_date)
    print(format_preview(targets))

    summary = purge_targets(targets, apply=args.apply)
    if args.apply:
        print(
            "Deleted "
            f"{summary['files_deleted']} files and pruned {summary['directories_deleted']} empty directories."
        )
    else:
        print("Dry run only. Re-run with --apply to delete these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
