#!/usr/bin/env python3
"""Audit, manifest, and validate affected DINOv3 resolution-7 recovery data."""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import h3
import duckdb
import pandas as pd

from dinov3_utils import normalize_city_name, resolve_city_file_stem


DEFAULT_ROOT = Path("/lustre1/g/geog_pyloo/05_timemachine")
DEFAULT_AFFECTED_CITIES = (
    "Amsterdam",
    "Gombe",
    "Kampala",
    "Kozhikode",
    "Malegaon",
    "Sitapur",
    "Vijayawada",
)
FINAL_COLUMNS = ["hex_id1", "hex_id2", "similarity", "city_1", "city_2"]
KNOWN_CITY_ALIASES = {
    "kozhikode": ("Calicut",),
    "vijayawada": ("Bezawada",),
}


@dataclass(frozen=True)
class CityIndexDiscovery:
    city: str
    status: str
    path: Path | None
    candidates: tuple[Path, ...]
    checked_roots: list[dict[str, object]]
    aliases: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["path"] = str(self.path) if self.path else None
        result["candidates"] = [str(path) for path in self.candidates]
        result["aliases"] = list(self.aliases)
        return result


@dataclass(frozen=True)
class RecoveryPaths:
    root: Path
    index_roots: tuple[Path, ...]
    embed_root: Path
    h3_root: Path
    pairwise_root: Path
    aggregate_root: Path
    required_h3_root: Path | None = None
    core_h3_root: Path | None = None
    resolution: int = 7


def _comparison_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _city_aliases(
    city: str,
    metadata_aliases: Sequence[str] = (),
    configured_aliases: Sequence[str] = (),
) -> tuple[str, ...]:
    known = KNOWN_CITY_ALIASES.get(normalize_city_name(city), ())
    values = [city, *metadata_aliases, *configured_aliases, *known]
    return tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _stem_matches_alias(stem: str, aliases: Sequence[str]) -> bool:
    candidate = _comparison_key(stem)
    for alias in aliases:
        key = _comparison_key(alias)
        if not key:
            continue
        if candidate == key or candidate.startswith(key) or key.startswith(candidate):
            return True
    return False


def discover_city_index(
    city: str,
    index_roots: Sequence[str | Path],
    metadata_aliases: Sequence[str] = (),
    configured_aliases: Sequence[str] = (),
    preferred_stem: str | None = None,
) -> CityIndexDiscovery:
    """Conservatively resolve one city index and retain absence evidence."""
    aliases = _city_aliases(
        city,
        metadata_aliases,
        [*configured_aliases, *([preferred_stem] if preferred_stem else [])],
    )
    checked: list[dict[str, object]] = []
    candidates: list[Path] = []
    for raw_root in index_roots:
        root = Path(raw_root)
        files = sorted(root.glob("*.parquet")) if root.is_dir() else []
        checked.append(
            {
                "path": str(root),
                "exists": root.is_dir(),
                "parquet_count": len(files),
            }
        )
        if preferred_stem:
            candidates.extend(
                path
                for path in files
                if _comparison_key(path.stem) == _comparison_key(preferred_stem)
            )
        else:
            candidates.extend(path for path in files if _stem_matches_alias(path.stem, aliases))

    unique = tuple(sorted(set(candidates)))
    if not unique:
        return CityIndexDiscovery(city, "absent", None, unique, checked, aliases)
    if len(unique) > 1:
        return CityIndexDiscovery(city, "ambiguous", None, unique, checked, aliases)
    return CityIndexDiscovery(city, "resolved", unique[0], unique, checked, aliases)


def _read_names(path: Path) -> tuple[pd.DataFrame, set[str], set[str]]:
    frame = pd.read_parquet(path)
    if "path" not in frame.columns:
        raise ValueError(f"{path} must contain a path column")
    if "name" not in frame.columns:
        frame = frame.copy()
        frame["name"] = frame["path"].map(lambda value: Path(str(value)).name)
    names = set(frame["name"].dropna().astype(str))
    existing = {
        str(row.name)
        for row in frame[["name", "path"]].itertuples(index=False)
        if Path(str(row.path)).is_file()
    }
    return frame, names, existing


def _embedding_inventory(folder: Path) -> tuple[list[Path], pd.DataFrame, set[str]]:
    shards = sorted(folder.glob("*.parquet")) if folder.is_dir() else []
    if not shards:
        return [], pd.DataFrame(), set()
    frames = [pd.read_parquet(path) for path in shards]
    frame = pd.concat(frames, ignore_index=True)
    if "name" not in frame:
        raise ValueError(f"embedding shards in {folder} must contain name")
    names = set(frame["name"].dropna().astype(str))
    return shards, frame, names


def _discover_artifact_stems(
    city: str, paths: RecoveryPaths, preferred_stem: str | None = None
) -> tuple[str, ...]:
    aliases = _city_aliases(city, configured_aliases=[preferred_stem] if preferred_stem else [])
    candidates: set[str] = set()
    roots = [paths.embed_root, paths.root / "GSV" / "gsv_rgb"]
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            matches = (
                _comparison_key(entry.name) == _comparison_key(preferred_stem)
                if preferred_stem
                else _stem_matches_alias(entry.name, aliases)
            )
            if entry.is_dir() and matches:
                candidates.add(entry.name)
    return tuple(sorted(candidates))


def _find_city_parquet(root: Path | None, city: str, stem: str) -> Path | None:
    if root is None or not root.is_dir():
        return None
    direct = [
        root / f"{stem}.parquet",
        root / f"{city}.parquet",
        root / f"dinov3_city={city}_res_exclude=None.parquet",
    ]
    for path in direct:
        if path.exists():
            return path
    matches = [
        path
        for path in root.glob("*.parquet")
        if _stem_matches_alias(path.stem, [city, stem])
    ]
    return matches[0] if len(matches) == 1 else None


def _load_h3_set(path: Path | None, resolution: int) -> set[str]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_parquet(path)
    column = next((name for name in ("hex_id", "h3", "h3_id") if name in frame), None)
    if column is None:
        raise ValueError(f"{path} must contain a hex_id, h3, or h3_id column")
    if "res" in frame:
        frame = frame[pd.to_numeric(frame["res"], errors="coerce").eq(resolution)]
    return set(frame[column].dropna().astype(str))


def _to_h3(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)


def _current_membership(
    root: Path,
    stem: str,
    source_names: set[str],
    embedded_names: set[str],
    resolution: int,
) -> dict[str, object]:
    metadata = root / "GSV" / "gsv_rgb" / stem / "gsvmeta" / "gsv_pano.csv"
    if not metadata.exists():
        return {
            "pano_metadata_path": str(metadata),
            "pano_metadata_exists": False,
            "pano_metadata_count": 0,
            "source_panoid_join_count": 0,
            "embedded_panoid_join_count": 0,
            "source_res7_h3": set(),
            "embedded_res7_h3": set(),
        }
    frame = pd.read_csv(metadata)
    required = {"panoid", "lat", "lon"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{metadata} must contain {sorted(required)}")
    frame = frame.dropna(subset=["panoid", "lat", "lon"]).copy()
    frame["panoid"] = frame["panoid"].astype(str)
    source_panoids = {name[:22] for name in source_names}
    embedded_panoids = {name[:22] for name in embedded_names}
    source = frame[frame["panoid"].isin(source_panoids)].copy()
    embedded = frame[frame["panoid"].isin(embedded_panoids)].copy()

    def cells(rows: pd.DataFrame) -> set[str]:
        return {
            _to_h3(float(row.lat), float(row.lon), resolution)
            for row in rows.itertuples(index=False)
        }

    return {
        "pano_metadata_path": str(metadata),
        "pano_metadata_exists": True,
        "pano_metadata_count": int(len(frame)),
        "source_panoid_join_count": int(len(source)),
        "embedded_panoid_join_count": int(len(embedded)),
        "source_res7_h3": cells(source),
        "embedded_res7_h3": cells(embedded),
    }


def _count_pairwise_shards(root: Path, city: str, resolution: int) -> int:
    temp = root / "optimized" / "temp"
    if not temp.exists():
        return 0
    paths = set(temp.glob(f"city1={city}/city2=*/part_res={resolution}.parquet"))
    paths.update(temp.glob(f"city1=*/city2={city}/part_res={resolution}.parquet"))
    return sum(path.is_file() and path.stat().st_size > 0 for path in paths)


def _count_aggregate_rows(root: Path, city: str) -> int:
    if not root.exists():
        return 0
    count = 0
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["city_1", "city_2"])
        count += int(frame["city_1"].eq(city).sum() + frame["city_2"].eq(city).sum())
    return count


def audit_city(
    city: str,
    paths: RecoveryPaths,
    metadata_aliases: Sequence[str] = (),
    configured_aliases: Sequence[str] = (),
    preferred_stem: str | None = None,
) -> dict[str, object]:
    """Trace one city through every recovery boundary."""
    discovery = discover_city_index(
        city,
        paths.index_roots,
        metadata_aliases,
        configured_aliases,
        preferred_stem,
    )
    report: dict[str, object] = {
        "city": city,
        "resolution": paths.resolution,
        "source_index": discovery.as_dict(),
    }
    candidate_stems = tuple(
        dict.fromkeys(resolve_city_file_stem(alias) for alias in discovery.aliases)
    )
    checked_city_paths: list[dict[str, object]] = []
    for candidate_stem in candidate_stems:
        for index_root in paths.index_roots:
            candidate = Path(index_root) / f"{candidate_stem}.parquet"
            checked_city_paths.append(
                {"kind": "image_index", "path": str(candidate), "exists": candidate.exists()}
            )
        for kind, candidate in (
            ("embedding_directory", paths.embed_root / candidate_stem),
            ("gsv_city_directory", paths.root / "GSV" / "gsv_rgb" / candidate_stem),
        ):
            checked_city_paths.append(
                {"kind": kind, "path": str(candidate), "exists": candidate.exists()}
            )
    report["source_index"]["checked_city_paths"] = checked_city_paths
    artifact_stems = _discover_artifact_stems(city, paths, preferred_stem)
    report["source_index"]["artifact_stem_candidates"] = list(artifact_stems)
    if discovery.status == "resolved" and discovery.path is not None:
        index_stem = discovery.path.stem
        if preferred_stem:
            stem = preferred_stem
        elif not artifact_stems or index_stem in artifact_stems:
            stem = index_stem
        elif len(artifact_stems) == 1:
            stem = artifact_stems[0]
        else:
            report["source_index"].update(
                {
                    "index_stem": index_stem,
                    "row_count": 0,
                    "image_name_count": 0,
                    "existing_image_count": 0,
                }
            )
            report["first_broken_boundary"] = "source_image_index"
            report["recoverability"] = "unresolved_alias"
            return report
        source_frame, source_names, existing_names = _read_names(discovery.path)
    elif discovery.status == "absent" and len(artifact_stems) == 1:
        stem = artifact_stems[0]
        index_stem = stem
        source_frame, source_names, existing_names = pd.DataFrame(), set(), set()
    else:
        report["source_index"].update(
            {"row_count": 0, "image_name_count": 0, "existing_image_count": 0}
        )
        report["first_broken_boundary"] = "source_image_index"
        report["recoverability"] = (
            "source_imagery_absent" if not artifact_stems else "unresolved_alias"
        )
        return report

    report["source_index"].update(
        {
            "selected_stem": stem,
            "index_stem": index_stem,
            "row_count": int(len(source_frame)),
            "image_name_count": len(source_names),
            "existing_image_count": len(existing_names),
            "missing_image_count": len(source_names - existing_names),
        }
    )
    gsv_city_root = paths.root / "GSV" / "gsv_rgb" / stem
    gsv_images = (
        [
            path
            for path in gsv_city_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        ]
        if gsv_city_root.is_dir()
        else []
    )
    report["source_index"].update(
        {
            "gsv_city_root": str(gsv_city_root),
            "gsv_city_root_exists": gsv_city_root.is_dir(),
            "gsv_image_file_count": len(gsv_images),
            "gsv_image_examples": [str(path) for path in gsv_images[:5]],
        }
    )
    shards, embedding_frame, embedded_names = _embedding_inventory(paths.embed_root / stem)
    report["embeddings"] = {
        "path": str(paths.embed_root / stem),
        "shard_count": len(shards),
        "unique_name_count": len(embedded_names),
        "finished_expected_count": len(existing_names & embedded_names),
        "missing_count": len(existing_names - embedded_names),
        "extra_finished_count": len(embedded_names - existing_names),
    }

    required_path = _find_city_parquet(paths.required_h3_root, city, stem)
    core_path = _find_city_parquet(paths.core_h3_root, city, stem)
    required_h3 = _load_h3_set(required_path, paths.resolution)
    core_h3 = _load_h3_set(core_path, paths.resolution)
    current = _current_membership(
        paths.root, stem, existing_names, embedded_names, paths.resolution
    )
    source_h3 = current.pop("source_res7_h3")
    embedded_h3 = current.pop("embedded_res7_h3")
    report["current_membership"] = {
        **current,
        "source_res7_h3_count": len(source_h3),
        "embedded_res7_h3_count": len(embedded_h3),
        "required_h3_path": str(required_path) if required_path else None,
        "required_h3_count": len(required_h3),
        "required_overlap_count": len(embedded_h3 & required_h3),
        "core_h3_path": str(core_path) if core_path else None,
        "core_h3_count": len(core_h3),
        "core_source_overlap_count": len(source_h3 & core_h3),
        "core_overlap_count": len(embedded_h3 & core_h3),
        "core_without_source_image_count": len(core_h3 - source_h3),
        "core_with_source_without_embedding_count": len((core_h3 & source_h3) - embedded_h3),
    }

    summary_path = _find_city_parquet(paths.h3_root, city, stem)
    summary_h3 = _load_h3_set(summary_path, paths.resolution)
    report["h3_summary"] = {
        "path": str(summary_path) if summary_path else None,
        "res7_h3_count": len(summary_h3),
        "required_overlap_count": len(summary_h3 & required_h3),
        "core_overlap_count": len(summary_h3 & core_h3),
        "current_mapping_overlap_count": len(summary_h3 & embedded_h3),
    }
    report["pairwise"] = {
        "root": str(paths.pairwise_root),
        "shard_count": _count_pairwise_shards(paths.pairwise_root, city, paths.resolution),
    }
    report["aggregate"] = {
        "root": str(paths.aggregate_root),
        "endpoint_row_count": _count_aggregate_rows(paths.aggregate_root, city),
    }

    if discovery.status == "absent":
        boundary = "source_image_index"
        if gsv_images:
            recoverability = "source_index_rebuildable"
        elif embedded_names and report["current_membership"]["embedded_panoid_join_count"]:
            recoverability = "recoverable_from_existing_embeddings"
        else:
            recoverability = "source_imagery_absent"
    elif not existing_names:
        boundary = "image_files"
        recoverability = "source_imagery_absent"
    elif report["embeddings"]["missing_count"]:
        boundary = "dino_embeddings"
        recoverability = "recoverable"
    elif not embedded_h3:
        boundary = "image_to_h3_membership"
        recoverability = "recoverable"
    elif not summary_h3 or (core_h3 and not summary_h3.intersection(core_h3)):
        boundary = "res7_h3_vector_summary"
        recoverability = "recoverable"
    elif report["pairwise"]["shard_count"] == 0:
        boundary = "pairwise_shards"
        recoverability = "recoverable"
    elif report["aggregate"]["endpoint_row_count"] == 0:
        boundary = "aggregated_city_pair_output"
        recoverability = "recoverable"
    else:
        boundary = "none"
        recoverability = "complete"
    report["first_broken_boundary"] = boundary
    report["recoverability"] = recoverability
    return report


def _usable_h3(path: Path, resolution: int) -> bool:
    if not path.exists():
        return False
    frame = pd.read_parquet(path, columns=["res"])
    return bool(pd.to_numeric(frame["res"], errors="coerce").eq(resolution).any())


def build_affected_pair_manifest(
    cities: Sequence[str],
    affected_cities: Sequence[str],
    h3_root: str | Path,
    output_path: str | Path,
    resolution: int = 7,
    input_template: str = "dinov3_city={city}_res_exclude=None.parquet",
) -> list[tuple[str, str]]:
    root = Path(h3_root)
    affected = set(affected_cities)
    usable = [
        city
        for city in sorted(set(cities))
        if _usable_h3(root / input_template.format(city=city), resolution)
    ]
    pairs = [pair for pair in combinations(usable, 2) if affected.intersection(pair)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{city1}|{city2}\n" for city1, city2 in pairs))
    return pairs


def write_recovery_manifests(
    audit_path: str | Path,
    embed_manifest: str | Path,
    h3_manifest: str | Path,
    absent_cities_path: str | Path,
) -> dict[str, int]:
    """Split audited cities into embedding, H3-rebuild, and proven-absent lists."""
    payload = json.loads(Path(audit_path).read_text())
    embed_rows: list[str] = []
    h3_rows: list[str] = []
    absent: list[str] = []
    unresolved: list[str] = []
    for report in payload.get("cities", []):
        city = str(report["city"])
        source = report.get("source_index", {})
        stem = source.get("selected_stem")
        recoverability = report.get("recoverability")
        if recoverability in {"unresolved_alias", "source_index_rebuildable"}:
            unresolved.append(city)
            continue
        if recoverability == "source_imagery_absent":
            absent.append(city)
            continue
        if not stem:
            unresolved.append(city)
            continue
        if source.get("status") == "resolved" and int(source.get("existing_image_count", 0)):
            index_path = source.get("path")
            if not index_path:
                raise ValueError(f"resolved source index has no path for {city}")
            index_stem = source.get("index_stem") or Path(index_path).stem
            embed_rows.append(
                f"{city}|{index_stem}|{Path(index_path).parent}|{stem}"
            )
        embeddings = report.get("embeddings", {})
        membership = report.get("current_membership", {})
        if int(embeddings.get("unique_name_count", 0)) and int(
            membership.get("embedded_panoid_join_count", 0)
        ):
            h3_rows.append(f"{city}|{stem}")
    if unresolved:
        raise ValueError(f"unresolved city aliases require an explicit override: {unresolved}")

    outputs = [
        (Path(embed_manifest), embed_rows),
        (Path(h3_manifest), h3_rows),
        (Path(absent_cities_path), absent),
    ]
    for path, rows in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{row}\n" for row in rows))
    return {"embed": len(embed_rows), "h3": len(h3_rows), "absent": len(absent)}


def recover_missing_indices(
    audit_path: str | Path,
    root: str | Path,
    output_root: str | Path,
    allow_gsv_rebuild: bool = False,
) -> dict[str, int]:
    """Rebuild missing city indices only from image files proven to exist on disk."""
    payload = json.loads(Path(audit_path).read_text())
    root = Path(root)
    output_root = Path(output_root)
    recovered: dict[str, int] = {}
    rebuildable = [
        str(report["city"])
        for report in payload.get("cities", [])
        if report.get("source_index", {}).get("status") == "absent"
        and int(report.get("source_index", {}).get("gsv_image_file_count", 0)) > 0
    ]
    if rebuildable and not allow_gsv_rebuild:
        raise ValueError(
            "GSV index reconstruction requires explicit opt-in because raw GSV files "
            f"may differ from validation imagery: {rebuildable}"
        )
    for report in payload.get("cities", []):
        source = report.get("source_index", {})
        if source.get("status") != "absent":
            continue
        stem = source.get("selected_stem")
        if not stem or int(source.get("gsv_image_file_count", 0)) == 0:
            continue
        city_root = root / "GSV" / "gsv_rgb" / str(stem)
        images = sorted(
            path.resolve()
            for path in city_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / f"{stem}.parquet"
        frame = pd.DataFrame(
            {"path": [str(path) for path in images], "name": [path.name for path in images]}
        )
        temporary = output.with_name(f".{output.name}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
        recovered[str(report["city"])] = len(frame)
    return recovered


def _safe_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"refusing to replace non-symlink overlay path: {destination}")
    destination.symlink_to(source)


def build_h3_overlay(
    original_root: str | Path,
    recovered_root: str | Path,
    overlay_root: str | Path,
    affected_cities: Sequence[str],
    resolution: int = 7,
) -> dict[str, int]:
    """Link unaffected originals and recovered affected H3 summaries into one root."""
    original = Path(original_root)
    recovered = Path(recovered_root)
    overlay = Path(overlay_root)
    affected_norm = {normalize_city_name(city) for city in affected_cities}
    selected: dict[str, tuple[str, Path]] = {}
    for source_label, root in (("original", original), ("recovered", recovered)):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("dinov3_city=*_res_exclude=None.parquet")):
            city = path.name[len("dinov3_city=") : -len("_res_exclude=None.parquet")]
            is_affected = normalize_city_name(city) in affected_norm
            if source_label == "original" and is_affected:
                continue
            if source_label == "recovered" and not is_affected:
                continue
            if _usable_h3(path, resolution):
                selected[city] = (source_label, path)
    counts = {"original": 0, "recovered": 0}
    for _city, (label, source) in sorted(selected.items()):
        _safe_symlink(source, overlay / source.name)
        counts[label] += 1
    return counts


def _pair_key(path: Path, resolution: int) -> tuple[str, str] | None:
    if path.name != f"part_res={resolution}.parquet":
        return None
    city2_part = path.parent.name
    city1_part = path.parent.parent.name
    if not city1_part.startswith("city1=") or not city2_part.startswith("city2="):
        return None
    return city1_part[6:], city2_part[6:]


def _valid_pairwise_shard(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    frame = pd.read_parquet(path)
    required = {"hex_id1", "hex_id2", "similarity"}
    return not frame.empty and required.issubset(frame.columns)


def build_pairwise_overlay(
    original_root: str | Path,
    recovered_root: str | Path,
    overlay_root: str | Path,
    affected_cities: Sequence[str],
    resolution: int = 7,
) -> dict[str, int]:
    """Link unaffected original shards and validated recovered affected shards."""
    affected_norm = {normalize_city_name(city) for city in affected_cities}
    selected: dict[tuple[str, str], tuple[str, Path]] = {}
    roots = (("original", Path(original_root)), ("recovered", Path(recovered_root)))
    for label, root in roots:
        temp = root / "optimized" / "temp"
        if not temp.is_dir():
            continue
        for path in sorted(temp.glob(f"city1=*/city2=*/part_res={resolution}.parquet")):
            key = _pair_key(path, resolution)
            if key is None or not _valid_pairwise_shard(path):
                continue
            is_affected = bool(affected_norm.intersection(map(normalize_city_name, key)))
            if label == "original" and is_affected:
                continue
            if label == "recovered" and not is_affected:
                continue
            selected[key] = (label, path)
    counts = {"original": 0, "recovered": 0}
    overlay = Path(overlay_root) / "optimized" / "temp"
    for (city1, city2), (label, source) in sorted(selected.items()):
        destination = overlay / f"city1={city1}" / f"city2={city2}" / source.name
        _safe_symlink(source, destination)
        counts[label] += 1
    return counts


def validate_h3_recovery(
    audit: dict[str, object] | str | Path,
    allowed_missing_cities: Sequence[str] = (),
) -> dict[str, object]:
    """Require every recoverable city to have source-backed current H3 output."""
    payload = (
        json.loads(Path(audit).read_text())
        if isinstance(audit, (str, Path))
        else audit
    )
    allowed = {normalize_city_name(city) for city in allowed_missing_cities}
    result: dict[str, object] = {"status": "valid", "cities": {}}
    errors: list[str] = []
    for report in payload.get("cities", []):
        city = str(report["city"])
        if normalize_city_name(city) in allowed:
            continue
        current = report.get("current_membership", {})
        summary = report.get("h3_summary", {})
        res7_count = int(summary.get("res7_h3_count", 0))
        mapping_overlap = int(summary.get("current_mapping_overlap_count", 0))
        core_source = int(current.get("core_source_overlap_count", 0))
        core_overlap = int(summary.get("core_overlap_count", 0))
        if res7_count == 0:
            errors.append(f"{city}: no recovered res=7 H3 vectors")
        if int(current.get("embedded_res7_h3_count", 0)) and mapping_overlap == 0:
            errors.append(f"{city}: recovered summary does not overlap current embedded mapping")
        if core_source and core_overlap == 0:
            errors.append(f"{city}: source-backed core H3 cells have no recovered vectors")
        result["cities"][city] = {
            "res7_h3_count": res7_count,
            "current_mapping_overlap_count": mapping_overlap,
            "source_backed_core_h3_count": core_source,
            "summary_core_overlap_count": core_overlap,
        }
    if errors:
        raise ValueError("H3 recovery validation failed: " + "; ".join(errors))
    return result


def validate_pair_manifest_shards(
    manifest_path: str | Path,
    pairwise_root: str | Path,
    resolution: int = 7,
) -> dict[str, int]:
    """Require one readable nonempty pairwise shard for every manifest row."""
    pairs: list[tuple[str, str]] = []
    for line in Path(manifest_path).read_text().splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) != 2 or not all(values):
            raise ValueError(f"invalid pair manifest row: {line}")
        pairs.append((values[0], values[1]))
    root = Path(pairwise_root)
    invalid: list[str] = []
    for city1, city2 in pairs:
        path = (
            root
            / "optimized"
            / "temp"
            / f"city1={city1}"
            / f"city2={city2}"
            / f"part_res={resolution}.parquet"
        )
        if not _valid_pairwise_shard(path):
            invalid.append(f"{city1}|{city2}: {path}")
    if invalid:
        raise ValueError("missing or invalid pairwise shards: " + "; ".join(invalid[:10]))
    return {"expected_pair_count": len(pairs), "valid_shard_count": len(pairs)}


def _export_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.glob("*.parquet") if path.is_file() or path.is_dir())


def _membership_by_city(
    membership_root: Path, cities: Iterable[str], resolution: int
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for city in cities:
        path = _find_city_parquet(membership_root, city, resolve_city_file_stem(city))
        result[city] = _load_h3_set(path, resolution)
    return result


def validate_final_export(
    export_root: str | Path,
    membership_root: str | Path,
    required_cities: Sequence[str],
    allowed_missing_cities: Sequence[str] = (),
    resolution: int = 7,
    duckdb_temp_dir: str | Path | None = None,
    duckdb_memory_limit: str | None = None,
) -> dict[str, object]:
    """Validate schema, values, orientation, uniqueness, and city-pair coverage."""
    paths = _export_paths(Path(export_root))
    if not paths:
        raise ValueError("no parquet export files found")
    files: list[str] = []
    for path in paths:
        if path.is_dir():
            files.extend(str(part) for part in sorted(path.rglob("*.parquet")))
        else:
            files.append(str(path))
    if not files:
        raise ValueError("no parquet parts found in export datasets")

    connection = duckdb.connect()
    try:
        if duckdb_memory_limit:
            escaped_limit = str(duckdb_memory_limit).replace("'", "''")
            connection.execute(f"SET memory_limit = '{escaped_limit}'")
        if duckdb_temp_dir:
            temp_dir = Path(duckdb_temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            escaped_temp = str(temp_dir).replace("'", "''")
            connection.execute(f"SET temp_directory = '{escaped_temp}'")
        relation = connection.read_parquet(files, union_by_name=False)
        if relation.columns != FINAL_COLUMNS:
            raise ValueError(
                f"schema mismatch: expected exactly {FINAL_COLUMNS}, got {relation.columns}"
            )
        relation.create_view("export_data")
        row_count = int(connection.execute("SELECT COUNT(*) FROM export_data").fetchone()[0])
        if row_count == 0:
            raise ValueError("final export contains zero rows")
        null_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM export_data
                WHERE hex_id1 IS NULL OR hex_id2 IS NULL OR similarity IS NULL
                   OR city_1 IS NULL OR city_2 IS NULL
                """
            ).fetchone()[0]
        )
        if null_count:
            raise ValueError("null city, H3, or similarity fields found")
        invalid_numeric = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM export_data
                WHERE TRY_CAST(similarity AS DOUBLE) IS NULL
                   OR NOT isfinite(TRY_CAST(similarity AS DOUBLE))
                """
            ).fetchone()[0]
        )
        if invalid_numeric:
            raise ValueError("non-finite similarity values found")
        if connection.execute(
            "SELECT COUNT(*) FROM export_data WHERE TRY_CAST(similarity AS DOUBLE) = 0.0"
        ).fetchone()[0]:
            raise ValueError("exact-zero similarity values are forbidden as missing sentinels")
        if connection.execute(
            """
            SELECT COUNT(*) FROM export_data
            WHERE TRY_CAST(similarity AS DOUBLE) < -1.000001
               OR TRY_CAST(similarity AS DOUBLE) > 1.000001
            """
        ).fetchone()[0]:
            raise ValueError("similarity range is outside plausible cosine bounds")

        export_cities = {
            str(row[0])
            for row in connection.execute(
                "SELECT city_1 FROM export_data UNION SELECT city_2 FROM export_data"
            ).fetchall()
        }
        all_cities = sorted(set(required_cities) | export_cities)
        membership = _membership_by_city(Path(membership_root), all_cities, resolution)
        membership_rows = [
            {"city_norm": normalize_city_name(city), "hex_id": hex_id}
            for city, cells in membership.items()
            for hex_id in cells
        ]
        membership_frame = pd.DataFrame(
            membership_rows, columns=["city_norm", "hex_id"]
        ).drop_duplicates()
        aliases = pd.DataFrame(
            {
                "raw_city": all_cities,
                "city_norm": [normalize_city_name(city) for city in all_cities],
            }
        )
        connection.register("h3_membership", membership_frame)
        connection.register("city_aliases", aliases)
        connection.execute(
            """
            CREATE TEMP VIEW normalized_export AS
            SELECT e.*, a1.city_norm AS city1_norm, a2.city_norm AS city2_norm
            FROM export_data e
            LEFT JOIN city_aliases a1 ON e.city_1 = a1.raw_city
            LEFT JOIN city_aliases a2 ON e.city_2 = a2.raw_city
            """
        )
        violations = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM normalized_export e
                WHERE NOT EXISTS (
                    SELECT 1 FROM h3_membership m
                    WHERE m.city_norm = e.city1_norm AND m.hex_id = e.hex_id1
                ) OR NOT EXISTS (
                    SELECT 1 FROM h3_membership m
                    WHERE m.city_norm = e.city2_norm AND m.hex_id = e.hex_id2
                )
                """
            ).fetchone()[0]
        )
        if violations:
            raise ValueError(f"membership orientation violations found in {violations} rows")

        duplicates = int(
            connection.execute(
                """
                WITH canonical AS (
                    SELECT
                        CASE WHEN city1_norm < city2_norm
                               OR (city1_norm = city2_norm AND hex_id1 <= hex_id2)
                             THEN city1_norm ELSE city2_norm END AS city_lo,
                        CASE WHEN city1_norm < city2_norm
                               OR (city1_norm = city2_norm AND hex_id1 <= hex_id2)
                             THEN hex_id1 ELSE hex_id2 END AS hex_lo,
                        CASE WHEN city1_norm < city2_norm
                               OR (city1_norm = city2_norm AND hex_id1 <= hex_id2)
                             THEN city2_norm ELSE city1_norm END AS city_hi,
                        CASE WHEN city1_norm < city2_norm
                               OR (city1_norm = city2_norm AND hex_id1 <= hex_id2)
                             THEN hex_id2 ELSE hex_id1 END AS hex_hi
                    FROM normalized_export
                )
                SELECT COUNT(*) FROM (
                    SELECT city_lo, hex_lo, city_hi, hex_hi
                    FROM canonical GROUP BY ALL HAVING COUNT(*) > 1
                ) duplicates
                """
            ).fetchone()[0]
        )
        if duplicates:
            raise ValueError("duplicate canonical city/H3 endpoint rows found")

        city_h3_counts = {
            str(city): int(count)
            for city, count in connection.execute(
                """
                SELECT city, COUNT(DISTINCT hex_id)
                FROM (
                    SELECT city_1 AS city, hex_id1 AS hex_id FROM export_data
                    UNION ALL
                    SELECT city_2 AS city, hex_id2 AS hex_id FROM export_data
                ) endpoints
                GROUP BY city ORDER BY city
                """
            ).fetchall()
        }
        observed_pairs = {
            tuple(sorted((str(city1), str(city2))))
            for city1, city2 in connection.execute(
                "SELECT DISTINCT city1_norm, city2_norm FROM normalized_export"
            ).fetchall()
        }
        present_norms = {normalize_city_name(city) for city in city_h3_counts}
        missing_cities = sorted(
            city for city in required_cities if normalize_city_name(city) not in present_norms
        )
        allowed_norms = {normalize_city_name(city) for city in allowed_missing_cities}
        unexpected_missing = [
            city for city in missing_cities if normalize_city_name(city) not in allowed_norms
        ]
        if unexpected_missing:
            raise ValueError(f"required cities missing from export: {unexpected_missing}")
        recoverable = [
            city for city in required_cities if normalize_city_name(city) not in allowed_norms
        ]
        required_pairs = {
            tuple(sorted((normalize_city_name(a), normalize_city_name(b))))
            for a, b in combinations(recoverable, 2)
        }
        missing_pairs = sorted(required_pairs - observed_pairs)
        if missing_pairs:
            raise ValueError(f"missing required city pairs: {missing_pairs[:5]}")
        similarity_min, similarity_max = connection.execute(
            "SELECT MIN(similarity), MAX(similarity) FROM export_data"
        ).fetchone()
        return {
            "status": "valid",
            "resolution": resolution,
            "files": [str(path) for path in paths],
            "row_count": row_count,
            "similarity_min": float(similarity_min),
            "similarity_max": float(similarity_max),
            "city_h3_counts": city_h3_counts,
            "pair_count": len(observed_pairs),
            "missing_cities": missing_cities,
            "allowed_missing_cities": sorted(allowed_missing_cities),
            "missing_required_pairs": missing_pairs,
        }
    finally:
        connection.close()


def _write_report(report: object, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))


def _write_audit_csv(reports: Sequence[dict[str, object]], output: str | Path) -> None:
    rows = []
    for report in reports:
        source = report.get("source_index", {})
        embeddings = report.get("embeddings", {})
        current = report.get("current_membership", {})
        summary = report.get("h3_summary", {})
        pairwise = report.get("pairwise", {})
        aggregate = report.get("aggregate", {})
        rows.append(
            {
                "city": report.get("city"),
                "first_broken_boundary": report.get("first_broken_boundary"),
                "recoverability": report.get("recoverability"),
                "index_status": source.get("status"),
                "selected_stem": source.get("selected_stem"),
                "index_rows": source.get("row_count", 0),
                "existing_images": source.get("existing_image_count", 0),
                "gsv_images": source.get("gsv_image_file_count", 0),
                "embedding_shards": embeddings.get("shard_count", 0),
                "finished_expected": embeddings.get("finished_expected_count", 0),
                "missing_embeddings": embeddings.get("missing_count", 0),
                "extra_embeddings": embeddings.get("extra_finished_count", 0),
                "current_res7_h3": current.get("embedded_res7_h3_count", 0),
                "current_core_overlap": current.get("core_overlap_count", 0),
                "summary_res7_h3": summary.get("res7_h3_count", 0),
                "summary_core_overlap": summary.get("core_overlap_count", 0),
                "pairwise_shards": pairwise.get("shard_count", 0),
                "aggregate_endpoint_rows": aggregate.get("endpoint_row_count", 0),
            }
        )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _parse_city_stems(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"city stem override must be CITY=STEM: {value}")
        city, stem = (part.strip() for part in value.split("=", 1))
        if not city or not stem:
            raise ValueError(f"city stem override must be CITY=STEM: {value}")
        result[normalize_city_name(city)] = stem
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Trace affected cities across all boundaries")
    audit.add_argument("--city", action="append", dest="cities")
    audit.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    audit.add_argument("--index-root", action="append", type=Path, required=True)
    audit.add_argument("--embed-root", type=Path, required=True)
    audit.add_argument("--h3-root", type=Path, required=True)
    audit.add_argument("--pairwise-root", type=Path, required=True)
    audit.add_argument("--aggregate-root", type=Path, required=True)
    audit.add_argument("--required-h3-root", type=Path)
    audit.add_argument("--core-h3-root", type=Path)
    audit.add_argument("--resolution", type=int, default=7)
    audit.add_argument("--city-stem", action="append", default=[])
    audit.add_argument("--output-json", type=Path, required=True)
    audit.add_argument("--output-csv", type=Path)

    manifest = subparsers.add_parser("manifest", help="Build affected-only pair manifest")
    manifest.add_argument("--city-meta", type=Path, required=True)
    manifest.add_argument("--affected-city", action="append", dest="affected")
    manifest.add_argument("--h3-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--resolution", type=int, default=7)

    manifests = subparsers.add_parser(
        "recovery-manifests", help="Split an audit into embedding, H3, and absent lists"
    )
    manifests.add_argument("--audit-json", type=Path, required=True)
    manifests.add_argument("--embed-manifest", type=Path, required=True)
    manifests.add_argument("--h3-manifest", type=Path, required=True)
    manifests.add_argument("--absent-cities", type=Path, required=True)

    recover = subparsers.add_parser(
        "recover-indices", help="Rebuild absent indices from real GSV image files"
    )
    recover.add_argument("--audit-json", type=Path, required=True)
    recover.add_argument("--root", type=Path, required=True)
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--output-json", type=Path, required=True)
    recover.add_argument(
        "--allow-gsv-rebuild",
        action="store_true",
        help="Explicitly allow raw GSV image files to become a recovered embedding index",
    )

    overlays = subparsers.add_parser(
        "build-overlays", help="Overlay recovered affected artifacts on untouched originals"
    )
    overlays.add_argument("--original-h3-root", type=Path, required=True)
    overlays.add_argument("--recovered-h3-root", type=Path, required=True)
    overlays.add_argument("--h3-overlay-root", type=Path, required=True)
    overlays.add_argument("--original-pairwise-root", type=Path, required=True)
    overlays.add_argument("--recovered-pairwise-root", type=Path, required=True)
    overlays.add_argument("--pairwise-overlay-root", type=Path, required=True)
    overlays.add_argument("--affected-city", action="append", dest="affected")
    overlays.add_argument("--resolution", type=int, default=7)
    overlays.add_argument("--output-json", type=Path, required=True)

    check_h3 = subparsers.add_parser("check-h3", help="Gate recovered H3 summaries")
    check_h3.add_argument("--audit-json", type=Path, required=True)
    check_h3.add_argument("--allowed-missing-city", action="append", default=[])
    check_h3.add_argument("--output-json", type=Path, required=True)

    check_pairs = subparsers.add_parser(
        "check-pairs", help="Gate all expected affected pairwise shards"
    )
    check_pairs.add_argument("--manifest", type=Path, required=True)
    check_pairs.add_argument("--pairwise-root", type=Path, required=True)
    check_pairs.add_argument("--resolution", type=int, default=7)
    check_pairs.add_argument("--output-json", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate final B5c parquet output")
    validate.add_argument("--export-root", type=Path, required=True)
    validate.add_argument("--membership-root", type=Path, required=True)
    validate.add_argument("--required-city", action="append", dest="required")
    validate.add_argument("--allowed-missing-city", action="append", default=[])
    validate.add_argument("--resolution", type=int, default=7)
    validate.add_argument("--duckdb-temp-dir", type=Path)
    validate.add_argument("--duckdb-memory-limit", default=None)
    validate.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        paths = RecoveryPaths(
            root=args.root,
            index_roots=tuple(args.index_root),
            embed_root=args.embed_root,
            h3_root=args.h3_root,
            pairwise_root=args.pairwise_root,
            aggregate_root=args.aggregate_root,
            required_h3_root=args.required_h3_root,
            core_h3_root=args.core_h3_root,
            resolution=args.resolution,
        )
        overrides = _parse_city_stems(args.city_stem)
        reports = [
            audit_city(
                city,
                paths,
                preferred_stem=overrides.get(normalize_city_name(city)),
            )
            for city in (args.cities or DEFAULT_AFFECTED_CITIES)
        ]
        _write_report({"cities": reports}, args.output_json)
        if args.output_csv:
            _write_audit_csv(reports, args.output_csv)
    elif args.command == "manifest":
        meta = pd.read_csv(args.city_meta)
        if "City" not in meta:
            raise ValueError(f"{args.city_meta} must contain City")
        build_affected_pair_manifest(
            meta["City"].dropna().astype(str).tolist(),
            args.affected or DEFAULT_AFFECTED_CITIES,
            args.h3_root,
            args.output,
            args.resolution,
        )
    elif args.command == "recovery-manifests":
        counts = write_recovery_manifests(
            args.audit_json,
            args.embed_manifest,
            args.h3_manifest,
            args.absent_cities,
        )
        print(json.dumps(counts, sort_keys=True))
    elif args.command == "recover-indices":
        report = recover_missing_indices(
            args.audit_json,
            args.root,
            args.output_root,
            allow_gsv_rebuild=args.allow_gsv_rebuild,
        )
        _write_report(report, args.output_json)
    elif args.command == "build-overlays":
        affected = args.affected or DEFAULT_AFFECTED_CITIES
        report = {
            "h3": build_h3_overlay(
                args.original_h3_root,
                args.recovered_h3_root,
                args.h3_overlay_root,
                affected,
                args.resolution,
            ),
            "pairwise": build_pairwise_overlay(
                args.original_pairwise_root,
                args.recovered_pairwise_root,
                args.pairwise_overlay_root,
                affected,
                args.resolution,
            ),
        }
        _write_report(report, args.output_json)
    elif args.command == "check-h3":
        report = validate_h3_recovery(args.audit_json, args.allowed_missing_city)
        _write_report(report, args.output_json)
    elif args.command == "check-pairs":
        report = validate_pair_manifest_shards(
            args.manifest, args.pairwise_root, args.resolution
        )
        _write_report(report, args.output_json)
    else:
        report = validate_final_export(
            args.export_root,
            args.membership_root,
            args.required or DEFAULT_AFFECTED_CITIES,
            args.allowed_missing_city,
            args.resolution,
            args.duckdb_temp_dir,
            args.duckdb_memory_limit,
        )
        _write_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
