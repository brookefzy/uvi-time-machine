"""POI-tier contracts and deterministic balanced sampling for mode fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from dinov3_utils import normalize_city_name


STRATA = ("core", "suburban", "occupied_rural", "no_poi")
DEFAULT_STRATUM_WEIGHTS = {
    "core": 0.40,
    "suburban": 0.30,
    "occupied_rural": 0.20,
    "no_poi": 0.10,
}
REQUIRED_COLUMNS = {
    "city",
    "hex_id",
    "resolution",
    "landuse_tier",
    "poi_density",
    "poi_diversity",
    "poi_density_z",
    "poi_diversity_z",
    "urban_intensity",
}
TIER_METADATA_COLUMNS = [
    "landuse_tier",
    "training_stratum",
    "poi_density",
    "poi_diversity",
    "poi_density_z",
    "poi_diversity_z",
    "urban_intensity",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_training_config(
    landuse_tiers_path: Path,
    *,
    max_images_per_city: int,
    max_images_per_h3: int,
    stratum_weights: str | dict[str, float] | None,
    sampling_seed: int,
    requested_k_values: list[int],
) -> dict:
    k_values = [int(value) for value in requested_k_values]
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("requested K values must be positive")
    return {
        "training_sampling_strategy": "city_poi_stratified_v1",
        "landuse_tiers_sha256": file_sha256(landuse_tiers_path),
        "landuse_tiers_resolution": 8,
        "max_training_images_per_city": int(max_images_per_city),
        "max_training_images_per_h3": int(max_images_per_h3),
        "stratum_weights": parse_stratum_weights(stratum_weights),
        "training_sampling_seed": int(sampling_seed),
        "requested_k_values": k_values,
    }


def training_config_matches(existing: dict, expected: dict) -> bool:
    return all(existing.get(key) == value for key, value in expected.items())


def parse_stratum_weights(value: str | dict[str, float] | None) -> dict[str, float]:
    if value is None:
        weights = DEFAULT_STRATUM_WEIGHTS.copy()
    elif isinstance(value, str):
        try:
            weights = {
                key.strip(): float(number)
                for item in value.split(",")
                for key, number in [item.split("=", 1)]
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("stratum weights must use name=value comma syntax") from exc
    else:
        weights = {str(key): float(number) for key, number in value.items()}
    if set(weights) != set(STRATA):
        raise ValueError(f"stratum weights must define exactly: {', '.join(STRATA)}")
    if any(not np.isfinite(number) or number < 0 for number in weights.values()):
        raise ValueError("stratum weights must be finite and non-negative")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("stratum weights must sum to one")
    return {stratum: weights[stratum] for stratum in STRATA}


def load_landuse_tiers(path: Path, expected_resolution: int = 8) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"city": "string", "hex_id": "string"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"land-use tier table is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("land-use tier table must not be empty")
    if set(frame["resolution"].dropna().unique()) != {expected_resolution}:
        raise ValueError(f"land-use tiers must contain only resolution {expected_resolution}")
    unexpected = set(frame["landuse_tier"].dropna().unique()) - {
        "core",
        "suburban",
        "rural",
    }
    if unexpected or frame["landuse_tier"].isna().any():
        raise ValueError(f"unexpected land-use tiers: {sorted(unexpected)}")
    for column in (
        "poi_density",
        "poi_diversity",
        "poi_density_z",
        "poi_diversity_z",
        "urban_intensity",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        present = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all():
            raise ValueError(f"land-use tier column contains non-finite values: {column}")
    if frame[["city", "hex_id", "poi_density", "poi_diversity"]].isna().any().any():
        raise ValueError("land-use tier identity/density columns must not be null")
    if (frame["poi_density"] < 0).any():
        raise ValueError("poi_density must be non-negative")
    frame = frame.copy()
    frame["_city_key"] = frame["city"].map(normalize_city_name)
    if (frame["_city_key"] == "").any():
        raise ValueError("land-use tier city names must normalize to non-empty keys")
    if frame.duplicated(["_city_key", "hex_id"]).any():
        raise ValueError("land-use tier table contains duplicate canonical city-H3 keys")
    frame["training_stratum"] = frame["landuse_tier"]
    frame.loc[frame["landuse_tier"] == "rural", "training_stratum"] = "occupied_rural"
    no_poi = frame["poi_density"].eq(0) | frame["urban_intensity"].isna()
    frame.loc[no_poi, "training_stratum"] = "no_poi"
    frame.attrs["source_path"] = str(path)
    frame.attrs["source_sha256"] = file_sha256(path)
    return frame


def attach_landuse_strata(
    sampled: pd.DataFrame,
    tiers: pd.DataFrame,
    expected_resolution: int = 8,
) -> tuple[pd.DataFrame, dict]:
    missing = {"city", "hex_id"} - set(sampled.columns)
    if missing:
        raise ValueError(f"sampled data is missing columns: {sorted(missing)}")
    resolution_column = "res" if "res" in sampled else "resolution"
    if resolution_column not in sampled:
        raise ValueError("sampled data requires res or resolution")
    if set(sampled[resolution_column].dropna().unique()) != {expected_resolution}:
        raise ValueError(f"sampled data must contain only resolution {expected_resolution}")
    work = sampled.copy()
    work["_city_key"] = work["city"].map(normalize_city_name)
    tier_keys = tiers[["_city_key", "hex_id", *TIER_METADATA_COLUMNS]]
    joined = work.merge(
        tier_keys,
        on=["_city_key", "hex_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        preview = joined.loc[joined["_merge"].ne("both"), ["city", "hex_id"]].head()
        raise ValueError(
            f"{unmatched} sampled rows are unmatched in land-use tiers; "
            f"first keys: {preview.to_dict('records')}"
        )
    sampled_keys = work[["_city_key", "hex_id"]].drop_duplicates()
    unused_tier_keys = tiers[["_city_key", "hex_id"]].merge(
        sampled_keys, on=["_city_key", "hex_id"], how="left", indicator=True
    )
    joined = joined.drop(columns="_merge")
    audit = {
        "tier_source_rows": int(len(tiers)),
        "tier_source_city_count": int(tiers["_city_key"].nunique()),
        "tier_source_city_h3_count": int(len(tiers)),
        "matched_sampled_rows": int(len(joined)),
        "unmatched_sampled_rows": unmatched,
        "unused_tier_city_h3_count": int(unused_tier_keys["_merge"].eq("left_only").sum()),
        "stratum_counts": {
            key: int(value)
            for key, value in joined["training_stratum"].value_counts().items()
        },
    }
    return joined, audit


def _stable_hash(frame: pd.DataFrame, seed: int) -> pd.Series:
    hash_key = hashlib.sha256(str(seed).encode("ascii")).hexdigest()[:16]
    values = frame[["_city_key", "hex_id", "name"]].astype("string")
    return pd.util.hash_pandas_object(values, index=False, hash_key=hash_key)


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {key: total * weight for key, weight in weights.items()}
    result = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - result[key]), STRATA.index(key)))
    for key in order[:remaining]:
        result[key] += 1
    return result


def build_stratified_training_pool(
    frame: pd.DataFrame,
    *,
    max_images_per_city: int = 2000,
    max_images_per_h3: int = 5,
    stratum_weights: dict[str, float] | str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_images_per_city < 1 or max_images_per_h3 < 1:
        raise ValueError("training image caps must be positive")
    required = {"city", "hex_id", "name", "training_stratum"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"stratified training data is missing columns: {sorted(missing)}")
    unexpected = set(frame["training_stratum"].dropna().unique()) - set(STRATA)
    if unexpected or frame["training_stratum"].isna().any():
        raise ValueError(f"unexpected training strata: {sorted(unexpected)}")
    weights = parse_stratum_weights(stratum_weights)
    work = frame.drop_duplicates(["city", "name"]).copy()
    if "_city_key" not in work:
        work["_city_key"] = work["city"].map(normalize_city_name)
    work["_sample_hash"] = _stable_hash(work, seed)
    work = (
        work.sort_values(["_city_key", "hex_id", "_sample_hash"], kind="stable")
        .groupby(["_city_key", "hex_id"], group_keys=False, sort=False)
        .head(max_images_per_h3)
    )
    selected_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for city_key, city_rows in work.groupby("_city_key", sort=True):
        city_budget = min(max_images_per_city, len(city_rows))
        requested = _largest_remainder(city_budget, weights)
        queues: dict[str, pd.DataFrame] = {}
        positions: dict[str, int] = {}
        chosen: dict[str, list[pd.DataFrame]] = {stratum: [] for stratum in STRATA}
        for stratum in STRATA:
            candidates = city_rows.loc[city_rows["training_stratum"] == stratum]
            if stratum == "core":
                candidates = candidates.sort_values(
                    ["urban_intensity", "_sample_hash"],
                    ascending=[False, True],
                    kind="stable",
                )
            else:
                candidates = candidates.sort_values("_sample_hash", kind="stable")
            queues[stratum] = candidates
            count = min(requested[stratum], len(candidates))
            if count:
                chosen[stratum].append(candidates.iloc[:count])
            positions[stratum] = count
        deficit = city_budget - sum(positions.values())
        while deficit:
            active = [
                stratum
                for stratum in STRATA
                if positions[stratum] < len(queues[stratum]) and weights[stratum] > 0
            ]
            if not active:
                break
            active_total = sum(weights[stratum] for stratum in active)
            allocation = _largest_remainder(
                deficit,
                {
                    stratum: (weights[stratum] / active_total if stratum in active else 0.0)
                    for stratum in STRATA
                },
            )
            added = 0
            for stratum in active:
                available = len(queues[stratum]) - positions[stratum]
                count = min(allocation[stratum], available)
                if count:
                    start = positions[stratum]
                    chosen[stratum].append(queues[stratum].iloc[start : start + count])
                    positions[stratum] += count
                    added += count
            if not added:
                break
            deficit -= added
        for stratum in STRATA:
            available = len(queues[stratum])
            selected_count = positions[stratum]
            audit_rows.append(
                {
                    "city": city_rows["city"].iloc[0],
                    "city_key": city_key,
                    "training_stratum": stratum,
                    "city_budget": city_budget,
                    "requested_count": requested[stratum],
                    "available_count": available,
                    "selected_count": selected_count,
                    "shortfall_count": max(requested[stratum] - selected_count, 0),
                    "redistributed_in_count": max(selected_count - requested[stratum], 0),
                }
            )
            selected_frames.extend(chosen[stratum])
    if not selected_frames:
        raise ValueError("stratified training pool is empty")
    selected = pd.concat(selected_frames, ignore_index=True).drop(columns="_sample_hash")
    audit = pd.DataFrame(audit_rows)
    return selected, audit
