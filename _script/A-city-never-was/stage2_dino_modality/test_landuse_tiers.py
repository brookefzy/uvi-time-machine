from pathlib import Path

import pandas as pd
import pytest


def module():
    from stage2_dino_modality import landuse_tiers

    return landuse_tiers


def tier_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["bogotá", "bogotá", "bogotá", "bogotá"],
            "hex_id": ["core", "suburban", "rural", "empty"],
            "resolution": [8, 8, 8, 8],
            "landuse_tier": ["core", "suburban", "rural", "rural"],
            "poi_density": [100.0, 20.0, 2.0, 0.0],
            "poi_diversity": [.9, .7, .2, 0.0],
            "poi_density_z": [2.0, .5, -.5, None],
            "poi_diversity_z": [1.0, .3, -.8, None],
            "urban_intensity": [1.5, .4, -.65, None],
        }
    )


def test_load_and_attach_tiers_normalizes_city_and_builds_four_strata(tmp_path: Path):
    path = tmp_path / "tiers.csv"
    tier_frame().to_csv(path, index=False)
    tiers = module().load_landuse_tiers(path)
    sampled = pd.DataFrame(
        {
            "city": ["Bogotá"] * 4,
            "hex_id": ["core", "suburban", "rural", "empty"],
            "res": [8] * 4,
            "name": ["a", "b", "c", "d"],
        }
    )

    joined, audit = module().attach_landuse_strata(sampled, tiers)

    assert joined.training_stratum.tolist() == [
        "core",
        "suburban",
        "occupied_rural",
        "no_poi",
    ]
    assert audit["unmatched_sampled_rows"] == 0
    assert audit["matched_sampled_rows"] == 4


def test_tier_contract_rejects_duplicate_canonical_city_h3_keys(tmp_path: Path):
    frame = tier_frame()
    frame = pd.concat([frame, frame.iloc[[0]].assign(city="Bogotá")], ignore_index=True)
    path = tmp_path / "tiers.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        module().load_landuse_tiers(path)


def test_attach_requires_complete_city_h3_coverage(tmp_path: Path):
    path = tmp_path / "tiers.csv"
    tier_frame().to_csv(path, index=False)
    sampled = pd.DataFrame(
        {"city": ["Bogotá"], "hex_id": ["missing"], "res": [8], "name": ["x"]}
    )

    with pytest.raises(ValueError, match="unmatched"):
        module().attach_landuse_strata(sampled, module().load_landuse_tiers(path))


def balanced_candidates(rows_per_stratum: int = 10) -> pd.DataFrame:
    rows = []
    for stratum_index, stratum in enumerate(module().STRATA):
        for index in range(rows_per_stratum):
            rows.append(
                {
                    "city": "A",
                    "_city_key": "a",
                    "hex_id": f"{stratum}-{index}",
                    "name": f"{stratum}-{index}.jpg",
                    "training_stratum": stratum,
                    "urban_intensity": 100 - index if stratum == "core" else index,
                    "e_0000": float(stratum_index),
                }
            )
    return pd.DataFrame(rows)


def test_stratified_pool_applies_weights_and_is_order_independent():
    frame = balanced_candidates()

    selected, audit = module().build_stratified_training_pool(
        frame,
        max_images_per_city=10,
        max_images_per_h3=1,
        seed=17,
    )
    shuffled, _ = module().build_stratified_training_pool(
        frame.sample(frac=1, random_state=9),
        max_images_per_city=10,
        max_images_per_h3=1,
        seed=17,
    )

    assert selected.training_stratum.value_counts().to_dict() == {
        "core": 4,
        "suburban": 3,
        "occupied_rural": 2,
        "no_poi": 1,
    }
    assert set(selected.name) == set(shuffled.name)
    assert audit.selected_count.sum() == 10
    assert not selected.duplicated(["city", "name"]).any()


def test_stratified_pool_caps_h3_and_redistributes_shortfall():
    frame = balanced_candidates(rows_per_stratum=5)
    frame.loc[frame.training_stratum == "core", "hex_id"] = "one-core-hex"

    selected, audit = module().build_stratified_training_pool(
        frame,
        max_images_per_city=10,
        max_images_per_h3=1,
        seed=3,
    )

    assert len(selected) == 10
    assert selected.groupby(["city", "hex_id"]).size().max() == 1
    core = audit.loc[audit.training_stratum == "core"].iloc[0]
    assert core.available_count == 1
    assert core.requested_count == 4
    assert core.selected_count == 1
    assert core.shortfall_count == 3
    assert audit.redistributed_in_count.sum() == 3


def test_stratified_pool_prioritizes_highest_intensity_core_rows():
    frame = balanced_candidates(rows_per_stratum=10)

    selected, _ = module().build_stratified_training_pool(
        frame,
        max_images_per_city=5,
        max_images_per_h3=1,
        stratum_weights={"core": 1, "suburban": 0, "occupied_rural": 0, "no_poi": 0},
    )

    assert selected.urban_intensity.tolist() == [100, 99, 98, 97, 96]


def test_parse_stratum_weights_rejects_incomplete_or_non_unit_weights():
    with pytest.raises(ValueError, match="exactly"):
        module().parse_stratum_weights("core=.5,suburban=.5")
    with pytest.raises(ValueError, match="sum to one"):
        module().parse_stratum_weights(
            "core=.4,suburban=.4,occupied_rural=.4,no_poi=.4"
        )


def test_expected_training_config_fingerprints_tiers_and_k_values(tmp_path):
    tiers = tmp_path / "tiers.csv"
    tiers.write_text("first")

    config = module().expected_training_config(
        tiers,
        max_images_per_city=2000,
        max_images_per_h3=5,
        stratum_weights="core=.4,suburban=.3,occupied_rural=.2,no_poi=.1",
        sampling_seed=42,
        requested_k_values=[32, 64, 128],
    )
    changed = tmp_path / "changed.csv"
    changed.write_text("second")
    changed_config = module().expected_training_config(
        changed,
        max_images_per_city=2000,
        max_images_per_h3=5,
        stratum_weights="core=.4,suburban=.3,occupied_rural=.2,no_poi=.1",
        sampling_seed=42,
        requested_k_values=[32, 64, 128],
    )

    assert config["requested_k_values"] == [32, 64, 128]
    assert config["landuse_tiers_sha256"] != changed_config["landuse_tiers_sha256"]
    assert module().training_config_matches(config, config)
    assert not module().training_config_matches(config, changed_config)
