from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def test_poi_stratified_fit_audits_pool_and_preserves_original_histogram_counts(tmp_path: Path):
    sampled_root = tmp_path / "sampled"
    output_root = tmp_path / "output"
    sampled_root.mkdir()
    tier_rows = []
    original_frames = []
    strata = [
        ("core", 100.0, 1.2, [1.0, 0.0]),
        ("suburban", 20.0, .4, [0.0, 1.0]),
        ("rural", 2.0, -.4, [-1.0, 0.0]),
        ("rural", 0.0, None, [0.0, -1.0]),
    ]
    for city in ("Alpha City", "Beta City"):
        rows = []
        city_key = city.lower().replace(" ", "")
        for stratum_index, (tier, density, intensity, vector) in enumerate(strata):
            hex_id = f"{city_key}-{stratum_index}"
            tier_rows.append(
                {
                    "city": city_key,
                    "hex_id": hex_id,
                    "resolution": 8,
                    "landuse_tier": tier,
                    "poi_density": density,
                    "poi_diversity": .5 if density else 0.0,
                    "poi_density_z": intensity,
                    "poi_diversity_z": intensity,
                    "urban_intensity": intensity,
                }
            )
            for image_index in range(3):
                rows.append(
                    {
                        "city": city,
                        "hex_id": hex_id,
                        "res": 8,
                        "name": f"{hex_id}-{image_index}.jpg",
                        "e_0000": np.float32(vector[0]),
                        "e_0001": np.float32(vector[1]),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_parquet(sampled_root / f"city={city}.parquet", index=False)
        original_frames.append(frame)
    tier_path = tmp_path / "tiers.csv"
    pd.DataFrame(tier_rows).to_csv(tier_path, index=False)
    script = Path(__file__).with_name("02_fit_evaluate_codebooks.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(sampled_root),
            "--output-root",
            str(output_root),
            "--landuse-tiers",
            str(tier_path),
            "--k",
            "2",
            "--max-training-images-per-city",
            "8",
            "--max-training-images-per-h3",
            "2",
            "--stability-seed-count",
            "2",
            "--niter",
            "5",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    audit = __import__("json").loads((output_root / "training_pool_audit.json").read_text())
    assert audit["training_sampling_strategy"] == "city_poi_stratified_v1"
    assert audit["join_audit"]["unmatched_sampled_rows"] == 0
    assert audit["training_pool_image_count"] == 16
    scorecard = pd.read_parquet(output_root / "scorecard.parquet")
    assert scorecard.loc[0, "training_sampling_strategy"] == "city_poi_stratified_v1"
    assert scorecard.loc[0, "landuse_tiers_sha256"] == audit["landuse_tiers_sha256"]

    from stage2_dino_modality.mode_ops import assign_modes, build_histogram

    original = pd.concat(original_frames, ignore_index=True)
    vectors = original[["e_0000", "e_0001"]].to_numpy("float32")
    assignments = assign_modes(
        original.drop(columns=["e_0000", "e_0001"]),
        vectors,
        np.eye(2, dtype=np.float32),
        "model",
    )
    histogram = build_histogram(assignments)
    assert histogram.groupby(["city", "hex_id"]).sampled_image_count.first().eq(3).all()
