import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd


def load():
    script = Path(__file__).with_name("03_build_mode_gallery.py")
    spec = importlib.util.spec_from_file_location("gallery", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gallery_cli_resolves_repository_modules_outside_repo_working_directory(tmp_path):
    script = Path(__file__).with_name("03_build_mode_gallery.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gallery_reads_partitioned_datasets_and_joins_on_city_and_name(tmp_path):
    module = load()
    sampled = tmp_path / "sampled"
    index = tmp_path / "index"
    sampled.mkdir(); index.mkdir()
    pd.DataFrame({"city":["A"],"name":["same.jpg"],"hex_id":["h"],"e_0000":[1.]}).to_parquet(sampled / "city=A.parquet")
    pd.DataFrame({"city":["B"],"name":["same.jpg"],"hex_id":["wrong"],"e_0000":[1.]}).to_parquet(sampled / "city=B.parquet")
    pd.DataFrame({"city":["A"],"name":["same.jpg"],"path":["/a.jpg"]}).to_parquet(index / "city=A.parquet")
    pd.DataFrame({"city":["B"],"name":["same.jpg"],"path":["/b.jpg"]}).to_parquet(index / "city=B.parquet")
    centroids = pd.DataFrame({"mode_id":[0],"e_0000":[1.]})
    rows = module.build_representatives(module.read_parquet_dataset(sampled), centroids, module.read_parquet_dataset(index))
    assert set(rows.path) == {"/a.jpg", "/b.jpg"}


def test_gallery_derives_city_from_partition_filename_when_index_omits_it(tmp_path):
    module = load()
    path = tmp_path / "city=Paris.parquet"
    pd.DataFrame({"name":["image.jpg"], "path":["/image.jpg"]}).to_parquet(path)
    assert module.read_parquet_dataset(path).city.tolist() == ["Paris"]


def test_gallery_maps_existing_city_stem_index_files_to_sampled_cities(tmp_path):
    module = load()
    sampled = pd.DataFrame({"city":["Hong Kong"],"name":["image.jpg"],"hex_id":["h"],"e_0000":[1.]})
    index_path = tmp_path / "hongkong.parquet"
    pd.DataFrame({"name":["image.jpg"], "path":["/image.jpg"]}).to_parquet(index_path)
    centroids = pd.DataFrame({"mode_id":[0], "e_0000":[1.]})
    rows = module.build_representatives(sampled, centroids, module.read_parquet_dataset(index_path))
    assert rows.city.tolist() == ["Hong Kong"]


def test_gallery_selects_distinct_cities_before_reusing_a_city():
    module = load()
    sampled = pd.DataFrame(
        {
            "city": ["A", "A", "B"],
            "name": ["a-best.jpg", "a-second.jpg", "b-best.jpg"],
            "hex_id": ["a1", "a2", "b1"],
            "e_0000": [1.0, 0.99, 0.8],
        }
    )
    index = pd.DataFrame(
        {
            "city": ["A", "A", "B"],
            "name": ["a-best.jpg", "a-second.jpg", "b-best.jpg"],
            "path": ["/a-best.jpg", "/a-second.jpg", "/b-best.jpg"],
        }
    )
    centroids = pd.DataFrame({"mode_id": [0], "e_0000": [1.0]})

    rows = module.build_representatives(
        sampled,
        centroids,
        index,
        images_per_mode=2,
    )

    assert rows.city.tolist() == ["A", "B"]
    assert rows.name.tolist() == ["a-best.jpg", "b-best.jpg"]


def test_gallery_attaches_tier_metadata_and_reports_zero_core_support(tmp_path):
    module = load()
    rows = pd.DataFrame(
        {
            "city": ["Bogotá"],
            "hex_id": ["h1"],
            "res": [8],
            "name": ["image.jpg"],
            "path": ["images/image.jpg"],
            "mode_id": [0],
            "assignment_cosine": [.9],
        }
    )
    tiers = pd.DataFrame(
        {
            "city": ["bogotá"],
            "hex_id": ["h1"],
            "resolution": [8],
            "landuse_tier": ["suburban"],
            "poi_density": [10.0],
            "poi_diversity": [.5],
            "poi_density_z": [.2],
            "poi_diversity_z": [.4],
            "urban_intensity": [.3],
        }
    )
    tier_path = tmp_path / "tiers.csv"
    tiers.to_csv(tier_path, index=False)
    enriched = module.attach_gallery_tiers(rows, tier_path)
    output = tmp_path / "index.html"

    module.render_gallery(enriched, output)

    html = output.read_text()
    assert enriched.landuse_tier.tolist() == ["suburban"]
    assert "urban intensity 0.3000" in html
    assert "core representatives: 0" in html
    assert "suburban representatives: 1" in html
