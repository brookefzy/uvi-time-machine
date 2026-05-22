#!/usr/bin/env python3
"""
Regression tests for the external stage3 POI tier configuration.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
import pandas as pd


MODULE_PATH = Path(
    "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/"
    "_scripts/03_landuse/stage3_poi_density_diversity_tiers.py"
)


def stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_module():
    stub_module("geopandas", GeoDataFrame=object, read_parquet=lambda *_args, **_kwargs: None)
    stub_module("h3")
    stub_module("matplotlib")
    stub_module("matplotlib.pyplot", subplots=lambda *args, **kwargs: (None, None), tight_layout=lambda: None, close=lambda *_args, **_kwargs: None)
    stub_module(
        "pyarrow",
        __version__="0.0.0",
        Table=types.SimpleNamespace(from_pandas=lambda *args, **kwargs: None),
        string=lambda: None,
        schema=lambda *args, **kwargs: None,
        field=lambda *args, **kwargs: None,
        types=types.SimpleNamespace(
            is_large_string=lambda *_: False,
            is_dictionary=lambda *_: False,
            is_string=lambda *_: False,
        ),
    )
    stub_module("pyarrow.parquet", write_table=lambda *_args, **_kwargs: None)
    stub_module("shapely")
    stub_module("shapely.geometry", Polygon=lambda coords: coords)

    poi_taxonomy = stub_module(
        "poi_category_taxonomy",
        get_taxonomy_metadata=lambda: {},
        get_top_category=lambda _value: "test",
        normalize_top_category=lambda value: value,
    )
    sys.modules["poi_category_taxonomy"] = poi_taxonomy

    spec = importlib.util.spec_from_file_location("stage3_poi_density_diversity_tiers", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_module_without_optional_stubs():
    for module_name in [
        "stage3_poi_density_diversity_tiers",
        "geopandas",
        "h3",
        "matplotlib",
        "matplotlib.pyplot",
        "pyarrow",
        "pyarrow.parquet",
        "shapely",
        "shapely.geometry",
    ]:
        sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        "stage3_poi_density_diversity_tiers",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestStage3PoiTierConfig(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_stage3_default_resolution_is_8(self):
        self.assertEqual(self.module.DEFAULT_RESOLUTION, 8)

    def test_stage3_builds_output_folder_from_resolution(self):
        self.assertEqual(
            self.module.build_output_folder(8),
            "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/landuse_poi_res=8",
        )
        self.assertEqual(
            self.module.build_output_folder(7),
            "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/landuse_poi_res=7",
        )

    def test_collect_city_processing_status_reports_ready_and_missing_hex_ids(self):
        self.module.load_city_hex_ids = (
            lambda city, res: {"hex-1", "hex-2"} if city == "malegaon" and res == 8 else None
        )
        poi_files = [
            "/tmp/malegaon.parquet",
            "/tmp/london.parquet",
        ]

        statuses = self.module.collect_city_processing_status(poi_files, [8])

        self.assertEqual(
            statuses,
            [
                {
                    "city": "malegaon",
                    "resolution": 8,
                    "status": "ready",
                    "hex_count": 2,
                },
                {
                    "city": "london",
                    "resolution": 8,
                    "status": "missing_hex_ids",
                    "hex_count": 0,
                },
            ],
        )

    def test_stage3_uses_boundary_hex_cache_root(self):
        self.assertIn("city_boundary_hex_ids", self.module.CITY_HEX_ROOT)

    def test_compute_city_hex_features_keeps_zero_poi_boundary_hexes(self):
        class FakePoint:
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y

        places_df = pd.DataFrame(
            {
                "geometry": [FakePoint(0.0, 0.0)],
                "categories": [["food"]],
            }
        )

        original_prepare = self.module.prepare_places_categories
        original_load_hex_ids = self.module.load_city_hex_ids
        original_h3 = self.module.latlng_to_cell_compat
        original_area = self.module.hex_area_km2
        try:
            self.module.prepare_places_categories = lambda df: df.assign(
                top_category="food",
                macro_category="food",
            )
            self.module.load_city_hex_ids = lambda city, res: {"hex-a", "hex-b"}
            self.module.latlng_to_cell_compat = lambda lat, lng, res: "hex-a"
            self.module.hex_area_km2 = lambda res: 1.0

            result = self.module.compute_city_hex_features(
                places_df=places_df,
                city="malegaon",
                res_list=[8],
            )

            self.assertEqual(set(result["h3_index"]), {"hex-a", "hex-b"})
            zero_row = result[result["h3_index"] == "hex-b"].iloc[0]
            self.assertEqual(zero_row["poi_count"], 0)
            self.assertEqual(zero_row["poi_density"], 0)
        finally:
            self.module.prepare_places_categories = original_prepare
            self.module.load_city_hex_ids = original_load_hex_ids
            self.module.latlng_to_cell_compat = original_h3
            self.module.hex_area_km2 = original_area

    def test_check_only_mode_skips_output_directory_creation(self):
        recorded = []
        original_argv = sys.argv
        original_glob = self.module.glob
        original_makedirs = self.module.os.makedirs
        original_load_city_hex_ids = self.module.load_city_hex_ids
        original_print_summary = self.module.print_processing_status_summary
        try:
            sys.argv = ["stage3_poi_density_diversity_tiers.py", "--check-only"]
            self.module.glob = lambda _pattern: ["/tmp/malegaon.parquet"]
            self.module.os.makedirs = (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("os.makedirs should not be called in --check-only mode")
                )
            )
            self.module.load_city_hex_ids = lambda city, res: {"hex-1"} if city == "malegaon" and res == 8 else None
            self.module.print_processing_status_summary = lambda statuses: recorded.extend(statuses)

            self.module.main()

            self.assertEqual(
                recorded,
                [
                    {
                        "city": "malegaon",
                        "resolution": 8,
                        "status": "ready",
                        "hex_count": 1,
                    }
                ],
            )
        finally:
            sys.argv = original_argv
            self.module.glob = original_glob
            self.module.os.makedirs = original_makedirs
            self.module.load_city_hex_ids = original_load_city_hex_ids
            self.module.print_processing_status_summary = original_print_summary

class TestStage3CheckOnlyImports(unittest.TestCase):
    def test_module_imports_without_optional_geospatial_dependencies(self):
        module = load_module_without_optional_stubs()
        self.assertTrue(hasattr(module, "collect_city_processing_status"))
        self.assertTrue(hasattr(module, "build_output_folder"))


if __name__ == "__main__":
    unittest.main()
