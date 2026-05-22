#!/usr/bin/env python3
"""
Regression tests for the external city-boundary H3 generator.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(
    "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/"
    "_scripts/03_landuse/generate_city_boundary_hex_ids.py"
)


def stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_module():
    stub_module(
        "geopandas",
        read_file=lambda *_args, **_kwargs: None,
        GeoDataFrame=object,
    )
    stub_module("h3")
    stub_module("pandas")

    spec = importlib.util.spec_from_file_location(
        "generate_city_boundary_hex_ids",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGenerateCityBoundaryHexIds(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_default_resolution_is_8(self):
        self.assertEqual(self.module.DEFAULT_RESOLUTION, 8)

    def test_output_path_is_partitioned_by_resolution_and_city(self):
        self.assertEqual(
            self.module.build_output_path("london", 8),
            "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_curated/03_similarity_grid/city_boundary_hex_ids/res=8/london.parquet",
        )


if __name__ == "__main__":
    unittest.main()
