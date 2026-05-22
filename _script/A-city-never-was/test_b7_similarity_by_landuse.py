#!/usr/bin/env python3
"""
Regression tests for B7_similarity_by_landuse.py.
"""

import importlib.util
import tempfile
import os
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "B7_similarity_by_landuse.py"


def load_module():
    if "duckdb" not in sys.modules:
        fake_duckdb = types.ModuleType("duckdb")

        class FakeConnection:
            def execute(self, *_args, **_kwargs):
                return self

            def df(self):
                raise NotImplementedError

            def close(self):
                return None

        fake_duckdb.connect = lambda *_args, **_kwargs: FakeConnection()
        sys.modules["duckdb"] = fake_duckdb

    if "tqdm" not in sys.modules:
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
        sys.modules["tqdm"] = fake_tqdm

    spec = importlib.util.spec_from_file_location("b7_similarity_by_landuse", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestB7SimilarityByLanduseHelpers(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_remote_defaults_target_server_paths(self):
        self.assertEqual(
            self.module.DEFAULT_PAIRWISE_ROOT,
            "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
        )
        self.assertEqual(
            self.module.DEFAULT_STAGE3_LANDUSE_ROOT,
            "/lustre1/g/geog_pyloo/05_timemachine/_transformed/landuse_poi_res=8",
        )
        self.assertEqual(
            self.module.DEFAULT_EXPORT_FOLDER,
            "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_by_landuse",
        )

    def test_normalize_city_name_strips_us_state_suffix(self):
        self.assertEqual(self.module.normalize_city_name("Portland, OR"), "portland")
        self.assertEqual(self.module.normalize_city_name("Sao Paulo"), "saopaulo")

    def test_resolve_landuse_keys_for_stage3_all(self):
        self.assertEqual(
            self.module.resolve_landuse_keys("all", "stage3"),
            ["core", "suburban", "rural", "built", "all"],
        )

    def test_build_summary_output_path_for_stage3_zero_fill(self):
        output_path = self.module.build_summary_output_path(
            export_folder="/tmp/output",
            landuse_type="core",
            res=8,
            zero_fill_avg=True,
            landuse_source_name="stage3",
            tier_method="pct",
        )
        self.assertEqual(
            output_path,
            "/tmp/output/similarity_summary_core_city_res=8_pct_zerofill.csv",
        )

    def test_summarize_resolution_coverage_flags_sparse_result(self):
        report = self.module.summarize_resolution_coverage(
            records=[
                ("accra", [6, 7]),
                ("london", [6, 7]),
                ("malegaon", [6, 7, 8]),
            ],
            resolution=8,
        )
        self.assertEqual(report.city_count, 1)
        self.assertEqual(report.city_names, ["malegaon"])
        self.assertTrue(report.is_sparse)

    def test_build_parser_accepts_legacy_cli_aliases(self):
        parser = self.module.build_parser()
        args = parser.parse_args(
            [
                "--zero-fill-avg",
                "--use-two-phase",
                "--landuse-source",
                "stage3",
                "--tier-method",
                "pct",
                "--res",
                "8",
            ]
        )
        self.assertTrue(args.zero_fill_avg)
        self.assertTrue(args.use_two_phase)
        self.assertEqual(args.resolution, 8)
        self.assertEqual(args.landuse_source, "stage3")
        self.assertEqual(args.tier_method, "pct")

    def test_detect_pairwise_source_supports_optimized_city_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(
                tmpdir, "similarity_intracity_city=Alpha_res=8.parquet"
            )
            os.makedirs(dataset_dir)
            Path(dataset_dir, "part_0.parquet").touch()

            config = self.module.detect_pairwise_source(tmpdir, 8)

        self.assertEqual(config.source, "optimized_city")
        self.assertEqual(config.city1_column, "city_1")
        self.assertEqual(config.city2_column, "city_2")
        self.assertEqual(config.glob_pattern, os.path.join(tmpdir, "*res=8.parquet"))

    def test_collect_pairwise_input_paths_supports_mixed_optimized_city_layouts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            single_file = Path(tmpdir, "similarity_intracity_city=Sydney_res=8.parquet")
            single_file.touch()

            dataset_dir = Path(tmpdir, "similarity_intracity_city=Astrakhan_res=8.parquet")
            dataset_dir.mkdir()
            dataset_part = dataset_dir / "part_0.parquet"
            dataset_part.touch()

            config = self.module.detect_pairwise_source(tmpdir, 8)
            paths = self.module.collect_pairwise_input_paths(config)

        self.assertEqual(
            paths,
            [str(dataset_part), str(single_file)],
        )


if __name__ == "__main__":
    unittest.main()
