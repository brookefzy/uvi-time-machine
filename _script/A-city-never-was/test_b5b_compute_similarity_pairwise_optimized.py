#!/usr/bin/env python3
"""
Regression tests for B5b_compute_similarity_pairwise-optimized.py.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parent / "B5b_compute_similarity_pairwise-optimized.py"
)


def load_module():
    fake_duckdb = types.ModuleType("duckdb")

    class FakeConnection:
        def execute(self, *_args, **_kwargs):
            return self

        def fetchdf(self):
            raise NotImplementedError

        def close(self):
            return None

    fake_duckdb.connect = lambda *_args, **_kwargs: FakeConnection()
    sys.modules.setdefault("duckdb", fake_duckdb)

    fake_sparse = types.ModuleType("scipy.sparse")
    fake_sparse.csr_matrix = lambda *args, **kwargs: None
    fake_sparse.issparse = lambda _value: False

    fake_distance = types.ModuleType("scipy.spatial.distance")
    fake_distance.cdist = lambda *args, **kwargs: None

    fake_spatial = types.ModuleType("scipy.spatial")
    fake_spatial.distance = fake_distance

    fake_scipy = types.ModuleType("scipy")
    fake_scipy.sparse = fake_sparse
    fake_scipy.spatial = fake_spatial

    sys.modules.setdefault("scipy", fake_scipy)
    sys.modules.setdefault("scipy.sparse", fake_sparse)
    sys.modules.setdefault("scipy.spatial", fake_spatial)
    sys.modules.setdefault("scipy.spatial.distance", fake_distance)

    fake_pairwise = types.ModuleType("sklearn.metrics.pairwise")
    fake_pairwise.cosine_similarity = lambda *args, **kwargs: None

    fake_metrics = types.ModuleType("sklearn.metrics")
    fake_metrics.pairwise = fake_pairwise

    fake_random_projection = types.ModuleType("sklearn.random_projection")

    class FakeGaussianRandomProjection:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, features):
            return features

    fake_random_projection.GaussianRandomProjection = FakeGaussianRandomProjection

    fake_decomposition = types.ModuleType("sklearn.decomposition")

    class FakeTruncatedSVD:
        def __init__(self, *args, **kwargs):
            self.explained_variance_ratio_ = [1.0]

        def fit_transform(self, features):
            return features

    fake_decomposition.TruncatedSVD = FakeTruncatedSVD

    fake_sklearn = types.ModuleType("sklearn")
    fake_sklearn.metrics = fake_metrics
    fake_sklearn.random_projection = fake_random_projection
    fake_sklearn.decomposition = fake_decomposition

    sys.modules.setdefault("sklearn", fake_sklearn)
    sys.modules.setdefault("sklearn.metrics", fake_metrics)
    sys.modules.setdefault("sklearn.metrics.pairwise", fake_pairwise)
    sys.modules.setdefault("sklearn.random_projection", fake_random_projection)
    sys.modules.setdefault("sklearn.decomposition", fake_decomposition)

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules.setdefault("tqdm", fake_tqdm)

    spec = importlib.util.spec_from_file_location("b5b_optimized", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestOptimizedSimilarityDefaults(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_threshold_preserves_original_positive_similarity_behavior(self):
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": self.temp_dir.name,
            "RES_EXCLUDE": 11,
        }

        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        self.assertEqual(processor.similarity_threshold, 0.01)

    def test_setup_logging_writes_to_script_local_logs_directory(self):
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": self.temp_dir.name,
            "RES_EXCLUDE": 11,
        }

        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        self.assertTrue(processor.log_file.is_absolute())
        self.assertEqual(processor.log_file.parent, MODULE_PATH.parent / "logs")
        self.assertTrue(processor.log_file.exists())

    def test_run_persists_remaining_cities_when_interrupted(self):
        city_meta_path = Path(self.temp_dir.name) / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta", "Gamma"]}).to_csv(
            city_meta_path, index=False
        )

        export_dir = Path(self.temp_dir.name) / "export"
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": str(export_dir),
            "RES_EXCLUDE": 11,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        processed_groups = []

        def fake_process_city_group(cities, _resolution):
            processed_groups.append(list(cities))
            if cities == ["Beta"]:
                raise RuntimeError("boom")
            return pd.DataFrame()

        processor.process_city_group = fake_process_city_group

        with self.assertRaisesRegex(RuntimeError, "boom"):
            processor.run(str(city_meta_path), resolution=6, group_size=1)

        progress_path = export_dir / "optimized" / "_progress_res=6_optimized.json"
        progress = json.loads(progress_path.read_text())

        self.assertEqual(processed_groups, [["Alpha"], ["Beta"]])
        self.assertEqual(progress["completed_cities"], ["Alpha"])
        self.assertEqual(progress["pending_cities"], ["Beta", "Gamma"])
        self.assertEqual(progress["status"], "failed")

    def test_run_resumes_from_previously_saved_pending_cities(self):
        city_meta_path = Path(self.temp_dir.name) / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta", "Gamma"]}).to_csv(
            city_meta_path, index=False
        )

        export_dir = Path(self.temp_dir.name) / "export"
        output_dir = export_dir / "optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "_progress_res=6_optimized.json"
        progress_path.write_text(
            json.dumps(
                {
                    "resolution": 6,
                    "group_size": 1,
                    "completed_cities": ["Alpha"],
                    "pending_cities": ["Beta", "Gamma"],
                    "status": "in_progress",
                }
            )
        )

        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": str(export_dir),
            "RES_EXCLUDE": 11,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        processed_groups = []

        def fake_process_city_group(cities, _resolution):
            processed_groups.append(list(cities))
            return pd.DataFrame()

        processor.process_city_group = fake_process_city_group
        processor.run(str(city_meta_path), resolution=6, group_size=1)

        progress = json.loads(progress_path.read_text())

        self.assertEqual(processed_groups, [["Beta"], ["Gamma"]])
        self.assertEqual(progress["completed_cities"], ["Alpha", "Beta", "Gamma"])
        self.assertEqual(progress["pending_cities"], [])
        self.assertEqual(progress["status"], "completed")


if __name__ == "__main__":
    unittest.main()
