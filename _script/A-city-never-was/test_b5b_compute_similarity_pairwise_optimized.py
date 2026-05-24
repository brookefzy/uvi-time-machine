#!/usr/bin/env python3
"""
Regression tests for B5b_compute_similarity_pairwise-optimized.py.
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", message="GPU support not available.*")


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

    def fake_cosine_similarity(features1, features2=None):
        left = np.asarray(features1, dtype=float)
        right = left if features2 is None else np.asarray(features2, dtype=float)
        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        left_safe = left / np.clip(left_norm, 1e-12, None)
        right_safe = right / np.clip(right_norm, 1e-12, None)
        return left_safe @ right_safe.T

    fake_pairwise.cosine_similarity = fake_cosine_similarity

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    def test_run_persists_remaining_pairs_when_interrupted(self):
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

        processed_pairs = []

        def fake_process_city_pair(city1, city2, _resolution):
            processed_pairs.append((city1, city2))
            if (city1, city2) == ("Alpha", "Gamma"):
                raise RuntimeError("boom")
            return pd.DataFrame()

        processor.process_city_pair = fake_process_city_pair

        with self.assertRaisesRegex(RuntimeError, "boom"):
            processor.run(str(city_meta_path), resolution=6, group_size=1)

        progress_path = export_dir / "optimized" / "_progress_res=6_optimized.json"
        progress = json.loads(progress_path.read_text())

        self.assertEqual(processed_pairs, [("Alpha", "Beta"), ("Alpha", "Gamma")])
        self.assertEqual(progress["completed_pair_keys"], ["Alpha__Beta__res=6"])
        self.assertEqual(
            progress["pending_pairs"],
            [["Alpha", "Gamma"], ["Beta", "Gamma"]],
        )
        self.assertEqual(progress["status"], "failed")

    def test_run_resumes_from_previously_saved_pending_pairs(self):
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
                    "completed_pair_keys": ["Alpha__Beta__res=6"],
                    "pending_pairs": [["Alpha", "Gamma"], ["Beta", "Gamma"]],
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

        processed_pairs = []

        def fake_process_city_pair(city1, city2, _resolution):
            processed_pairs.append((city1, city2))
            return pd.DataFrame()

        processor.process_city_pair = fake_process_city_pair
        processor.run(str(city_meta_path), resolution=6, group_size=1)

        progress = json.loads(progress_path.read_text())

        self.assertEqual(processed_pairs, [("Alpha", "Gamma"), ("Beta", "Gamma")])
        self.assertEqual(
            progress["completed_pair_keys"],
            ["Alpha__Beta__res=6", "Alpha__Gamma__res=6", "Beta__Gamma__res=6"],
        )
        self.assertEqual(progress["pending_pairs"], [])
        self.assertEqual(progress["status"], "completed")

    def test_create_city_pairs_matches_original_global_order(self):
        city_meta_path = Path(self.temp_dir.name) / "city_meta.csv"
        pd.DataFrame({"City": ["Gamma", "Alpha", "Beta", "Alpha"]}).to_csv(
            city_meta_path, index=False
        )

        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": self.temp_dir.name,
            "RES_EXCLUDE": 11,
        }

        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        city_pairs = processor.create_city_pairs(str(city_meta_path))

        self.assertEqual(
            city_pairs,
            [("Alpha", "Beta"), ("Alpha", "Gamma"), ("Beta", "Gamma")],
        )

    def test_run_resumes_from_previously_completed_pairs(self):
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
                    "completed_pair_keys": ["Alpha__Beta__res=6"],
                    "pending_pairs": [["Alpha", "Gamma"], ["Beta", "Gamma"]],
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

        processed_pairs = []

        def fake_process_city_pair(city1, city2, _resolution):
            processed_pairs.append((city1, city2))
            return pd.DataFrame()

        processor.process_city_pair = fake_process_city_pair
        processor.merge_city_results = lambda *_args, **_kwargs: None
        processor.cleanup_temp_outputs = lambda *_args, **_kwargs: None
        processor.run(str(city_meta_path), resolution=6, group_size=10)

        progress = json.loads(progress_path.read_text())

        self.assertEqual(processed_pairs, [("Alpha", "Gamma"), ("Beta", "Gamma")])
        self.assertEqual(
            progress["completed_pair_keys"],
            ["Alpha__Beta__res=6", "Alpha__Gamma__res=6", "Beta__Gamma__res=6"],
        )
        self.assertEqual(progress["pending_pairs"], [])
        self.assertEqual(progress["status"], "completed")

    def test_blocked_city_pair_similarity_matches_exact_results(self):
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": self.temp_dir.name,
            "RES_EXCLUDE": 11,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        alpha = pd.DataFrame(
            [
                {"hex_id": "a1", "city": "Alpha", "0": 1.0, "1": 0.0},
                {"hex_id": "a2", "city": "Alpha", "0": 0.0, "1": 1.0},
            ]
        )
        beta = pd.DataFrame(
            [
                {"hex_id": "b1", "city": "Beta", "0": 1.0, "1": 0.0},
                {"hex_id": "b2", "city": "Beta", "0": 1.0, "1": 1.0},
            ]
        )

        required_columns = ["hex_id", "city", *processor.vector_columns]
        alpha = alpha.reindex(columns=required_columns, fill_value=0.0)
        beta = beta.reindex(columns=required_columns, fill_value=0.0)

        def fake_load_city_features(city_name, _resolution):
            if city_name == "Alpha":
                return alpha.copy()
            if city_name == "Beta":
                return beta.copy()
            return pd.DataFrame()

        processor.load_city_features = fake_load_city_features

        blocked = processor.compute_city_pair_similarity_blocked(
            "Alpha", "Beta", resolution=6, row_block_size=1
        )

        actual = {
            (row.hex_id1, row.hex_id2, row.city1, row.city2): round(row.similarity, 6)
            for row in blocked.itertuples(index=False)
        }

        self.assertEqual(
            actual,
            {
                ("a1", "b1", "Alpha", "Beta"): 1.0,
                ("a1", "b2", "Alpha", "Beta"): 0.707107,
                ("a2", "b2", "Alpha", "Beta"): 0.707107,
            },
        )

    def test_run_without_checkpoint_skips_pairs_for_completed_city_outputs(self):
        city_meta_path = Path(self.temp_dir.name) / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta", "Gamma"]}).to_csv(
            city_meta_path, index=False
        )

        export_dir = Path(self.temp_dir.name) / "export"
        output_dir = export_dir / "optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "similarity_city=Alpha_res=6_optimized.parquet").touch()

        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": str(export_dir),
            "RES_EXCLUDE": 11,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        processed_pairs = []

        def fake_process_city_pair(city1, city2, _resolution):
            processed_pairs.append((city1, city2))
            return pd.DataFrame()

        processor.process_city_pair = fake_process_city_pair
        processor.run(str(city_meta_path), resolution=6, group_size=10)

        self.assertEqual(processed_pairs, [("Beta", "Gamma")])

    def test_run_with_purge_all_for_resolution_recomputes_from_clean_state(self):
        city_meta_path = Path(self.temp_dir.name) / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta", "Gamma"]}).to_csv(
            city_meta_path, index=False
        )

        export_dir = Path(self.temp_dir.name) / "export"
        output_dir = export_dir / "optimized"
        temp_dir = output_dir / "temp" / "city1=Alpha" / "city2=Beta"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "_progress_res=6_optimized.json").write_text(
            json.dumps(
                {
                    "resolution": 6,
                    "completed_pair_keys": ["Alpha__Beta__res=6"],
                    "pending_pairs": [["Alpha", "Gamma"], ["Beta", "Gamma"]],
                    "status": "in_progress",
                }
            )
        )
        (output_dir / "similarity_city=Alpha_res=6_optimized.parquet").touch()
        (temp_dir / "part_res=6.parquet").touch()

        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": str(export_dir),
            "RES_EXCLUDE": 11,
            "purge_all_for_resolution": True,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        processed_pairs = []

        def fake_process_city_pair(city1, city2, _resolution):
            processed_pairs.append((city1, city2))
            return pd.DataFrame()

        processor.process_city_pair = fake_process_city_pair
        processor.run(str(city_meta_path), resolution=6, group_size=10)

        self.assertEqual(
            processed_pairs,
            [("Alpha", "Beta"), ("Alpha", "Gamma"), ("Beta", "Gamma")],
        )

    def test_purge_resolution_outputs_before_date_only_removes_older_matching_files(self):
        export_dir = Path(self.temp_dir.name) / "export"
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": str(export_dir),
            "RES_EXCLUDE": 11,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        output_dir = processor.get_output_dir()
        temp_pair_dir = output_dir / "temp" / "city1=Alpha" / "city2=Beta"
        temp_pair_dir.mkdir(parents=True, exist_ok=True)
        stale_progress = output_dir / "_progress_res=6_optimized.json"
        stale_final = output_dir / "similarity_city=Alpha_res=6_optimized.parquet"
        fresh_final = output_dir / "similarity_city=Beta_res=6_optimized.parquet"
        other_resolution = output_dir / "similarity_city=Alpha_res=7_optimized.parquet"
        stale_temp = temp_pair_dir / "part_res=6.parquet"
        other_temp = temp_pair_dir / "part_res=7.parquet"

        for path in [
            stale_progress,
            stale_final,
            fresh_final,
            other_resolution,
            stale_temp,
            other_temp,
        ]:
            path.touch()

        stale_ts = datetime(2024, 5, 17, 12, 0, 0).timestamp()
        fresh_ts = datetime(2026, 5, 23, 12, 0, 0).timestamp()
        os.utime(stale_progress, (stale_ts, stale_ts))
        os.utime(stale_final, (stale_ts, stale_ts))
        os.utime(stale_temp, (stale_ts, stale_ts))
        os.utime(fresh_final, (fresh_ts, fresh_ts))
        os.utime(other_resolution, (stale_ts, stale_ts))
        os.utime(other_temp, (stale_ts, stale_ts))

        summary = processor.purge_resolution_outputs(
            output_dir, resolution=6, before_date=date(2026, 5, 23)
        )

        self.assertEqual(summary["files_deleted"], 3)
        self.assertFalse(stale_progress.exists())
        self.assertFalse(stale_final.exists())
        self.assertFalse(stale_temp.exists())
        self.assertTrue(fresh_final.exists())
        self.assertTrue(other_resolution.exists())
        self.assertTrue(other_temp.exists())

    def test_merge_city_results_only_writes_requested_city1_output(self):
        export_dir = Path(self.temp_dir.name) / "export"
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir.name,
            "CURATE_FOLDER_EXPORT": str(export_dir),
            "RES_EXCLUDE": 11,
        }
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)

        output_dir = processor.get_output_dir()
        pair_dir = processor.get_pair_temp_dir(output_dir, "Alpha", "Beta")
        shard_path = pair_dir / "part_res=6.parquet"
        shard_path.touch()

        original_read_parquet = self.module.pd.read_parquet
        original_to_parquet = pd.DataFrame.to_parquet
        written_paths = []

        def fake_read_parquet(path, *args, **kwargs):
            if Path(path) == shard_path:
                return pd.DataFrame(
                    [
                        {
                            "hex_id1": "a1",
                            "hex_id2": "b1",
                            "city1": "Alpha",
                            "city2": "Beta",
                            "similarity": 1.0,
                        },
                        {
                            "hex_id1": "b1",
                            "hex_id2": "b2",
                            "city1": "Beta",
                            "city2": "Beta",
                            "similarity": 0.7,
                        },
                    ]
                )
            return original_read_parquet(path, *args, **kwargs)

        def fake_to_parquet(self_df, path, *args, **kwargs):
            written_paths.append(str(path))
            Path(path).touch()

        self.module.pd.read_parquet = fake_read_parquet
        pd.DataFrame.to_parquet = fake_to_parquet
        self.addCleanup(setattr, self.module.pd, "read_parquet", original_read_parquet)
        self.addCleanup(setattr, pd.DataFrame, "to_parquet", original_to_parquet)

        processor.merge_city_results(output_dir, resolution=6)

        self.assertEqual(len(written_paths), 1)
        self.assertIn("Alpha", written_paths[0])
        self.assertIn("res=6", written_paths[0])
        self.assertNotIn("Beta_optimized.parquet", written_paths[0])


if __name__ == "__main__":
    unittest.main()
