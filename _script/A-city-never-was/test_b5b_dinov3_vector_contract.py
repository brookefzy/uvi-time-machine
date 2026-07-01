#!/usr/bin/env python3
"""DINOv3 vector-contract tests for B5b optimized pairwise similarity."""

import importlib.util
import sys
import tempfile
import types
import unittest
import warnings
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

    spec = importlib.util.spec_from_file_location("b5b_optimized_dinov3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module


class TestDinov3VectorContract(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_root = self.root / "dinov3_hex"
        self.output_root = self.root / "dinov3_pairwise"
        self.source_root.mkdir()
        self.output_root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_processor(self, **overrides):
        config = {
            "CURATE_FOLDER_SOURCE": str(self.source_root),
            "CURATE_FOLDER_EXPORT": str(self.output_root),
            "RES_EXCLUDE": 11,
            "INPUT_TEMPLATE": "dinov3_city={city}_res_exclude={res_exclude}.parquet",
            "FEATURE_PREFIX": "e_",
            "METRIC_LABEL": "cosine",
            "similarity_threshold": 0.0,
            "row_block_size": 1,
            "use_sparse": False,
        }
        config.update(overrides)
        processor = self.module.OptimizedSimilarityProcessor(config, log_level="WARNING")
        self.addCleanup(processor.close)
        return processor

    def write_city(self, city, rows):
        path = self.source_root / f"dinov3_city={city}_res_exclude=11.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def test_dinov3_prefix_discovers_numeric_embedding_columns_and_computes_res8_cosine(self):
        self.write_city(
            "Alpha",
            [
                {
                    "hex_id": "a1",
                    "res": 8,
                    "img_count": 2,
                    "e_0001": 0.0,
                    "e_0000": 1.0,
                    "model_name": "dinov3-test",
                },
                {
                    "hex_id": "a_skip",
                    "res": 7,
                    "img_count": 1,
                    "e_0001": 1.0,
                    "e_0000": 0.0,
                    "model_name": "dinov3-test",
                },
            ],
        )
        self.write_city(
            "Beta",
            [
                {
                    "hex_id": "b1",
                    "res": 8,
                    "img_count": 3,
                    "e_0001": 0.0,
                    "e_0000": 1.0,
                    "model_name": "dinov3-test",
                },
                {
                    "hex_id": "b2",
                    "res": 8,
                    "img_count": 4,
                    "e_0001": 1.0,
                    "e_0000": 0.0,
                    "model_name": "dinov3-test",
                },
            ],
        )

        processor = self.make_processor()
        processor.validate_input_contracts(["Alpha", "Beta"])
        result = processor.compute_city_pair_similarity_blocked(
            "Alpha", "Beta", resolution=8, row_block_size=1
        )

        self.assertEqual(processor.vector_columns, ["e_0000", "e_0001"])
        self.assertEqual(
            [
                (row.hex_id1, row.hex_id2, row.city1, row.city2, round(row.similarity, 6))
                for row in result.itertuples(index=False)
            ],
            [("a1", "b1", "Alpha", "Beta", 1.0)],
        )

    def test_dinov3_dimension_mismatch_aborts_before_pairwise_work_and_names_offending_city(self):
        self.write_city(
            "Alpha",
            [{"hex_id": "a1", "res": 8, "e_0000": 1.0, "e_0001": 0.0}],
        )
        beta_path = self.write_city(
            "Beta",
            [
                {
                    "hex_id": "b1",
                    "res": 8,
                    "e_0000": 1.0,
                    "e_0001": 0.0,
                    "e_0002": 0.0,
                }
            ],
        )
        processor = self.make_processor()

        with self.assertRaisesRegex(
            ValueError, rf"Beta.*{beta_path.name}.*vector columns"
        ):
            processor.validate_input_contracts(["Alpha", "Beta"])

    def test_dinov3_missing_city_input_aborts_before_pairwise_work(self):
        self.write_city(
            "Alpha",
            [{"hex_id": "a1", "res": 8, "e_0000": 1.0, "e_0001": 0.0}],
        )
        processor = self.make_processor()

        with self.assertRaisesRegex(FileNotFoundError, "Beta.*dinov3_city=Beta"):
            processor.validate_input_contracts(["Alpha", "Beta"])

    def test_dinov3_all_pairs_threshold_includes_zero_and_negative_cosines(self):
        self.write_city(
            "Alpha",
            [{"hex_id": "a_pos", "res": 8, "e_0000": 1.0, "e_0001": 0.0}],
        )
        self.write_city(
            "Beta",
            [
                {"hex_id": "b_zero", "res": 8, "e_0000": 0.0, "e_0001": 1.0},
                {"hex_id": "b_neg", "res": 8, "e_0000": -1.0, "e_0001": 0.0},
            ],
        )

        processor = self.make_processor(similarity_threshold=-1.0)
        processor.validate_input_contracts(["Alpha", "Beta"])
        result = processor.compute_city_pair_similarity_blocked(
            "Alpha", "Beta", resolution=8, row_block_size=10
        )

        self.assertEqual(
            sorted(
                (row.hex_id1, row.hex_id2, row.city1, row.city2, round(row.similarity, 6))
                for row in result.itertuples(index=False)
            ),
            [
                ("a_pos", "b_neg", "Alpha", "Beta", -1.0),
                ("a_pos", "b_zero", "Alpha", "Beta", 0.0),
            ],
        )

    def test_save_and_merge_preserve_metric_label_without_changing_pair_columns(self):
        processor = self.make_processor()
        output_dir = processor.get_output_dir()
        pair_df = pd.DataFrame(
            [
                {
                    "hex_id1": "a1",
                    "hex_id2": "b1",
                    "city1": "Alpha",
                    "city2": "Beta",
                    "similarity": 1.0,
                }
            ]
        )

        processor.save_pair_results(pair_df, output_dir, "Alpha", "Beta", 8)
        shard_file = (
            output_dir / "temp" / "city1=Alpha" / "city2=Beta" / "part_res=8.parquet"
        )
        shard = pd.read_parquet(shard_file)
        self.assertEqual(shard["metric"].tolist(), ["cosine"])

        processor.merge_city_results(output_dir, 8)
        merged = pd.read_parquet(
            output_dir / "similarity_city=Alpha_res=8_optimized.parquet"
        )
        self.assertEqual(merged["metric"].tolist(), ["cosine"])

    def test_classifier_defaults_still_use_fixed_columns_and_original_templates(self):
        processor = self.module.OptimizedSimilarityProcessor(
            {
                "CURATE_FOLDER_SOURCE": str(self.source_root),
                "CURATE_FOLDER_EXPORT": str(self.output_root),
                "RES_EXCLUDE": 11,
            },
            log_level="WARNING",
        )
        self.addCleanup(processor.close)

        self.assertEqual(processor.vector_columns, [str(x) for x in range(127)])
        self.assertEqual(
            processor.build_input_path("Alpha"),
            self.source_root / "prob_city=Alpha_res_exclude=11.parquet",
        )
        self.assertEqual(processor.similarity_threshold, 0.01)


if __name__ == "__main__":
    unittest.main()
