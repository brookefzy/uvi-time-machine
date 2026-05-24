#!/usr/bin/env python3
"""
Regression tests for the B5b artifact debug script.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "inspect_b5b_res8_artifacts.py"


def load_module():
    if "duckdb" not in sys.modules:
        fake_duckdb = types.ModuleType("duckdb")
        fake_duckdb.connect = lambda *_args, **_kwargs: None
        sys.modules["duckdb"] = fake_duckdb

    spec = importlib.util.spec_from_file_location("inspect_b5b_res8_artifacts", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestB5bArtifactDebugHelpers(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_summarize_city_source_df_reports_duplicate_hexes_and_vectors(self):
        df = pd.DataFrame(
            [
                {"hex_id": "h1", "vector_hash": "v1"},
                {"hex_id": "h1", "vector_hash": "v1"},
                {"hex_id": "h2", "vector_hash": "v2"},
                {"hex_id": "h2", "vector_hash": "v3"},
                {"hex_id": "h3", "vector_hash": "v3"},
            ]
        )

        summary = self.module.summarize_city_source_df("Hong Kong", df)

        self.assertEqual(summary["city"], "Hong Kong")
        self.assertEqual(summary["row_count"], 5)
        self.assertEqual(summary["unique_hex_count"], 3)
        self.assertEqual(summary["duplicate_hex_row_count"], 2)
        self.assertEqual(summary["duplicate_hex_id_count"], 2)
        self.assertEqual(summary["multi_vector_hex_count"], 1)
        self.assertEqual(summary["unique_vector_count"], 3)
        self.assertEqual(summary["duplicate_vector_row_count"], 2)

    def test_summarize_joined_pair_df_reports_exact_one_and_hash_matches(self):
        df = pd.DataFrame(
            [
                {
                    "hex_id1": "a1",
                    "hex_id2": "b1",
                    "similarity": 1.0,
                    "vector_hash1": "same",
                    "vector_hash2": "same",
                },
                {
                    "hex_id1": "a2",
                    "hex_id2": "b2",
                    "similarity": 1.0,
                    "vector_hash1": "left",
                    "vector_hash2": "right",
                },
                {
                    "hex_id1": "a3",
                    "hex_id2": "b3",
                    "similarity": 0.95,
                    "vector_hash1": "same2",
                    "vector_hash2": "same2",
                },
            ]
        )

        summary = self.module.summarize_joined_pair_df("Hong Kong", "Singapore", df)

        self.assertEqual(summary["city_1"], "Hong Kong")
        self.assertEqual(summary["city_2"], "Singapore")
        self.assertEqual(summary["pair_row_count"], 3)
        self.assertEqual(summary["sim_eq_1_count"], 2)
        self.assertEqual(summary["sim_ge_0_999999_count"], 2)
        self.assertEqual(summary["equal_hash_pair_count"], 2)
        self.assertEqual(summary["sim_eq_1_equal_hash_count"], 1)
        self.assertEqual(summary["sim_eq_1_unequal_hash_count"], 1)


if __name__ == "__main__":
    unittest.main()
