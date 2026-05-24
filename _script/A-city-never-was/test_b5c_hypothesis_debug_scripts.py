#!/usr/bin/env python3
"""
Regression tests for the B5c hypothesis debug scripts.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
LABEL_SCRIPT_PATH = ROOT / "check_b5c_shard_row_labels.py"
COMPARE_SCRIPT_PATH = ROOT / "compare_b5c_current_vs_row_labels.py"


def load_module(module_name: str, module_path: Path):
    if "duckdb" not in sys.modules:
        fake_duckdb = types.ModuleType("duckdb")
        fake_duckdb.connect = lambda *_args, **_kwargs: None
        sys.modules["duckdb"] = fake_duckdb

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestShardRowLabelChecks(unittest.TestCase):
    def setUp(self):
        self.label_module = load_module("check_b5c_shard_row_labels", LABEL_SCRIPT_PATH)
        self.compare_module = load_module(
            "compare_b5c_current_vs_row_labels", COMPARE_SCRIPT_PATH
        )

    def test_summarize_grouped_label_rows_detects_same_city_leakage(self):
        grouped_rows = pd.DataFrame(
            [
                {"row_city1": "Alpha", "row_city2": "Alpha", "row_count": 3},
                {"row_city1": "Alpha", "row_city2": "Beta", "row_count": 5},
                {"row_city1": "Beta", "row_city2": "Alpha", "row_count": 2},
                {"row_city1": "Beta", "row_city2": "Beta", "row_count": 4},
            ]
        )

        summary = self.label_module.summarize_grouped_label_rows(
            grouped_rows=grouped_rows,
            shard_city1="Alpha",
            shard_city2="Beta",
        )

        self.assertEqual(summary["total_rows"], 14)
        self.assertEqual(summary["true_intercity_rows"], 7)
        self.assertEqual(summary["same_city_rows"], 7)
        self.assertEqual(summary["misclassified_rows"], 7)
        self.assertEqual(summary["same_city_rows_city1"], 3)
        self.assertEqual(summary["same_city_rows_city2"], 4)
        self.assertEqual(summary["rows_matching_shard_pair"], 7)

    def test_build_metric_diff_highlights_current_logic_inflation(self):
        current_df = pd.DataFrame(
            [
                {
                    "city_1": "Alpha",
                    "city_2": "Beta",
                    "row_count": 14,
                    "max_similarity": 0.99999,
                    "avg_similarity": 0.81,
                    "sum_similarity": 11.34,
                }
            ]
        )
        fixed_df = pd.DataFrame(
            [
                {
                    "city_1": "Alpha",
                    "city_2": "Beta",
                    "row_count": 7,
                    "max_similarity": 0.18,
                    "avg_similarity": 0.05,
                    "sum_similarity": 0.35,
                }
            ]
        )

        diff_df = self.compare_module.build_metric_diff(current_df, fixed_df)

        self.assertEqual(len(diff_df), 1)
        row = diff_df.iloc[0]
        self.assertEqual(row["city_1"], "Alpha")
        self.assertEqual(row["city_2"], "Beta")
        self.assertEqual(row["current_row_count"], 14)
        self.assertEqual(row["fixed_row_count"], 7)
        self.assertAlmostEqual(row["row_count_ratio"], 2.0)
        self.assertAlmostEqual(row["max_similarity_delta"], 0.81999)
        self.assertAlmostEqual(row["sum_similarity_delta"], 10.99)

    def test_compare_script_normalizes_city_filters_like_b7(self):
        self.assertEqual(
            self.compare_module.normalize_city_name_for_filter("Hong Kong"),
            "hongkong",
        )
        self.assertEqual(
            self.compare_module.normalize_city_name_for_filter("HongKong"),
            "hongkong",
        )
        self.assertEqual(
            self.compare_module.normalize_city_name_for_filter("Portland, OR"),
            "portland",
        )

    def test_compare_script_resolves_raw_and_normalized_city_filters(self):
        available = ["Hong Kong", "Portland, OR", "Singapore"]

        self.assertEqual(
            self.compare_module.resolve_city_filter_value("Hong Kong", available),
            "Hong Kong",
        )
        self.assertEqual(
            self.compare_module.resolve_city_filter_value("hongkong", available),
            "Hong Kong",
        )
        self.assertEqual(
            self.compare_module.resolve_city_filter_value("HongKong", available),
            "Hong Kong",
        )
        self.assertEqual(
            self.compare_module.resolve_city_filter_value("portland", available),
            "Portland, OR",
        )


if __name__ == "__main__":
    unittest.main()
