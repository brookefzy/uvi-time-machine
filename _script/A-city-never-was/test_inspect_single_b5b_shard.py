#!/usr/bin/env python3
"""
Regression tests for the single-shard B5b inspection script.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "inspect_single_b5b_shard.py"


def load_module():
    if "duckdb" not in sys.modules:
        fake_duckdb = types.ModuleType("duckdb")
        fake_duckdb.connect = lambda *_args, **_kwargs: None
        sys.modules["duckdb"] = fake_duckdb

    spec = importlib.util.spec_from_file_location("inspect_single_b5b_shard", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestInspectSingleB5bShardHelpers(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_classify_membership_side(self):
        self.assertEqual(self.module.classify_membership_side(True, False), "city1_only")
        self.assertEqual(self.module.classify_membership_side(False, True), "city2_only")
        self.assertEqual(self.module.classify_membership_side(True, True), "both")
        self.assertEqual(self.module.classify_membership_side(False, False), "neither")

    def test_summarize_shard_rows_reports_row_labels_and_membership(self):
        df = pd.DataFrame(
            [
                {
                    "row_city1": "Accra",
                    "row_city2": "Berezniki",
                    "similarity": 0.9,
                    "left_in_city1": True,
                    "left_in_city2": False,
                    "right_in_city1": False,
                    "right_in_city2": True,
                },
                {
                    "row_city1": "Accra",
                    "row_city2": "Berezniki",
                    "similarity": 1.0,
                    "left_in_city1": True,
                    "left_in_city2": False,
                    "right_in_city1": True,
                    "right_in_city2": False,
                },
                {
                    "row_city1": "Accra",
                    "row_city2": "Berezniki",
                    "similarity": 1.0,
                    "left_in_city1": False,
                    "left_in_city2": True,
                    "right_in_city1": False,
                    "right_in_city2": True,
                },
                {
                    "row_city1": "Berezniki",
                    "row_city2": "Accra",
                    "similarity": 0.8,
                    "left_in_city1": False,
                    "left_in_city2": True,
                    "right_in_city1": True,
                    "right_in_city2": False,
                },
            ]
        )

        summary = self.module.summarize_shard_rows("Accra", "Berezniki", df)

        self.assertEqual(summary["city_1"], "Accra")
        self.assertEqual(summary["city_2"], "Berezniki")
        self.assertEqual(summary["pair_row_count"], 4)
        self.assertEqual(summary["row_label_match_count"], 3)
        self.assertEqual(summary["row_label_reversed_count"], 1)
        self.assertEqual(summary["strict_city1_to_city2_count"], 1)
        self.assertEqual(summary["strict_city2_to_city1_count"], 1)
        self.assertEqual(summary["same_city1_only_count"], 1)
        self.assertEqual(summary["same_city2_only_count"], 1)
        self.assertEqual(summary["sim_eq_1_count"], 2)
        self.assertEqual(summary["sim_eq_1_same_city1_only_count"], 1)
        self.assertEqual(summary["sim_eq_1_same_city2_only_count"], 1)


if __name__ == "__main__":
    unittest.main()
