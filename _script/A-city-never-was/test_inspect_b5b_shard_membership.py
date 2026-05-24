#!/usr/bin/env python3
"""
Regression tests for the B5b shard membership inspection script.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "inspect_b5b_shard_membership.py"


def load_module():
    if "duckdb" not in sys.modules:
        fake_duckdb = types.ModuleType("duckdb")
        fake_duckdb.connect = lambda *_args, **_kwargs: None
        sys.modules["duckdb"] = fake_duckdb

    spec = importlib.util.spec_from_file_location(
        "inspect_b5b_shard_membership", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestInspectB5bShardMembershipHelpers(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_classify_membership_side(self):
        self.assertEqual(
            self.module.classify_membership_side(True, False), "city1_only"
        )
        self.assertEqual(
            self.module.classify_membership_side(False, True), "city2_only"
        )
        self.assertEqual(self.module.classify_membership_side(True, True), "both")
        self.assertEqual(self.module.classify_membership_side(False, False), "neither")

    def test_summarize_shard_membership_df_reports_duplicates_and_mismatches(self):
        df = pd.DataFrame(
            [
                {
                    "hex_id1": "a1",
                    "hex_id2": "b1",
                    "similarity": 0.9,
                    "left_in_city1": True,
                    "left_in_city2": False,
                    "right_in_city1": False,
                    "right_in_city2": True,
                },
                {
                    "hex_id1": "a1",
                    "hex_id2": "b1",
                    "similarity": 0.8,
                    "left_in_city1": True,
                    "left_in_city2": False,
                    "right_in_city1": False,
                    "right_in_city2": True,
                },
                {
                    "hex_id1": "b2",
                    "hex_id2": "a2",
                    "similarity": 0.7,
                    "left_in_city1": False,
                    "left_in_city2": True,
                    "right_in_city1": True,
                    "right_in_city2": False,
                },
                {
                    "hex_id1": "a3",
                    "hex_id2": "x1",
                    "similarity": 1.0,
                    "left_in_city1": True,
                    "left_in_city2": False,
                    "right_in_city1": False,
                    "right_in_city2": False,
                },
                {
                    "hex_id1": "a4",
                    "hex_id2": "b4",
                    "similarity": 0.95,
                    "left_in_city1": True,
                    "left_in_city2": True,
                    "right_in_city1": False,
                    "right_in_city2": True,
                },
            ]
        )

        summary = self.module.summarize_shard_membership_df("Alpha", "Beta", df)

        self.assertEqual(summary["city_1"], "Alpha")
        self.assertEqual(summary["city_2"], "Beta")
        self.assertEqual(summary["pair_row_count"], 5)
        self.assertEqual(summary["unique_ordered_pair_count"], 4)
        self.assertEqual(summary["duplicate_ordered_pair_count"], 1)
        self.assertEqual(summary["unique_unordered_pair_count"], 4)
        self.assertEqual(summary["duplicate_unordered_pair_count"], 1)
        self.assertEqual(summary["expected_orientation_count"], 3)
        self.assertEqual(summary["reversed_orientation_count"], 1)
        self.assertEqual(summary["ambiguous_membership_count"], 1)
        self.assertEqual(summary["missing_membership_count"], 1)
        self.assertEqual(summary["strict_city1_to_city2_count"], 2)
        self.assertEqual(summary["strict_city2_to_city1_count"], 1)
        self.assertEqual(summary["sim_eq_1_count"], 1)
        self.assertEqual(summary["sim_eq_1_missing_membership_count"], 1)


if __name__ == "__main__":
    unittest.main()
