#!/usr/bin/env python3
"""
Regression tests for purge_old_temp_shards.py.
"""

import importlib.util
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "purge_old_temp_shards.py"


def load_module():
    spec = importlib.util.spec_from_file_location("purge_old_temp_shards", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPurgeOldTempShards(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_root = Path(self.temp_dir.name) / "optimized" / "temp"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def touch_with_mtime(self, path: Path, dt: datetime) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        timestamp = dt.timestamp()
        path.chmod(0o644)
        import os

        os.utime(path, (timestamp, timestamp))

    def test_collect_purge_targets_only_includes_files_before_cutoff(self):
        old_file = self.temp_root / "city1=Accra" / "city2=Berezniki" / "part_res=8.parquet"
        new_file = self.temp_root / "city1=Accra" / "city2=Boston" / "part_res=8.parquet"
        self.touch_with_mtime(old_file, datetime(2025, 5, 17, 12, 0, 0))
        self.touch_with_mtime(new_file, datetime(2025, 5, 24, 12, 0, 0))

        targets = self.module.collect_purge_targets(self.temp_root, date(2025, 5, 24))

        self.assertEqual(targets, [old_file.resolve()])

    def test_purge_targets_dry_run_leaves_files_in_place(self):
        old_file = self.temp_root / "city1=Accra" / "city2=Berezniki" / "part_res=8.parquet"
        self.touch_with_mtime(old_file, datetime(2025, 5, 17, 12, 0, 0))
        targets = [old_file]

        summary = self.module.purge_targets(targets, apply=False)

        self.assertTrue(old_file.exists())
        self.assertEqual(summary["files_matched"], 1)
        self.assertEqual(summary["files_deleted"], 0)

    def test_purge_targets_apply_deletes_files_and_empty_dirs(self):
        old_file = self.temp_root / "city1=Accra" / "city2=Berezniki" / "part_res=8.parquet"
        sibling_file = self.temp_root / "city1=Accra" / "city2=Lagos" / "part_res=8.parquet"
        self.touch_with_mtime(old_file, datetime(2025, 5, 17, 12, 0, 0))
        self.touch_with_mtime(sibling_file, datetime(2025, 5, 30, 12, 0, 0))
        old_city2_dir = old_file.parent
        old_city1_dir = old_city2_dir.parent

        summary = self.module.purge_targets([old_file], apply=True)

        self.assertFalse(old_file.exists())
        self.assertFalse(old_city2_dir.exists())
        self.assertTrue(old_city1_dir.exists())
        self.assertTrue(sibling_file.exists())
        self.assertEqual(summary["files_deleted"], 1)
        self.assertGreaterEqual(summary["directories_deleted"], 1)

    def test_validate_temp_root_rejects_non_optimized_temp_path(self):
        bad_root = Path(self.temp_dir.name) / "wrong-root"
        bad_root.mkdir()

        with self.assertRaisesRegex(ValueError, "optimized/temp"):
            self.module.validate_temp_root(bad_root)

    def test_scan_stats_reports_total_oldest_and_newest_dates(self):
        old_file = self.temp_root / "city1=Accra" / "city2=Berezniki" / "part_res=8.parquet"
        new_file = self.temp_root / "city1=Accra" / "city2=Boston" / "part_res=8.parquet"
        self.touch_with_mtime(old_file, datetime(2025, 5, 17, 12, 0, 0))
        self.touch_with_mtime(new_file, datetime(2026, 5, 24, 12, 0, 0))

        stats = self.module.scan_stats(self.temp_root)

        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["oldest_date"], date(2025, 5, 17))
        self.assertEqual(stats["newest_date"], date(2026, 5, 24))


if __name__ == "__main__":
    unittest.main()
