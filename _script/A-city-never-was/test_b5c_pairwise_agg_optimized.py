#!/usr/bin/env python3
"""
Regression tests for B5c_pairwise_agg_optimized.py.
"""

import importlib.util
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parent / "B5c_pairwise_agg_optimized.py"
)


def load_module():
    if "duckdb" not in sys.modules:
        fake_duckdb = types.ModuleType("duckdb")

        class FakeResult:
            def __init__(self, df):
                self._df = df

            def fetchdf(self):
                return self._df

        class FakeConnection:
            def execute(self, query):
                if "WITH shard_union AS" not in query:
                    return FakeResult(pd.DataFrame())

                matches = re.findall(
                    r"'([^']+)' AS shard_city1,\s*'([^']+)' AS shard_city2\s*FROM read_parquet\('([^']+)'\)",
                    query,
                    flags=re.S,
                )

                frames = []
                for city1, city2, path in matches:
                    frame = pd.read_parquet(path)[["hex_id1", "hex_id2", "similarity"]].copy()
                    frame["shard_city1"] = city1
                    frame["shard_city2"] = city2
                    frames.append(frame)

                shard_union = pd.concat(frames, ignore_index=True)
                shard_union["hex_id1_norm"] = shard_union[["hex_id1", "hex_id2"]].min(axis=1)
                shard_union["hex_id2_norm"] = shard_union[["hex_id1", "hex_id2"]].max(axis=1)

                result = (
                    shard_union.groupby(
                        ["hex_id1_norm", "hex_id2_norm", "shard_city1", "shard_city2"],
                        as_index=False,
                    )["similarity"]
                    .max()
                    .rename(
                        columns={
                            "hex_id1_norm": "hex_id1",
                            "hex_id2_norm": "hex_id2",
                            "shard_city1": "city_1",
                            "shard_city2": "city_2",
                        }
                    )
                    .sort_values(["similarity", "hex_id1", "hex_id2"], ascending=[False, True, True])
                    .reset_index(drop=True)
                )
                return FakeResult(result[["hex_id1", "hex_id2", "similarity", "city_1", "city_2"]])

            def close(self):
                return None

        fake_duckdb.connect = lambda *_args, **_kwargs: FakeConnection()
        sys.modules["duckdb"] = fake_duckdb

    if "tqdm" not in sys.modules:
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
        sys.modules["tqdm"] = fake_tqdm

    spec = importlib.util.spec_from_file_location("b5c_optimized", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestOptimizedPairwiseAggregation(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_dir = self.root / "source"
        self.export_root = self.root / "pairwise"
        self.temp_root = self.export_root / "optimized" / "temp"
        self.output_dir = self.root / "agg"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_store = {}
        self._orig_read_parquet = pd.read_parquet
        self._orig_to_parquet = pd.DataFrame.to_parquet

        def fake_read_parquet(path, *args, **kwargs):
            path_obj = Path(path)
            if path_obj in self.parquet_store:
                return self.parquet_store[path_obj].copy()
            return self._orig_read_parquet(path, *args, **kwargs)

        def fake_to_parquet(df, path, *args, **kwargs):
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            self.parquet_store[path_obj] = df.copy()
            path_obj.touch()

        pd.read_parquet = fake_read_parquet
        pd.DataFrame.to_parquet = fake_to_parquet

    def tearDown(self):
        pd.read_parquet = self._orig_read_parquet
        pd.DataFrame.to_parquet = self._orig_to_parquet
        self.temp_dir.cleanup()

    def make_processor(self, resolution=8):
        config = {
            "CURATE_FOLDER_SOURCE": str(self.source_dir),
            "CURATE_FOLDER_EXPORT2": str(self.export_root),
            "EXPORT_FOLDER": str(self.output_dir),
            "RES_SEL": resolution,
        }
        processor = self.module.OptimizedUrbanSimilarityProcessor(
            config, log_level="WARNING"
        )
        self.addCleanup(processor.close)
        return processor

    def write_pair_shard(self, city1, city2, resolution, rows):
        pair_dir = self.temp_root / f"city1={city1}" / f"city2={city2}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(
            pair_dir / f"part_res={resolution}.parquet", index=False
        )

    def test_process_city_similarity_reads_temp_shards_and_saves_intercity_output(self):
        processor = self.make_processor(resolution=8)
        self.write_pair_shard(
            "Alpha",
            "Beta",
            8,
            [
                {
                    "hex_id1": "a1",
                    "hex_id2": "b1",
                    "city1": "Alpha",
                    "city2": "Beta",
                    "similarity": 0.8,
                },
                {
                    "hex_id1": "b1",
                    "hex_id2": "a1",
                    "city1": "Beta",
                    "city2": "Alpha",
                    "similarity": 0.9,
                },
            ],
        )
        self.write_pair_shard(
            "Alpha",
            "Alpha",
            8,
            [
                {
                    "hex_id1": "a1",
                    "hex_id2": "a2",
                    "city1": "Alpha",
                    "city2": "Alpha",
                    "similarity": 0.7,
                }
            ],
        )

        inner_count, inter_count = processor.process_city_similarity("Alpha")

        self.assertEqual(inner_count, 1)
        self.assertEqual(inter_count, 1)

        output_file = (
            self.output_dir / "similarity_intracity_city=Alpha_res=8.parquet"
        )
        self.assertTrue(output_file.exists())

        result = pd.read_parquet(output_file)
        self.assertEqual(list(result.columns), ["hex_id1", "hex_id2", "similarity", "city_1", "city_2"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["hex_id1"], "a1")
        self.assertEqual(result.iloc[0]["hex_id2"], "b1")
        self.assertEqual(result.iloc[0]["similarity"], 0.9)
        self.assertEqual(result.iloc[0]["city_1"], "Alpha")
        self.assertEqual(result.iloc[0]["city_2"], "Beta")

    def test_process_city_similarity_skips_missing_temp_shards_cleanly(self):
        processor = self.make_processor(resolution=8)

        inner_count, inter_count = processor.process_city_similarity("MissingCity")

        self.assertEqual(inner_count, 0)
        self.assertEqual(inter_count, 0)
        self.assertFalse(
            (self.output_dir / "similarity_intracity_city=MissingCity_res=8.parquet").exists()
        )


if __name__ == "__main__":
    unittest.main()
