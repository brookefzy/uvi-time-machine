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
import json
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

            def fetchone(self):
                if self._df.empty:
                    return (0,)
                return tuple(self._df.iloc[0].tolist())

        class FakeConnection:
            def __init__(self):
                self.last_copy_query = None
                self.last_copy_output_path = None

            def execute(self, query):
                if "SELECT COUNT(*)" in query and "WITH shard_union AS" in query:
                    result = self._build_result(query)
                    if "city_1 != city_2" in query:
                        count = len(result[result["city_1"] != result["city_2"]])
                    else:
                        count = len(result[result["city_1"] == result["city_2"]])
                    return FakeResult(pd.DataFrame({"count": [count]}))

                if "COPY (" in query and "WITH shard_union AS" in query:
                    self.last_copy_query = query
                    inner_query, output_path, options = re.search(
                        r"COPY \((.*)\) TO '([^']+)'(?: \((.*)\))?",
                        query,
                        flags=re.S,
                    ).groups()
                    self.last_copy_output_path = Path(output_path)
                    result = self._build_result(inner_query)
                    inter_city = result[result["city_1"] != result["city_2"]].reset_index(
                        drop=True
                    )
                    if options and "FILE_SIZE_BYTES" in options:
                        output_dir = Path(output_path)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        midpoint = max(1, len(inter_city) // 2)
                        chunks = [inter_city.iloc[:midpoint], inter_city.iloc[midpoint:]]
                        for idx, chunk in enumerate(chunks):
                            if chunk.empty:
                                continue
                            pd.DataFrame.to_parquet(
                                chunk.reset_index(drop=True),
                                output_dir / f"part_{idx}.parquet",
                                index=False,
                            )
                    else:
                        pd.DataFrame.to_parquet(inter_city, output_path, index=False)
                    return FakeResult(pd.DataFrame())

                if "WITH shard_union AS" not in query:
                    return FakeResult(pd.DataFrame())

                return FakeResult(self._build_result(query))

            def _build_result(self, query):
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

                if not frames:
                    final_file_matches = re.findall(
                        r"FROM read_parquet\('([^']+)'\)",
                        query,
                        flags=re.S,
                    )
                    for path in final_file_matches:
                        frame = pd.read_parquet(path)[
                            ["hex_id1", "hex_id2", "similarity", "city1", "city2"]
                        ].copy()
                        frame = frame.rename(
                            columns={"city1": "shard_city1", "city2": "shard_city2"}
                        )
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
                return result[["hex_id1", "hex_id2", "similarity", "city_1", "city_2"]]

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
        self._orig_path_rename = Path.rename

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

        def fake_path_rename(path_obj, target):
            source = Path(path_obj)
            destination = Path(target)
            result = self._orig_path_rename(source, destination)
            moved_entries = {}
            for stored_path, frame in list(self.parquet_store.items()):
                try:
                    relative_path = stored_path.relative_to(source)
                except ValueError:
                    continue
                moved_entries[destination / relative_path] = frame
                del self.parquet_store[stored_path]
            self.parquet_store.update(moved_entries)
            return result

        pd.read_parquet = fake_read_parquet
        pd.DataFrame.to_parquet = fake_to_parquet
        Path.rename = fake_path_rename

    def tearDown(self):
        pd.read_parquet = self._orig_read_parquet
        pd.DataFrame.to_parquet = self._orig_to_parquet
        Path.rename = self._orig_path_rename
        self.temp_dir.cleanup()

    def make_processor(self, resolution=8, parquet_file_size="512MB"):
        config = {
            "CURATE_FOLDER_SOURCE": str(self.source_dir),
            "CURATE_FOLDER_EXPORT2": str(self.export_root),
            "EXPORT_FOLDER": str(self.output_dir),
            "RES_SEL": resolution,
            "RESUME": True,
            "PARQUET_FILE_SIZE_BYTES": parquet_file_size,
        }
        processor = self.module.OptimizedUrbanSimilarityProcessor(
            config, log_level="WARNING"
        )
        self.addCleanup(processor.close)
        return processor

    def read_output_dataset(self, output_path):
        output_path = Path(output_path)
        if output_path.is_dir():
            frames = [
                self.parquet_store[path]
                for path in sorted(self.parquet_store)
                if path.parent == output_path
            ]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.parquet_store[output_path].copy()

    def write_pair_shard(self, city1, city2, resolution, rows):
        pair_dir = self.temp_root / f"city1={city1}" / f"city2={city2}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(
            pair_dir / f"part_res={resolution}.parquet", index=False
        )

    def write_merged_city_file(self, city, resolution, rows):
        output_dir = self.export_root / "optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(
            output_dir / f"similarity_city={city}_res={resolution}_optimized.parquet",
            index=False,
        )

    def test_process_city_similarity_reads_temp_shards_and_saves_partitioned_intercity_output(self):
        processor = self.make_processor(resolution=8, parquet_file_size="64MB")
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
        self.assertTrue(output_file.is_dir())

        result = self.read_output_dataset(output_file)
        self.assertIsNotNone(processor.conn.last_copy_query)
        self.assertIn("FILE_SIZE_BYTES '64MB'", processor.conn.last_copy_query)
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

    def test_process_city_similarity_falls_back_to_merged_city_file_when_temp_shards_are_cleaned_up(self):
        processor = self.make_processor(resolution=8, parquet_file_size="0")
        self.write_merged_city_file(
            "Alpha",
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
                    "hex_id1": "a1",
                    "hex_id2": "a2",
                    "city1": "Alpha",
                    "city2": "Alpha",
                    "similarity": 0.7,
                },
            ],
        )

        inner_count, inter_count = processor.process_city_similarity("Alpha")

        self.assertEqual(inner_count, 1)
        self.assertEqual(inter_count, 1)

        output_file = (
            self.output_dir / "similarity_intracity_city=Alpha_res=8.parquet"
        )
        self.assertTrue(output_file.exists())

        result = self.read_output_dataset(output_file)
        self.assertEqual(list(result.columns), ["hex_id1", "hex_id2", "similarity", "city_1", "city_2"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["hex_id1"], "a1")
        self.assertEqual(result.iloc[0]["hex_id2"], "b1")
        self.assertEqual(result.iloc[0]["city_1"], "Alpha")
        self.assertEqual(result.iloc[0]["city_2"], "Beta")

    def test_run_resumes_by_skipping_existing_city_output(self):
        processor = self.make_processor(resolution=8)
        city_meta = self.root / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta"]}).to_csv(city_meta, index=False)
        existing_output = self.output_dir / "similarity_intracity_city=Alpha_res=8.parquet"
        existing_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"hex_id1": "a1", "hex_id2": "b1", "similarity": 0.5, "city_1": "Alpha", "city_2": "Beta"}]
        ).to_parquet(existing_output / "part_0.parquet", index=False)

        processed_cities = []

        def fake_process_city_similarity(city):
            processed_cities.append(city)
            return (0, 1)

        processor.process_city_similarity = fake_process_city_similarity
        processor.run(str(city_meta))

        self.assertEqual(processed_cities, ["Beta"])

    def test_run_writes_progress_file_for_completed_cities(self):
        processor = self.make_processor(resolution=8)
        progress_path = self.root / "_agg_progress.json"
        processor.config["AGG_PROGRESS_PATH"] = str(progress_path)
        city_meta = self.root / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta"]}).to_csv(city_meta, index=False)

        def fake_process_city_similarity(city):
            return (0, 1)

        processor.process_city_similarity = fake_process_city_similarity
        processor.run(str(city_meta))

        progress = json.loads(progress_path.read_text())
        self.assertEqual(progress["completed_cities"], ["Alpha", "Beta"])
        self.assertEqual(progress["pending_cities"], [])
        self.assertEqual(progress["status"], "completed")


if __name__ == "__main__":
    unittest.main()
