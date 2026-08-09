#!/usr/bin/env python3
"""
Regression tests for B5c_pairwise_agg_optimized.py.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
import json
from pathlib import Path

import pandas as pd
import duckdb as real_duckdb


MODULE_PATH = (
    Path(__file__).resolve().parent / "B5c_pairwise_agg_optimized.py"
)


def load_module():
    previous = {name: sys.modules.get(name) for name in ("duckdb", "pandas", "tqdm")}
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules["duckdb"] = real_duckdb
    sys.modules["pandas"] = pd
    sys.modules["tqdm"] = fake_tqdm
    try:
        spec = importlib.util.spec_from_file_location("b5c_optimized", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


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

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_processor(self, resolution=8, parquet_file_size="512MB"):
        for city, hex_ids in {"Alpha": ["a1", "a2"], "Beta": ["b1", "b2"]}.items():
            pd.DataFrame({"hex_id": hex_ids, "res": [resolution] * len(hex_ids)}).to_parquet(
                self.source_dir / f"membership_city={city}.parquet", index=False
            )
        config = {
            "CURATE_FOLDER_SOURCE": str(self.source_dir),
            "CURATE_FOLDER_EXPORT2": str(self.export_root),
            "EXPORT_FOLDER": str(self.output_dir),
            "RES_SEL": resolution,
            "RESUME": True,
            "PARQUET_FILE_SIZE_BYTES": parquet_file_size,
            "H3_MEMBERSHIP_ROOT": str(self.source_dir),
            "H3_INPUT_TEMPLATE": "membership_city={city}.parquet",
        }
        processor = self.module.OptimizedUrbanSimilarityProcessor(
            config, log_level="WARNING"
        )
        processor.load_h3_membership(["Alpha", "Beta"])
        self.addCleanup(processor.close)
        return processor

    def read_output_dataset(self, output_path):
        output_path = Path(output_path)
        if output_path.is_dir():
            frames = [pd.read_parquet(path) for path in sorted(output_path.glob("*.parquet"))]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return pd.read_parquet(output_path)

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
        self.assertTrue(output_file.is_dir())
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

    def test_dinov3_temp_shards_with_metadata_export_intercity_rows_only(self):
        processor = self.make_processor(resolution=8, parquet_file_size="0")
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
                    "similarity": 0.95,
                    "metric": "cosine",
                    "model_name": "dinov3-test",
                },
                {
                    "hex_id1": "a2",
                    "hex_id2": "b2",
                    "city1": "Alpha",
                    "city2": "Beta",
                    "similarity": 0.50,
                    "metric": "cosine",
                    "model_name": "dinov3-test",
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
                    "similarity": 0.99,
                    "metric": "cosine",
                    "model_name": "dinov3-test",
                }
            ],
        )

        inner_count, inter_count = processor.process_city_similarity("Alpha")

        self.assertEqual(inner_count, 1)
        self.assertEqual(inter_count, 2)
        result = self.read_output_dataset(
            self.output_dir / "similarity_intracity_city=Alpha_res=8.parquet"
        )
        self.assertEqual(list(result.columns), ["hex_id1", "hex_id2", "similarity", "city_1", "city_2"])
        self.assertEqual(len(result), 2)
        self.assertTrue((result["city_1"] != result["city_2"]).all())

    def test_dinov3_merged_optimized_file_exports_intercity_rows_only_after_temp_cleanup(self):
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
                    "similarity": 0.95,
                    "metric": "cosine",
                    "model_name": "dinov3-test",
                },
                {
                    "hex_id1": "a1",
                    "hex_id2": "a2",
                    "city1": "Alpha",
                    "city2": "Alpha",
                    "similarity": 0.99,
                    "metric": "cosine",
                    "model_name": "dinov3-test",
                },
            ],
        )

        inner_count, inter_count = processor.process_city_similarity("Alpha")

        self.assertEqual(inner_count, 1)
        self.assertEqual(inter_count, 1)
        result = self.read_output_dataset(
            self.output_dir / "similarity_intracity_city=Alpha_res=8.parquet"
        )
        self.assertEqual(list(result.columns), ["hex_id1", "hex_id2", "similarity", "city_1", "city_2"])
        self.assertEqual(len(result), 1)
        self.assertTrue((result["city_1"] != result["city_2"]).all())

    def test_run_resumes_by_skipping_existing_city_output(self):
        processor = self.make_processor(resolution=8)
        city_meta = self.root / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta"]}).to_csv(city_meta, index=False)
        existing_output = self.output_dir / "similarity_intracity_city=Alpha_res=8.parquet"
        existing_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"hex_id1": "a1", "hex_id2": "b1", "similarity": 0.5, "city_1": "Alpha", "city_2": "Beta"}]
        ).to_parquet(existing_output / "part_0.parquet", index=False)
        (self.output_dir / "_audit_res=8.json").write_text(
            json.dumps(
                {
                    "validation_contract": self.module.VALIDATION_CONTRACT,
                    "resolution": 8,
                    "by_city": {"Alpha": {"emitted_rows": 1}},
                }
            )
        )

        processed_cities = []

        def fake_process_city_similarity(city):
            processed_cities.append(city)
            return (0, 1)

        processor.process_city_similarity = fake_process_city_similarity
        processor.run(str(city_meta))

        self.assertEqual(processed_cities, ["Beta"])

    def test_run_does_not_resume_from_legacy_unvalidated_progress_or_outputs(self):
        processor = self.make_processor(resolution=8)
        city_meta = self.root / "city_meta.csv"
        pd.DataFrame({"City": ["Alpha", "Beta"]}).to_csv(city_meta, index=False)
        processor.config["AGG_PROGRESS_PATH"] = str(self.root / "legacy_progress.json")
        Path(processor.config["AGG_PROGRESS_PATH"]).write_text(
            json.dumps(
                {
                    "resolution": 8,
                    "completed_cities": ["Alpha", "Beta"],
                    "pending_cities": [],
                    "status": "completed",
                }
            )
        )
        for city in ("Alpha", "Beta"):
            output = self.output_dir / f"similarity_intracity_city={city}_res=8.parquet"
            output.mkdir(parents=True)
            pd.DataFrame(
                [{"hex_id1": "a1", "hex_id2": "b1", "similarity": 0.5, "city_1": "Alpha", "city_2": "Beta"}]
            ).to_parquet(output / "part_0.parquet", index=False)

        processed_cities = []
        processor.process_city_similarity = lambda city: (processed_cities.append(city) or (0, 1))

        processor.run(str(city_meta))

        self.assertEqual(processed_cities, ["Alpha", "Beta"])

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
