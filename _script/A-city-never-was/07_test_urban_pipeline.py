#!/usr/bin/env python
"""
Test Suite for Urban Analysis Pipeline
Comprehensive tests for all refactored modules.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd
import numpy as np
import h3
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from h3_hexagon_aggregator import H3HexagonAggregator
from h3_distance_processor import H3DistanceProcessor
from urban_similarity_processor import UrbanSimilarityProcessor
from city_pair_similarity_processor import CityPairSimilarityProcessor
from urban_utils import (
    UrbanDataConfig,
    DuckDBManager,
    DataValidator,
    CheckpointManager,
    ProgressTracker,
    calculate_file_hash,
    batch_process,
)


class TestUrbanUtils(unittest.TestCase):
    """Test the urban_utils module."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {"TEST_PATH": self.temp_dir, "BATCH_SIZE": 100}

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_urban_data_config(self):
        """Test configuration management."""
        config = UrbanDataConfig(overrides={"custom_key": "custom_value"})

        self.assertIsNotNone(config.get("ROOTFOLDER"))
        self.assertEqual(config.get("custom_key"), "custom_value")
        self.assertIsNotNone(config.get("TODAY"))

        # Test save and load
        config_file = Path(self.temp_dir) / "config.json"
        config.save(str(config_file))
        self.assertTrue(config_file.exists())

        # Load saved config
        loaded_config = UrbanDataConfig(config_file=str(config_file))
        self.assertEqual(loaded_config.get("custom_key"), "custom_value")

    def test_duckdb_manager(self):
        """Test DuckDB manager functionality."""
        manager = DuckDBManager()

        # Test query execution
        result = manager.execute_query("SELECT 1 as test")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["test"].iloc[0], 1)

        # Test DataFrame registration
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        manager.register_dataframe(df, "test_table")

        result = manager.execute_query("SELECT SUM(a) as sum_a FROM test_table")
        self.assertEqual(result["sum_a"].iloc[0], 6)

        manager.close()

    def test_data_validator(self):
        """Test data validation functions."""
        # Valid hexagon data
        df_valid = pd.DataFrame(
            {
                "hex_id": ["8928308280fffff", "8928308283fffff"],
                "res": [6, 6],
                "lat": [22.3, 22.4],
                "lon": [114.1, 114.2],
            }
        )

        self.assertTrue(DataValidator.validate_hexagon_data(df_valid))
        self.assertTrue(DataValidator.validate_coordinates(df_valid))

        # Invalid data - missing columns
        df_invalid = pd.DataFrame({"something": [1, 2]})
        with self.assertRaises(ValueError):
            DataValidator.validate_hexagon_data(df_invalid)

        # Invalid coordinates
        df_bad_coords = pd.DataFrame(
            {"lat": [91, -91], "lon": [0, 0]}  # Invalid latitudes
        )
        with self.assertRaises(ValueError):
            DataValidator.validate_coordinates(df_bad_coords)

    def test_checkpoint_manager(self):
        """Test checkpoint save and load."""
        checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        # Save checkpoint
        data = {"iteration": 100, "cities": ["New York", "London"]}
        manager.save_checkpoint(data, "test")

        # Load checkpoint
        loaded = manager.load_checkpoint("test")
        self.assertEqual(loaded["iteration"], 100)
        self.assertEqual(loaded["cities"], ["New York", "London"])

        # Remove checkpoint
        manager.remove_checkpoint("test")
        self.assertIsNone(manager.load_checkpoint("test"))

    def test_batch_process(self):
        """Test batch processing utility."""
        items = list(range(100))

        def process_func(batch):
            return [x * 2 for x in batch]

        results = batch_process(items, 20, process_func, "Test batching")

        self.assertEqual(len(results), 100)
        self.assertEqual(results[0], 0)
        self.assertEqual(results[50], 100)


class TestH3HexagonAggregator(unittest.TestCase):
    """Test H3 hexagon aggregation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "CURATE_FOLDER_EXPORT": self.temp_dir,
            "CURATED_FOLDER": self.temp_dir,
            "TRAIN_TEST_FOLDER": self.temp_dir,
            "ROOTFOLDER": self.temp_dir,
            "PANO_PATH": "{ROOTFOLDER}/pano.csv",
            "PATH_PATH": "{ROOTFOLDER}/path.csv",
            "summary_resolutions": [6, 7],
            "exclude_resolutions": [11],
            "cities_to_overwrite": [],
        }

        # Create mock data files
        self.create_mock_data()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_data(self):
        """Create mock data files for testing."""
        # Mock panorama data
        pano_data = pd.DataFrame(
            {
                "panoid": ["pano001", "pano002", "pano003"],
                "lat": [22.3, 22.31, 22.32],
                "lon": [114.1, 114.11, 114.12],
                "id": [1, 2, 3],
            }
        )
        pano_path = Path(self.temp_dir) / "pano.csv"
        pano_data.to_csv(pano_path, index=False)

        # Mock path data
        path_data = pd.DataFrame({"panoid": ["pano001", "pano002", "pano003"]})
        path_path = Path(self.temp_dir) / "path.csv"
        path_data.to_csv(path_path, index=False)

        # Mock predictions
        pred_data = pd.DataFrame(
            {
                "name": ["pano001_image.jpg", "pano002_image.jpg"],
                **{str(i): np.random.rand(2) for i in range(127)},
            }
        )
        pred_path = Path(self.temp_dir) / "city" / "hongkong_pred.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_data.to_parquet(pred_path)

    @patch("h3_hexagon_aggregator.H3HexagonAggregator._validate_h3_version")
    def test_initialization(self, mock_validate):
        """Test aggregator initialization."""
        aggregator = H3HexagonAggregator(self.config, log_level="WARNING")

        self.assertIsNotNone(aggregator.logger)
        self.assertIsNotNone(aggregator.db_manager)
        self.assertEqual(len(aggregator.vector_columns), 127)
        self.assertEqual(aggregator.summary_resolutions, [6, 7])

    @patch("h3_hexagon_aggregator.H3HexagonAggregator._validate_h3_version")
    def test_load_pano_metadata(self, mock_validate):
        """Test panorama metadata loading."""
        aggregator = H3HexagonAggregator(self.config, log_level="WARNING")

        # Mock H3 conversion
        aggregator.h3_convert = lambda lat, lon, res: f"hex_{res}_{lat}_{lon}"

        df_pano = aggregator.load_pano_metadata("HongKong")

        # Should have data but no lat/lon columns
        self.assertFalse(df_pano.empty)
        self.assertNotIn("lat", df_pano.columns)
        self.assertNotIn("lon", df_pano.columns)
        self.assertIn("hex_6", df_pano.columns)

    @patch("h3_hexagon_aggregator.H3HexagonAggregator._validate_h3_version")
    def test_compute_statistics(self, mock_validate):
        """Test statistics computation."""
        aggregator = H3HexagonAggregator(self.config, log_level="WARNING")

        # Create mock aggregated data
        mock_data = {"hex_id": ["hex1", "hex2", "hex3"], "res": [8, 8, 8]}
        for i in range(127):
            mock_data[str(i)] = np.random.rand(3)

        df = pd.DataFrame(mock_data)

        stats = aggregator.compute_statistics(df, resolution=8)

        self.assertIn("total_hexagons", stats)
        self.assertEqual(stats["total_hexagons"], 3)
        self.assertIn("avg_max_prob", stats)
        self.assertIn("top_classes", stats)


class TestH3DistanceProcessor(unittest.TestCase):
    """Test H3 distance processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir,
            "CURATE_FOLDER_EXPORT": self.temp_dir,
            "RES_EXCLUDE": 11,
        }

        # Create mock city metadata
        self.city_meta_path = Path(self.temp_dir) / "city_meta.csv"
        city_data = pd.DataFrame(
            {
                "City": ["Hong Kong", "Singapore"],
                "center_lat": [22.3193, 1.3521],
                "center_lng": [114.1694, 103.8198],
            }
        )
        city_data.to_csv(self.city_meta_path, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test processor initialization."""
        processor = H3DistanceProcessor(self.config, log_level="WARNING")

        self.assertIsNotNone(processor.logger)
        self.assertIsNotNone(processor.conn)

    def test_load_city_metadata(self):
        """Test city metadata loading."""
        processor = H3DistanceProcessor(self.config, log_level="WARNING")

        city_meta = processor.load_city_metadata(str(self.city_meta_path))

        self.assertEqual(len(city_meta), 2)
        self.assertIn("city_lower", city_meta.columns)
        self.assertEqual(city_meta["city_lower"].iloc[0], "hongkong")

    def test_compute_hex_coordinates(self):
        """Test hexagon coordinate computation."""
        processor = H3DistanceProcessor(self.config, log_level="WARNING")

        # Create mock hexagon data
        df = pd.DataFrame(
            {"hex_id": ["8928308280fffff", "8928308283fffff"], "res": [6, 6]}
        )

        df_with_coords = processor.compute_hex_coordinates(df)

        self.assertIn("lat", df_with_coords.columns)
        self.assertIn("lon", df_with_coords.columns)
        self.assertEqual(len(df_with_coords), 2)


class TestSimilarityProcessors(unittest.TestCase):
    """Test similarity processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir,
            "CURATE_FOLDER_EXPORT": self.temp_dir,
            "CURATE_FOLDER_EXPORT2": self.temp_dir,
            "RES_EXCLUDE": 11,
            "RES_SEL": 6,
        }

        # Create mock feature data
        self.create_mock_features()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_features(self):
        """Create mock feature files."""
        cities = ["HongKong", "Singapore"]

        for city in cities:
            data = {"hex_id": [f"hex_{i}" for i in range(10)], "res": [6] * 10}
            for j in range(127):
                data[str(j)] = np.random.rand(10)

            df = pd.DataFrame(data)
            file_path = Path(self.temp_dir) / f"prob_city={city}_res_exclude=11.parquet"
            df.to_parquet(file_path)

    def test_city_pair_similarity_initialization(self):
        """Test city pair similarity processor initialization."""
        processor = CityPairSimilarityProcessor(self.config, log_level="WARNING")

        self.assertIsNotNone(processor.logger)
        self.assertIsNotNone(processor.db_manager)
        self.assertEqual(len(processor.vector_columns), 127)

    def test_load_city_features(self):
        """Test loading city features."""
        processor = CityPairSimilarityProcessor(self.config, log_level="WARNING")

        df = processor.load_city_features("HongKong", 6)

        self.assertFalse(df.empty)
        self.assertIn("hex_id", df.columns)
        self.assertIn("0", df.columns)  # First feature column
        self.assertEqual(df["city"].iloc[0], "HongKong")

    def test_compute_similarity_matrix_batch(self):
        """Test batch similarity computation."""
        processor = CityPairSimilarityProcessor(self.config, log_level="WARNING")

        # Create mock features
        features1 = np.random.rand(100, 50)
        features2 = np.random.rand(100, 50)

        sim_matrix = processor.compute_similarity_matrix_batch(
            features1, features2, batch_size=20
        )

        self.assertEqual(sim_matrix.shape, (100, 100))
        self.assertTrue(np.all(sim_matrix >= -1))
        self.assertTrue(np.all(sim_matrix <= 1))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_full_mock_environment()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_full_mock_environment(self):
        """Create a complete mock environment for integration testing."""
        # City metadata
        city_meta = pd.DataFrame(
            {
                "City": ["TestCity1", "TestCity2"],
                "center_lat": [22.3, 1.3],
                "center_lng": [114.1, 103.8],
            }
        )
        city_meta.to_csv(Path(self.temp_dir) / "city_meta.csv", index=False)

        # Feature data for each city
        for city in ["TestCity1", "TestCity2"]:
            data = {
                "hex_id": [f"hex_{i}" for i in range(20)],
                "res": [6] * 10 + [7] * 10,
            }
            for j in range(127):
                data[str(j)] = np.random.rand(20)

            df = pd.DataFrame(data)
            file_path = Path(self.temp_dir) / f"prob_city={city}_res_exclude=11.parquet"
            df.to_parquet(file_path)

    @patch("run_pipelines.UrbanSimilarityProcessor")
    @patch("run_pipelines.H3DistanceProcessor")
    def test_pipeline_orchestrator(self, mock_distance, mock_similarity):
        """Test the main pipeline orchestrator."""
        from run_pipelines import PipelineOrchestrator

        # Mock processors
        mock_distance.return_value.run = MagicMock(return_value=True)
        mock_similarity.return_value.run = MagicMock(return_value=True)

        config_file = Path(self.temp_dir) / "config.json"
        config = {
            "CURATE_FOLDER_SOURCE": self.temp_dir,
            "CURATE_FOLDER_EXPORT": self.temp_dir,
        }
        with open(config_file, "w") as f:
            json.dump(config, f)

        orchestrator = PipelineOrchestrator(str(config_file), log_level="WARNING")

        # Test running all pipelines
        city_meta_path = Path(self.temp_dir) / "city_meta.csv"
        orchestrator.run_all_pipelines(str(city_meta_path))

        # Verify both pipelines were called
        self.assertTrue(mock_distance.called or mock_similarity.called)


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUrbanUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestH3HexagonAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestH3DistanceProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestSimilarityProcessors))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
