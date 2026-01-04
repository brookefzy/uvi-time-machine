#!/usr/bin/env python
"""
Urban Analysis Pipeline Orchestrator
Main script to orchestrate urban similarity and distance processing pipelines.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

from urban_similarity_processor import UrbanSimilarityProcessor
from h3_distance_processor import H3DistanceProcessor
from urban_utils import UrbanDataConfig, CheckpointManager


class PipelineOrchestrator:
    """Orchestrate multiple urban analysis pipelines."""

    def __init__(self, config_file: str = None, log_level: str = "INFO"):
        """
        Initialize the orchestrator.

        Args:
            config_file: Optional path to configuration file
            log_level: Logging level
        """
        self.config = UrbanDataConfig(config_file)
        self.setup_logging(log_level)
        self.checkpoint_manager = CheckpointManager()

    def setup_logging(self, log_level: str) -> None:
        """Setup logging for the orchestrator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_orchestrator_{timestamp}.log"

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline orchestrator initialized. Log: {log_file}")

    def run_similarity_pipeline(
        self, city_meta_path: str, resume: bool = False
    ) -> bool:
        """
        Run the urban similarity processing pipeline.

        Args:
            city_meta_path: Path to city metadata CSV
            resume: Whether to resume from checkpoint

        Returns:
            Success status
        """
        self.logger.info("Starting Urban Similarity Pipeline")

        try:
            # Check for checkpoint
            if resume:
                checkpoint = self.checkpoint_manager.load_checkpoint("similarity")
                if checkpoint:
                    self.logger.info(
                        f"Resuming from checkpoint: {checkpoint.get('last_city', 'unknown')}"
                    )

            # Configure similarity processor
            similarity_config = {
                "ROOTFOLDER": self.config.get("ROOTFOLDER"),
                "CURATE_FOLDER_SOURCE": self.config.get("CURATE_FOLDER_SOURCE"),
                "CURATE_FOLDER_EXPORT": self.config.get("CURATE_FOLDER_EXPORT"),
                "CURATE_FOLDER_EXPORT2": self.config.get(
                    "CURATE_FOLDER_EXPORT2",
                    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair",
                ),
                "RES_EXCLUDE": self.config.get("RES_EXCLUDE", 11),
                "RES_SEL": self.config.get("RES_SEL", 7),
                "EXPORT_FOLDER": f"/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_{self.config.get('TODAY')}",
            }

            processor = UrbanSimilarityProcessor(similarity_config)
            processor.run(city_meta_path)

            self.logger.info("Urban Similarity Pipeline completed successfully")

            # Remove checkpoint on success
            if resume:
                self.checkpoint_manager.remove_checkpoint("similarity")

            return True

        except Exception as e:
            self.logger.error(f"Similarity pipeline failed: {e}", exc_info=True)
            return False

    def run_distance_pipeline(
        self, city_meta_path: str, resolutions: list = None
    ) -> bool:
        """
        Run the H3 distance processing pipeline.

        Args:
            city_meta_path: Path to city metadata CSV
            resolutions: List of H3 resolutions to process

        Returns:
            Success status
        """
        self.logger.info("Starting H3 Distance Pipeline")

        try:
            # Configure distance processor
            distance_config = {
                "RES_EXCLUDE": self.config.get("RES_EXCLUDE", 11),
                "CURATE_FOLDER_SOURCE": self.config.get("CURATE_FOLDER_SOURCE"),
                "CURATE_FOLDER_EXPORT": self.config.get("CURATE_FOLDER_EXPORT"),
            }

            processor = H3DistanceProcessor(distance_config)
            processor.run(
                city_meta_path=city_meta_path,
                resolutions=resolutions or [6, 7],
                compute_pairwise=True,
            )

            self.logger.info("H3 Distance Pipeline completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Distance pipeline failed: {e}", exc_info=True)
            return False

    def run_all_pipelines(self, city_meta_path: str) -> None:
        """
        Run all pipelines in sequence.

        Args:
            city_meta_path: Path to city metadata CSV
        """
        self.logger.info("Starting all urban analysis pipelines")

        # Pipeline execution order
        pipelines = [
            ("H3 Distance", self.run_distance_pipeline),
            ("Urban Similarity", self.run_similarity_pipeline),
        ]

        results = {}
        for name, pipeline_func in pipelines:
            self.logger.info(f"Executing pipeline: {name}")
            success = pipeline_func(city_meta_path)
            results[name] = "Success" if success else "Failed"

            if not success:
                self.logger.warning(
                    f"Pipeline {name} failed, continuing with others..."
                )

        # Summary
        self.logger.info("Pipeline execution summary:")
        for name, status in results.items():
            self.logger.info(f"  {name}: {status}")

        # Save results
        results_file = (
            Path("logs") / f"pipeline_results_{self.config.get('TIMESTAMP')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {results_file}")


def main():
    """Main entry point with CLI interface."""

    parser = argparse.ArgumentParser(description="Urban Analysis Pipeline Orchestrator")

    parser.add_argument(
        "--pipeline",
        choices=["similarity", "distance", "all"],
        default="all",
        help="Which pipeline to run",
    )

    parser.add_argument(
        "--city-meta", default="../city_meta.csv", help="Path to city metadata CSV file"
    )

    parser.add_argument("--config", help="Path to configuration JSON file")

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[6, 7],
        help="H3 resolutions to process (for distance pipeline)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )

    args = parser.parse_args()

    # Validate city metadata file exists
    if not Path(args.city_meta).exists():
        print(f"Error: City metadata file not found: {args.city_meta}")
        sys.exit(1)

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        config_file=args.config, log_level=args.log_level
    )

    # Run selected pipeline(s)
    if args.pipeline == "similarity":
        success = orchestrator.run_similarity_pipeline(args.city_meta, args.resume)
    elif args.pipeline == "distance":
        success = orchestrator.run_distance_pipeline(args.city_meta, args.resolutions)
    else:  # all
        orchestrator.run_all_pipelines(args.city_meta)
        success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
