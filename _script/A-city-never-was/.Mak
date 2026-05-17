# Makefile for Urban Analysis Pipeline
# Provides convenient commands for common development and deployment tasks

.PHONY: help install test clean format lint check run-all run-similarity run-distance run-aggregation profile docker

# Default target
help:
	@echo "Urban Analysis Pipeline - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install all dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-gpu      - Install with GPU support"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test            - Run all tests"
	@echo "  make test-coverage   - Run tests with coverage report"
	@echo "  make lint            - Run code linting"
	@echo "  make format          - Format code with black"
	@echo "  make check           - Run all checks (lint + test)"
	@echo ""
	@echo "Running Pipelines:"
	@echo "  make run-all         - Run all pipelines"
	@echo "  make run-similarity  - Run similarity processing"
	@echo "  make run-distance    - Run distance processing"
	@echo "  make run-aggregation - Run H3 aggregation"
	@echo "  make run-optimized   - Run optimized processor"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           - Clean temporary files and caches"
	@echo "  make profile         - Profile code performance"
	@echo "  make docker          - Build Docker container"
	@echo "  make docs            - Generate documentation"
	@echo "  make benchmark       - Run performance benchmarks"

# Installation targets
install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed successfully"

install-dev: install
	pip install pytest pytest-cov black flake8 mypy jupyter
	@echo "✓ Development dependencies installed"

install-gpu: install
	pip install cupy-cuda11x  # Adjust for your CUDA version
	@echo "✓ GPU support installed"

# Testing targets
test:
	@echo "Running unit tests..."
	python -m pytest test_urban_pipeline.py -v

test-coverage:
	@echo "Running tests with coverage..."
	python -m pytest test_urban_pipeline.py --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality targets
lint:
	@echo "Running linting checks..."
	flake8 *.py --max-line-length=120 --exclude=test_*.py
	mypy *.py --ignore-missing-imports

format:
	@echo "Formatting code with black..."
	black *.py --line-length=120

check: lint test
	@echo "✓ All checks passed"

# Pipeline execution targets
run-all:
	python run_pipelines.py --pipeline all --city-meta ../city_meta.csv

run-similarity:
	python urban_similarity_processor.py

run-distance:
	python h3_distance_processor.py

run-aggregation:
	python h3_hexagon_aggregator.py --city all

run-optimized:
	python optimized_similarity_processor.py --batch-size 10000

run-city-pairs:
	python city_pair_similarity_processor.py --res-sel 6 --parallel --workers 4

# Run with specific city
run-city:
	@read -p "Enter city name: " city; \
	python h3_hexagon_aggregator.py --city "$$city"

# Utility targets
clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ .pytest_cache *.pyc
	rm -rf logs/*.log
	rm -rf checkpoints/*
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"

profile:
	@echo "Profiling code performance..."
	python -m cProfile -o profile.stats run_pipelines.py --pipeline similarity
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Docker targets
docker:
	docker build -t urban-analysis:latest .

docker-run:
	docker run -it --rm \
		-v $$(pwd)/data:/data \
		-v $$(pwd)/outputs:/outputs \
		urban-analysis:latest

# Documentation
docs:
	@echo "Generating documentation..."
	python -m pydoc -w urban_similarity_processor
	python -m pydoc -w h3_distance_processor
	python -m pydoc -w h3_hexagon_aggregator
	python -m pydoc -w city_pair_similarity_processor
	python -m pydoc -w urban_utils
	@echo "✓ Documentation generated as HTML files"

# Benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	python -c "import timeit; \
		setup = 'from urban_utils import DuckDBManager'; \
		print('DuckDB query time:', timeit.timeit('manager = DuckDBManager(); manager.execute_query(\"SELECT 1\")', setup=setup, number=1000)/1000, 'seconds')"

# Development helpers
watch:
	@echo "Watching for file changes..."
	@while true; do \
		inotifywait -e modify *.py; \
		make test; \
	done

# Create necessary directories
setup-dirs:
	mkdir -p logs checkpoints data outputs
	@echo "✓ Directories created"

# Initial setup for new developers
setup: install-dev setup-dirs
	@echo "✓ Development environment ready"
	@echo "Run 'make help' to see available commands"

# Run with custom config
run-config:
	@read -p "Enter config file path: " config; \
	python run_pipelines.py --config "$$config" --pipeline all

# Monitor logs
monitor:
	tail -f logs/*.log

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	@python -c "import sys; print('Python:', sys.version)"
	@python -c "import pandas; print('Pandas:', pandas.__version__)"
	@python -c "import numpy; print('NumPy:', numpy.__version__)"
	@python -c "import duckdb; print('DuckDB:', duckdb.__version__)"
	@python -c "import h3; print('H3:', h3.__version__)"
	@python -c "try: import cupy; print('CuPy:', cupy.__version__); except: print('CuPy: Not installed')"
	@echo "✓ System check complete"

# Performance comparison
compare:
	@echo "Running performance comparison..."
	@echo "Original version (if available):"
	time python original_script.py 2>/dev/null || echo "Original script not found"
	@echo "\nRefactored version:"
	time python urban_similarity_processor.py
	@echo "✓ Comparison complete"

# Generate sample config
sample-config:
	@echo "Generating sample configuration..."
	@cat > sample_config.json <<EOF
	{
	  "ROOTFOLDER": "/path/to/data",
	  "CURATE_FOLDER_SOURCE": "/path/to/source",
	  "CURATE_FOLDER_EXPORT": "/path/to/output",
	  "RES_EXCLUDE": 11,
	  "RES_SEL": 6,
	  "memory_limit": "16GB",
	  "batch_size": 5000,
	  "similarity_threshold": 0.1,
	  "use_gpu": false,
	  "use_approximation": false
	}
	EOF
	@echo "✓ Sample config created as sample_config.json"

# Quick start for new users
quickstart: setup sample-config
	@echo ""
	@echo "========================================="
	@echo "Urban Analysis Pipeline - Quick Start"
	@echo "========================================="
	@echo ""
	@echo "1. Edit sample_config.json with your paths"
	@echo "2. Run 'make test' to verify installation"
	@echo "3. Run 'make run-all' to process all cities"
	@echo ""
	@echo "For more options, run 'make help'"