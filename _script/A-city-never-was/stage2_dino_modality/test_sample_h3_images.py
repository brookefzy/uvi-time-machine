import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from sample_similar_pairs.common import CityVectors


SCRIPT_PATH = Path(__file__).with_name("01_sample_h3_images.py")


def load_script():
    spec = importlib.util.spec_from_file_location("sample_h3_images", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_city_is_repeatable_and_audits_eligible_h3_rows():
    module = load_script()
    metadata = pd.DataFrame(
        {
            "name": ["z.jpg", "a.jpg", "b.jpg", "outside.jpg"],
            "panoid": ["z", "a", "b", "o"],
            "hex_id": ["h1", "h1", "h2", "h3"],
            "lat": [1.0] * 4,
            "lon": [2.0] * 4,
        }
    )
    city = CityVectors("Paris", metadata, ["e_0000", "e_0001"], np.eye(4, 2, dtype=np.float32))

    sampled, audit = module.sample_city(city, max_images_per_h3=2, before_rows=6)

    assert sampled.metadata["name"].tolist() == ["a.jpg", "z.jpg", "b.jpg", "outside.jpg"]
    assert audit == {"before_rows": 6, "eligible_rows": 4, "sampled_rows": 4, "undersupplied_h3_count": 2}


def test_parser_exposes_required_sampling_contract_options():
    parser = load_script().build_parser()
    for option in ("--city", "--embedding-root", "--rootfolder", "--train-test-folder", "--res-exclude", "--min-year", "--max-year", "--h3-resolution", "--max-images-per-h3", "--output"):
        assert option in parser._option_string_actions


def test_script_runs_directly_from_its_stage_directory():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=SCRIPT_PATH.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
