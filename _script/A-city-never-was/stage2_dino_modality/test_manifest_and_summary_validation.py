import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_summary():
    script = Path(__file__).with_name("08_summarize_mode_citypairs.py")
    spec = importlib.util.spec_from_file_location("summary", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_summary_rejects_missing_expected_city_pairs():
    frame = pd.DataFrame({"city_1":["A"],"city_2":["B"],"js_similarity":[.5]})
    with pytest.raises(ValueError, match="expected 3"):
        load_summary().validate_expected_pairs(frame, ["A", "B", "C"])
