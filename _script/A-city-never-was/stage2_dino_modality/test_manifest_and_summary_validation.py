import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_summary():
    script = Path(__file__).with_name("08_summarize_mode_citypairs.py")
    spec = importlib.util.spec_from_file_location("summary", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_manifest():
    script = Path(__file__).parents[1] / "slurm" / "generate_dinov3_mode_pair_manifest.py"
    spec = importlib.util.spec_from_file_location("manifest", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_summary_rejects_missing_expected_city_pairs():
    frame = pd.DataFrame({"city_1":["A"],"city_2":["B"],"js_similarity":[.5]})
    with pytest.raises(ValueError, match="expected 3"):
        load_summary().validate_expected_pairs(frame, ["A", "B", "C"])


def test_manifest_can_audit_and_skip_absent_histogram_cities(tmp_path):
    root = tmp_path / "histograms"
    root.mkdir()
    valid = pd.DataFrame({"city": ["A"], "hex_id": ["h"], "res": [8], "mode_id": [0], "mode_image_count": [1], "sampled_image_count": [1], "mode_fraction": [1.0], "model_id": ["m"]})
    valid.to_parquet(root / "city=A.parquet", index=False)
    valid.assign(city="C").to_parquet(root / "city=C.parquet", index=False)

    available, skipped = load_manifest().collect_histogram_cities(
        ["A", "B", "C"], root, allow_missing=True
    )

    assert available == ["A", "C"]
    assert skipped == ["B"]
    assert load_manifest().city_pairs(available) == [("A", "C")]


def test_manifest_remains_strict_by_default_for_missing_histograms(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing histogram for city A"):
        load_manifest().collect_histogram_cities(["A"], tmp_path, allow_missing=False)


def test_manifest_does_not_skip_an_existing_invalid_histogram(tmp_path):
    pd.DataFrame({"city": ["A"], "res": [7], "model_id": ["m"]}).to_parquet(
        tmp_path / "city=A.parquet", index=False
    )

    with pytest.raises(ValueError, match="invalid histogram"):
        load_manifest().collect_histogram_cities(["A"], tmp_path, allow_missing=True)


def test_manifest_rejects_present_histogram_with_invalid_sparse_fractions(tmp_path):
    pd.DataFrame({"city": ["A"], "hex_id": ["h"], "res": [8], "mode_id": [0], "mode_image_count": [1], "sampled_image_count": [1], "mode_fraction": [.5], "model_id": ["m"]}).to_parquet(
        tmp_path / "city=A.parquet", index=False
    )

    with pytest.raises(ValueError, match="sum to one"):
        load_manifest().collect_histogram_cities(["A"], tmp_path, allow_missing=True)


def test_manifest_rejects_histogram_whose_city_does_not_match_filename(tmp_path):
    pd.DataFrame({"city": ["B"], "hex_id": ["h"], "res": [8], "mode_id": [0], "mode_image_count": [1], "sampled_image_count": [1], "mode_fraction": [1.0], "model_id": ["m"]}).to_parquet(
        tmp_path / "city=A.parquet", index=False
    )

    with pytest.raises(ValueError, match="does not contain exactly city A"):
        load_manifest().collect_histogram_cities(["A"], tmp_path, allow_missing=True)


@pytest.mark.parametrize("column", ["res", "model_id"])
def test_manifest_rejects_histogram_with_mixed_null_identity_fields(tmp_path, column):
    frame = pd.DataFrame({"city": ["A", "A"], "hex_id": ["h", "h"], "res": [8, 8], "mode_id": [0, 1], "mode_image_count": [1, 1], "sampled_image_count": [2, 2], "mode_fraction": [.5, .5], "model_id": ["m", "m"]})
    frame.loc[1, column] = None
    frame.to_parquet(tmp_path / "city=A.parquet", index=False)

    with pytest.raises(ValueError, match="invalid histogram"):
        load_manifest().collect_histogram_cities(["A"], tmp_path, allow_missing=True)


def test_manifest_requires_selected_model_id(tmp_path):
    pd.DataFrame({"city": ["A"], "hex_id": ["h"], "res": [8], "mode_id": [0], "mode_image_count": [1], "sampled_image_count": [1], "mode_fraction": [1.0], "model_id": ["old-model"]}).to_parquet(
        tmp_path / "city=A.parquet", index=False
    )

    with pytest.raises(ValueError, match="expected selected-model"):
        load_manifest().collect_histogram_cities(
            ["A"], tmp_path, allow_missing=True, expected_model_id="selected-model"
        )


def test_summary_uses_manifest_pairs_and_retains_empty_thresholded_pair():
    module = load_summary()
    frame = pd.DataFrame({"city_1": ["A"], "city_2": ["B"], "js_similarity": [.5]})
    expected = [("A", "B"), ("A", "C")]

    result = module.summarize_city_pairs(frame, expected_pairs=expected)

    missing = result[(result.city_1 == "A") & (result.city_2 == "C")].iloc[0]
    assert missing.pair_count_observed == 0
    assert pd.isna(missing.js_similarity_avg)
