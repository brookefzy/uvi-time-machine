import numpy as np
import pandas as pd


def test_select_model_prefers_smallest_valid_elbow():
    from stage2_dino_modality.mode_ops import select_model
    scores = pd.DataFrame({"k": [64, 128, 256], "status": ["ok"] * 3, "stability": [.95, .95, .95], "min_mode_share": [.02, .02, .02], "held_out_mean_cohesion": [.80, .804, .806]})
    assert select_model(scores, min_stability=.9, min_mode_share=.01, cohesion_gain_epsilon=.005)["selected_k"] == 128


def test_assign_and_histogram_preserve_per_hex_fractions():
    from stage2_dino_modality.mode_ops import assign_modes, build_histogram
    rows = pd.DataFrame({"city": ["A", "A"], "hex_id": ["h", "h"], "res": [8, 8], "name": ["a", "b"]})
    assigned = assign_modes(rows, np.array([[1, 0], [0, 1]], np.float32), np.array([[1, 0], [0, 1]], np.float32), "m")
    hist = build_histogram(assigned)
    assert hist["mode_fraction"].tolist() == [.5, .5]


def test_blocked_js_matches_identical_and_disjoint_cases():
    from stage2_dino_modality.mode_ops import blocked_js
    result = blocked_js(np.array([[1, 0]], np.float32), np.array([[1, 0], [0, 1]], np.float32))
    np.testing.assert_allclose(result, [[1, 0]])
