import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("02_fit_evaluate_codebooks.py")


def load_module():
    spec = importlib.util.spec_from_file_location("fit_codebooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fit_candidates_emits_unit_centroids_and_invalid_k_diagnostic():
    module = load_module()
    vectors = np.array([[1, 0], [.99, .01], [0, 1], [.01, .99]], dtype=np.float32)

    candidates = module.fit_candidates(vectors, [2, 9], seed=7, niter=5)

    assert candidates[2]["status"] == "ok"
    np.testing.assert_allclose(np.linalg.norm(candidates[2]["centroids"], axis=1), 1.0, atol=1e-5)
    assert candidates[9] == {"status": "invalid", "error": "k=9 exceeds training rows=4"}


def test_seed_stability_uses_label_permutation_invariant_adjusted_rand_score():
    module = load_module()
    assert module.seed_stability(np.array([0, 0, 1, 1]), np.array([9, 9, 3, 3])) == 1.0


def test_assignment_metrics_report_cosine_and_mode_support():
    module = load_module()
    metrics = module.assignment_metrics(
        np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
        np.array([[1, 0], [0, 1]], dtype=np.float32),
    )
    assert metrics["held_out_mean_cohesion"] == 1.0
    assert metrics["near_empty_mode_count"] == 0


def test_city_balanced_training_pool_caps_each_city_deterministically():
    module = load_module()
    frame = pd.DataFrame({"city": ["A", "A", "A", "B"], "name": ["z", "a", "b", "c"], "e_0000": [1., 1., 0., 0.], "e_0001": [0., 0., 1., 1.]})
    selected, columns = module.city_balanced_training_pool(frame, max_images_per_city=2)
    assert selected[["city", "name"]].values.tolist() == [["A", "a"], ["A", "b"], ["B", "c"]]
    assert columns == ["e_0000", "e_0001"]
