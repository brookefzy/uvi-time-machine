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


def test_split_train_holdout_is_city_stratified_disjoint_and_order_independent():
    module = load_module()
    pool = pd.DataFrame(
        {
            "city": ["A"] * 5 + ["B"] * 5,
            "name": [f"image-{index}" for index in range(5)] * 2,
            "e_0000": np.arange(10, dtype=np.float32),
        }
    )

    train, holdout = module.split_train_holdout(pool, fraction=.2, seed=17)
    shuffled_train, shuffled_holdout = module.split_train_holdout(
        pool.sample(frac=1, random_state=9), fraction=.2, seed=17
    )

    assert train.groupby("city").size().to_dict() == {"A": 4, "B": 4}
    assert holdout.groupby("city").size().to_dict() == {"A": 1, "B": 1}
    assert not set(zip(train.city, train.name)).intersection(zip(holdout.city, holdout.name))
    assert set(zip(train.city, train.name)) == set(zip(shuffled_train.city, shuffled_train.name))
    assert set(zip(holdout.city, holdout.name)) == set(zip(shuffled_holdout.city, shuffled_holdout.name))


def test_split_train_holdout_rejects_city_with_only_one_image():
    module = load_module()
    pool = pd.DataFrame({"city": ["A", "B", "B"], "name": ["only", "one", "two"]})

    with __import__("pytest").raises(ValueError, match="at least two images per city"):
        module.split_train_holdout(pool, fraction=.2, seed=17)


def test_multi_seed_stability_reports_all_pairwise_ari_statistics():
    module = load_module()
    labels = [
        np.array([0, 0, 1, 1]),
        np.array([9, 9, 3, 3]),
        np.array([0, 1, 0, 1]),
    ]

    result = module.summarize_seed_stability(labels)

    expected = [module.seed_stability(a, b) for a, b in __import__("itertools").combinations(labels, 2)]
    assert result["stability"] == __import__("pytest").approx(np.median(expected))
    assert result["stability_mean"] == __import__("pytest").approx(np.mean(expected))
    assert result["stability_min"] == __import__("pytest").approx(np.min(expected))
    assert result["stability_pair_count"] == 3
    assert result["stability_seed_count"] == 3


def test_stability_seed_sequence_requires_at_least_two_models():
    module = load_module()

    assert module.stability_seeds(primary_seed=42, count=5) == [42, 43, 44, 45, 46]
    with __import__("pytest").raises(ValueError, match="at least two"):
        module.stability_seeds(primary_seed=42, count=1)


def test_model_config_versions_holdout_and_stability_evaluation():
    module = load_module()

    config = module.build_model_config(
        k=16,
        primary_seed=42,
        seeds=[42, 43, 44, 45, 46],
        niter=100,
        columns=["e_0000", "e_0001"],
        max_training_images_per_city=100000,
        holdout_fraction=.2,
        holdout_split_seed=17,
    )

    assert config["holdout_strategy"] == "city_stratified_hash_v1"
    assert config["stability_strategy"] == "all_pairs_ari_median_v1"
    assert config["stability_seeds"] == [42, 43, 44, 45, 46]
    assert config["holdout_split_seed"] == 17
