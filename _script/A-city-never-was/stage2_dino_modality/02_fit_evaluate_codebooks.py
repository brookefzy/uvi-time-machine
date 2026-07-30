#!/usr/bin/env python3
"""Fit globally shared spherical FAISS DINOv3 codebook candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_dino_modality.common import normalize_rows, require_faiss, validate_vectors


def city_balanced_training_pool(frame: pd.DataFrame, max_images_per_city: int) -> tuple[pd.DataFrame, list[str]]:
    """Deterministically cap each city without replacing H3-balanced sampling."""
    if max_images_per_city < 1:
        raise ValueError("max_images_per_city must be positive")
    columns = sorted((column for column in frame.columns if column.startswith("e_") and column[2:].isdigit()), key=lambda column: int(column[2:]))
    if not columns or {"city", "name"} - set(frame.columns):
        raise ValueError("sampled data requires city, name, and embedding columns")
    selected = (frame.drop_duplicates(["city", "name"]).sort_values(["city", "name"], kind="stable").groupby("city", group_keys=False).head(max_images_per_city).reset_index(drop=True))
    return selected, columns


def seed_stability(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compare assignments without treating arbitrary cluster labels as meaningful."""
    return float(adjusted_rand_score(labels_a, labels_b))


def assignment_metrics(holdout_vectors: np.ndarray, centroids: np.ndarray) -> dict[str, float | int]:
    """Measure held-out cohesion and support for a normalized codebook."""
    vectors = normalize_rows(holdout_vectors)
    centers = normalize_rows(centroids)
    scores = vectors @ centers.T
    labels = scores.argmax(axis=1)
    best = scores[np.arange(len(scores)), labels]
    shares = np.bincount(labels, minlength=len(centers)) / len(labels)
    return {
        "held_out_mean_cohesion": float(best.mean()),
        "held_out_p05_cohesion": float(np.percentile(best, 5)),
        "min_mode_share": float(shares.min()),
        "median_mode_share": float(np.median(shares)),
        "near_empty_mode_count": int((shares == 0).sum()),
    }


def fit_candidates(vectors: np.ndarray, requested_k: list[int], seed: int = 42, niter: int = 50) -> dict[int, dict]:
    """Fit each valid K and retain invalid-K diagnostics for the scorecard."""
    training = normalize_rows(vectors)
    faiss = require_faiss()
    candidates: dict[int, dict] = {}
    for k in requested_k:
        if k < 1:
            candidates[k] = {"status": "invalid", "error": f"k={k} must be positive"}
            continue
        if k > len(training):
            candidates[k] = {"status": "invalid", "error": f"k={k} exceeds training rows={len(training)}"}
            continue
        model = faiss.Kmeans(training.shape[1], k, niter=niter, seed=seed, spherical=True, verbose=False)
        model.train(validate_vectors(training))
        candidates[k] = {"status": "ok", "centroids": normalize_rows(model.centroids.reshape(k, training.shape[1]))}
    return candidates
