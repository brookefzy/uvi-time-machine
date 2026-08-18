#!/usr/bin/env python3
"""Fit globally shared spherical FAISS DINOv3 codebook candidates."""

from __future__ import annotations

import argparse, hashlib, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_dino_modality.common import build_model_id, centroid_checksum, normalize_rows, require_faiss, validate_vectors
from stage2_dino_modality.mode_ops import select_model


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


def stability_seeds(primary_seed: int, count: int) -> list[int]:
    """Return consecutive independent seeds beginning with the saved-model seed."""
    if count < 2:
        raise ValueError("stability evaluation requires at least two seeds")
    return list(range(primary_seed, primary_seed + count))


def summarize_seed_stability(labels_by_seed: list[np.ndarray]) -> dict[str, float | int]:
    """Summarize every pairwise adjusted Rand score across fitted seeds."""
    if len(labels_by_seed) < 2:
        raise ValueError("stability evaluation requires at least two label sets")
    scores = np.asarray(
        [seed_stability(left, right) for left, right in combinations(labels_by_seed, 2)],
        dtype=np.float64,
    )
    return {
        "stability": float(np.median(scores)),
        "stability_mean": float(np.mean(scores)),
        "stability_min": float(np.min(scores)),
        "stability_max": float(np.max(scores)),
        "stability_std": float(np.std(scores)),
        "stability_pair_count": int(len(scores)),
        "stability_seed_count": int(len(labels_by_seed)),
    }


def split_train_holdout(frame: pd.DataFrame, fraction: float = .2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve an exact deterministic holdout fraction independently within each city."""
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    missing = {"city", "name"} - set(frame.columns)
    if missing:
        raise ValueError(f"holdout data is missing columns: {sorted(missing)}")
    city_sizes = frame.groupby("city", dropna=False).size()
    if (city_sizes < 2).any():
        cities = city_sizes[city_sizes < 2].index.astype(str).tolist()
        raise ValueError(f"city-stratified holdout requires at least two images per city: {cities}")

    hash_key = hashlib.sha256(str(seed).encode("ascii")).hexdigest()[:16]
    ranked = frame[["city", "name"]].copy()
    ranked["_holdout_hash"] = pd.util.hash_pandas_object(
        ranked[["city", "name"]], index=False, hash_key=hash_key
    ).to_numpy(dtype=np.uint64)
    ranked["_row"] = np.arange(len(frame))
    ranked = ranked.sort_values(["city", "_holdout_hash", "name"], kind="stable")
    ranked["_city_rank"] = ranked.groupby("city", dropna=False).cumcount()
    holdout_counts = np.floor(city_sizes * fraction + .5).astype(int).clip(lower=1)
    holdout_counts = np.minimum(holdout_counts, city_sizes - 1)
    ranked["_holdout_count"] = ranked["city"].map(holdout_counts)
    holdout_rows = ranked.loc[ranked["_city_rank"] < ranked["_holdout_count"], "_row"].to_numpy()
    is_holdout = np.zeros(len(frame), dtype=bool)
    is_holdout[holdout_rows] = True
    order = ["city", "name"]
    training = frame.loc[~is_holdout].sort_values(order, kind="stable").reset_index(drop=True)
    holdout = frame.loc[is_holdout].sort_values(order, kind="stable").reset_index(drop=True)
    return training, holdout


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


def _fit_candidates_normalized(training: np.ndarray, requested_k: list[int], seed: int, niter: int) -> dict[int, dict]:
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


def fit_candidates(vectors: np.ndarray, requested_k: list[int], seed: int = 42, niter: int = 50) -> dict[int, dict]:
    """Fit each valid K and retain invalid-K diagnostics for the scorecard."""
    return _fit_candidates_normalized(normalize_rows(vectors), requested_k, seed, niter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Sampled-image Parquet file or directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--max-training-images-per-city", type=int, default=100000)
    parser.add_argument("--holdout-fraction", type=float, default=.2)
    parser.add_argument("--holdout-split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42, help="Primary seed used for the saved centroid model")
    parser.add_argument("--stability-seed-count", type=int, default=5)
    parser.add_argument("--niter", type=int, default=50)
    return parser


def build_model_config(
    *,
    k: int,
    primary_seed: int,
    seeds: list[int],
    niter: int,
    columns: list[str],
    max_training_images_per_city: int,
    holdout_fraction: float,
    holdout_split_seed: int,
) -> dict:
    """Build the versioned immutable fitting and evaluation configuration."""
    return {
        "k": k,
        "seed": primary_seed,
        "stability_seeds": seeds,
        "stability_strategy": "all_pairs_ari_median_v1",
        "niter": niter,
        "embedding_columns": columns,
        "embedding_dim": len(columns),
        "max_training_images_per_city": max_training_images_per_city,
        "holdout_fraction": holdout_fraction,
        "holdout_strategy": "city_stratified_hash_v1",
        "holdout_split_seed": holdout_split_seed,
    }


def main():
    a = build_parser().parse_args()
    seeds = stability_seeds(a.seed, a.stability_seed_count)
    files = [a.input] if a.input.is_file() else sorted(a.input.rglob("*.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    pool, columns = city_balanced_training_pool(frame, a.max_training_images_per_city)
    training_frame, holdout_frame = split_train_holdout(pool, a.holdout_fraction, a.holdout_split_seed)
    del frame, pool
    training_city_count = int(training_frame.city.nunique())
    holdout_city_count = int(holdout_frame.city.nunique())
    training = normalize_rows(training_frame[columns].to_numpy("float32"))
    holdout = normalize_rows(holdout_frame[columns].to_numpy("float32"))
    del training_frame, holdout_frame
    candidate_runs = {
        seed: _fit_candidates_normalized(training, a.k, seed, a.niter)
        for seed in seeds
    }
    candidates = candidate_runs[a.seed]
    rows=[]
    for k,candidate in candidates.items():
        row={"k":k,"status":candidate["status"],"training_image_count":len(training),"error":candidate.get("error","")}
        if candidate["status"]=="ok":
            metrics = assignment_metrics(holdout, candidate["centroids"])
            labels_by_seed = [
                (holdout @ candidate_runs[seed][k]["centroids"].T).argmax(1)
                for seed in seeds
            ]
            stability = summarize_seed_stability(labels_by_seed)
            config = build_model_config(
                k=k,
                primary_seed=a.seed,
                seeds=seeds,
                niter=a.niter,
                columns=columns,
                max_training_images_per_city=a.max_training_images_per_city,
                holdout_fraction=a.holdout_fraction,
                holdout_split_seed=a.holdout_split_seed,
            )
            checksum=centroid_checksum(candidate["centroids"],config);model_id=build_model_id(checksum,config)
            row.update(metrics);row.update(stability);row.update({"model_id":model_id,"training_image_count":len(training),"holdout_image_count":len(holdout),"training_city_count":training_city_count,"holdout_city_count":holdout_city_count,"holdout_strategy":config["holdout_strategy"],"stability_strategy":config["stability_strategy"],"holdout_fraction":a.holdout_fraction,"holdout_split_seed":a.holdout_split_seed,"primary_seed":a.seed,"stability_seeds":",".join(map(str,seeds))});centroid=pd.DataFrame(candidate["centroids"],columns=columns);centroid.insert(0,"training_image_count",len(training));centroid.insert(0,"embedding_dim",len(columns));centroid.insert(0,"mode_id",range(k));centroid.insert(0,"k",k);centroid.insert(0,"model_id",model_id);target=a.output_root/f"codebook_candidates/k={k}";target.mkdir(parents=True,exist_ok=True);centroid.to_parquet(target/"centroids.parquet",index=False);(target/"metrics.json").write_text(json.dumps(row,sort_keys=True))
        rows.append(row)
    a.output_root.mkdir(parents=True,exist_ok=True);scorecard=pd.DataFrame(rows);scorecard.to_parquet(a.output_root/"scorecard.parquet",index=False)
    try:
        recommendation=select_model(scorecard); recommendation["model_id"]=str(scorecard.loc[scorecard.k==recommendation["selected_k"],"model_id"].iloc[0]);(a.output_root/"recommended_model.json").write_text(json.dumps(recommendation,sort_keys=True,indent=2))
    except ValueError:
        pass
if __name__=="__main__":main()
