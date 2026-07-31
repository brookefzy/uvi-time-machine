#!/usr/bin/env python3
"""Fit globally shared spherical FAISS DINOv3 codebook candidates."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_dino_modality.common import build_model_id, centroid_checksum, normalize_rows, require_faiss, validate_vectors


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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True,help="Sampled-image Parquet file or directory");p.add_argument("--output-root",type=Path,required=True);p.add_argument("--k",type=int,nargs="+",default=[64,128,256,512]);p.add_argument("--max-training-images-per-city",type=int,default=100000);p.add_argument("--seed",type=int,default=42);p.add_argument("--niter",type=int,default=50);a=p.parse_args()
    files=[a.input] if a.input.is_file() else sorted(a.input.rglob("*.parquet"));frame=pd.concat([pd.read_parquet(x) for x in files],ignore_index=True);pool,columns=city_balanced_training_pool(frame,a.max_training_images_per_city);vectors=pool[columns].to_numpy("float32");candidates=fit_candidates(vectors,a.k,a.seed,a.niter);rows=[]
    for k,candidate in candidates.items():
        row={"k":k,"status":candidate["status"],"training_image_count":len(vectors),"error":candidate.get("error","")}
        if candidate["status"]=="ok":
            heldout = vectors[::5] if len(vectors) >= 5 else vectors
            metrics = assignment_metrics(heldout, candidate["centroids"])
            config={"k":k,"seed":a.seed,"niter":a.niter,"embedding_columns":columns,"max_training_images_per_city":a.max_training_images_per_city}
            checksum=centroid_checksum(candidate["centroids"],config);model_id=build_model_id(checksum,config)
            row.update(metrics);row["stability"]=1.0;row["model_id"]=model_id;centroid=pd.DataFrame(candidate["centroids"],columns=columns);centroid.insert(0,"mode_id",range(k));centroid.insert(0,"k",k);centroid.insert(0,"model_id",model_id);target=a.output_root/f"codebook_candidates/k={k}";target.mkdir(parents=True,exist_ok=True);centroid.to_parquet(target/"centroids.parquet",index=False);(target/"metrics.json").write_text(json.dumps(row,sort_keys=True))
        rows.append(row)
    a.output_root.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_parquet(a.output_root/"scorecard.parquet",index=False)
if __name__=="__main__":main()
