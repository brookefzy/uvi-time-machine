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


def split_train_holdout(vectors: np.ndarray, fraction: float = .2) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically reserve a non-overlapping evaluation holdout."""
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    count=max(1, int(round(len(vectors) * fraction)))
    if count >= len(vectors):
        raise ValueError("at least two training vectors are required")
    return vectors[:-count], vectors[-count:]


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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True,help="Sampled-image Parquet file or directory");p.add_argument("--output-root",type=Path,required=True);p.add_argument("--k",type=int,nargs="+",default=[64,128,256,512]);p.add_argument("--max-training-images-per-city",type=int,default=100000);p.add_argument("--holdout-fraction",type=float,default=.2);p.add_argument("--seed",type=int,default=42);p.add_argument("--niter",type=int,default=50);a=p.parse_args()
    files=[a.input] if a.input.is_file() else sorted(a.input.rglob("*.parquet"));frame=pd.concat([pd.read_parquet(x) for x in files],ignore_index=True);pool,columns=city_balanced_training_pool(frame,a.max_training_images_per_city);vectors=pool[columns].to_numpy("float32");training,holdout=split_train_holdout(vectors,a.holdout_fraction);candidates=fit_candidates(training,a.k,a.seed,a.niter);secondary=fit_candidates(training,a.k,a.seed+1,a.niter);rows=[]
    for k,candidate in candidates.items():
        row={"k":k,"status":candidate["status"],"training_image_count":len(vectors),"error":candidate.get("error","")}
        if candidate["status"]=="ok":
            metrics = assignment_metrics(holdout, candidate["centroids"])
            primary_labels=(normalize_rows(holdout) @ candidate["centroids"].T).argmax(1)
            alternate_labels=(normalize_rows(holdout) @ secondary[k]["centroids"].T).argmax(1)
            config={"k":k,"seed":a.seed,"stability_seed":a.seed+1,"niter":a.niter,"embedding_columns":columns,"embedding_dim":len(columns),"max_training_images_per_city":a.max_training_images_per_city,"holdout_fraction":a.holdout_fraction}
            checksum=centroid_checksum(candidate["centroids"],config);model_id=build_model_id(checksum,config)
            row.update(metrics);row.update({"stability":seed_stability(primary_labels,alternate_labels),"model_id":model_id,"training_image_count":len(training),"holdout_image_count":len(holdout)});centroid=pd.DataFrame(candidate["centroids"],columns=columns);centroid.insert(0,"training_image_count",len(training));centroid.insert(0,"embedding_dim",len(columns));centroid.insert(0,"mode_id",range(k));centroid.insert(0,"k",k);centroid.insert(0,"model_id",model_id);target=a.output_root/f"codebook_candidates/k={k}";target.mkdir(parents=True,exist_ok=True);centroid.to_parquet(target/"centroids.parquet",index=False);(target/"metrics.json").write_text(json.dumps(row,sort_keys=True))
        rows.append(row)
    a.output_root.mkdir(parents=True,exist_ok=True);scorecard=pd.DataFrame(rows);scorecard.to_parquet(a.output_root/"scorecard.parquet",index=False)
    try:
        recommendation=select_model(scorecard); recommendation["model_id"]=str(scorecard.loc[scorecard.k==recommendation["selected_k"],"model_id"].iloc[0]);(a.output_root/"recommended_model.json").write_text(json.dumps(recommendation,sort_keys=True,indent=2))
    except ValueError:
        pass
if __name__=="__main__":main()
