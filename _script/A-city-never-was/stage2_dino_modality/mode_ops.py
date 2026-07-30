"""Selection, assignment, histogram, and exact JSD primitives."""
from __future__ import annotations
import numpy as np
import pandas as pd
from stage2_dino_modality.common import normalize_rows

def select_model(scorecard: pd.DataFrame, min_stability: float=.9, min_mode_share: float=.001, cohesion_gain_epsilon: float=.005) -> dict:
    valid = scorecard[(scorecard.status == "ok") & (scorecard.stability >= min_stability) & (scorecard.min_mode_share >= min_mode_share)].sort_values("k")
    if valid.empty: raise ValueError("no candidate meets support/stability thresholds")
    previous = None
    for row in valid.itertuples():
        if previous is not None and row.held_out_mean_cohesion - previous.held_out_mean_cohesion < cohesion_gain_epsilon:
            return {"selected_k": int(row.k), "rule": "smallest_valid_elbow"}
        previous = row
    best = valid.assign(score=lambda x: x.stability*x.held_out_mean_cohesion).sort_values("score", ascending=False).iloc[0]
    return {"selected_k": int(best.k), "rule": "max_stability_times_cohesion"}

def assign_modes(rows: pd.DataFrame, vectors: np.ndarray, centroids: np.ndarray, model_id: str) -> pd.DataFrame:
    scores = normalize_rows(vectors) @ normalize_rows(centroids).T
    result = rows.copy(); result["mode_id"] = scores.argmax(axis=1); result["assignment_cosine"] = scores.max(axis=1); result["model_id"] = model_id
    return result

def build_histogram(assignments: pd.DataFrame) -> pd.DataFrame:
    keys=["city","hex_id","res","mode_id","model_id"]
    counts=assignments.groupby(keys,as_index=False).size().rename(columns={"size":"mode_image_count"})
    totals=assignments.groupby(["city","hex_id","res","model_id"],as_index=False).size().rename(columns={"size":"sampled_image_count"})
    result=counts.merge(totals,on=["city","hex_id","res","model_id"]); result["mode_fraction"]=result.mode_image_count/result.sampled_image_count
    return result

def blocked_js(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    p=np.asarray(source,dtype=np.float32)[:,None,:]; q=np.asarray(target,dtype=np.float32)[None,:,:]; m=(p+q)/2
    def kl(x):
        values=np.broadcast_to(x, m.shape); terms=np.zeros_like(m); mask=values>0
        ratio=np.ones_like(m); np.divide(values,m,out=ratio,where=mask)
        np.log2(ratio,out=terms,where=mask)
        terms[mask] = values[mask] * terms[mask]
        return terms.sum(axis=2)
    return 1-np.sqrt((kl(p)+kl(q))/2)
