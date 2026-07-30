#!/usr/bin/env python3
"""Assign sampled images to a selected immutable global mode model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import assign_modes

def assign_sampled_images(rows: pd.DataFrame, vectors: np.ndarray, centroids: np.ndarray, selected_model_id: str, centroid_model_id: str) -> pd.DataFrame:
 if selected_model_id != centroid_model_id: raise ValueError("selected model ID does not match centroid model ID")
 return assign_modes(rows,vectors,centroids,selected_model_id)
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--centroids",type=Path,required=True);p.add_argument("--selected-model",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 rows=pd.read_parquet(a.input);centroids=pd.read_parquet(a.centroids);selected=json.loads(a.selected_model.read_text());model_id=centroids.model_id.iloc[0];vectors=rows[[c for c in rows if c.startswith("e_")]].to_numpy("float32");matrix=centroids[[c for c in centroids if c.startswith("e_")]].to_numpy("float32");out=assign_sampled_images(rows.drop(columns=[c for c in rows if c.startswith("e_")]),vectors,matrix,selected["model_id"],model_id);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(a.output,index=False)
if __name__=="__main__":main()
