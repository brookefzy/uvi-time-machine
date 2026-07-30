#!/usr/bin/env python3
"""Assign sampled images to a selected immutable global mode model."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import assign_modes

def assign_sampled_images(rows: pd.DataFrame, vectors: np.ndarray, centroids: np.ndarray, selected_model_id: str, centroid_model_id: str) -> pd.DataFrame:
 if selected_model_id != centroid_model_id: raise ValueError("selected model ID does not match centroid model ID")
 return assign_modes(rows,vectors,centroids,selected_model_id)
