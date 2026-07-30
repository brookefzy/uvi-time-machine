#!/usr/bin/env python3
"""Compute exact cross-city H3 Jensen--Shannon similarities."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import blocked_js

def compute_pairwise(source: pd.DataFrame,target: pd.DataFrame)->pd.DataFrame:
 city1=source.city.iloc[0];city2=target.city.iloc[0]
 if city1==city2:raise ValueError("city pair must be distinct")
 if source.model_id.nunique()!=1 or target.model_id.nunique()!=1 or source.model_id.iloc[0]!=target.model_id.iloc[0]:raise ValueError("histogram model IDs must match")
 k=max(source.mode_id.max(),target.mode_id.max())+1
 def dense(frame):
  ids=frame[["hex_id"]].drop_duplicates().hex_id.tolist(); out=np.zeros((len(ids),k),np.float32); lookup={x:i for i,x in enumerate(ids)}
  for r in frame.itertuples():out[lookup[r.hex_id],r.mode_id]=r.mode_fraction
  return ids,out
 hs,a=dense(source);ht,b=dense(target); scores=blocked_js(a,b)
 return pd.DataFrame([(city1,x,city2,y,source.model_id.iloc[0],float(scores[i,j]),float(1-scores[i,j]),float(scores[i,j])) for i,x in enumerate(hs) for j,y in enumerate(ht)],columns=["city_1","hex_id_1","city_2","hex_id_2","model_id","js_similarity","js_distance","similarity"])
