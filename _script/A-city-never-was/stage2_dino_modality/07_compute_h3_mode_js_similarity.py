#!/usr/bin/env python3
"""Compute exact cross-city H3 Jensen--Shannon similarities."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import blocked_js

def compute_pairwise(source: pd.DataFrame,target: pd.DataFrame, threshold: float=-1.0, row_block_size: int=64, target_block_size: int=2048)->pd.DataFrame:
 city1=source.city.iloc[0];city2=target.city.iloc[0]
 if city1==city2:raise ValueError("city pair must be distinct")
 if source.model_id.nunique()!=1 or target.model_id.nunique()!=1 or source.model_id.iloc[0]!=target.model_id.iloc[0]:raise ValueError("histogram model IDs must match")
 k=max(source.mode_id.max(),target.mode_id.max())+1
 def dense(frame):
  ids=frame[["hex_id"]].drop_duplicates().hex_id.tolist(); out=np.zeros((len(ids),k),np.float32); lookup={x:i for i,x in enumerate(ids)}
  for r in frame.itertuples():out[lookup[r.hex_id],r.mode_id]=r.mode_fraction
  return ids,out
 hs,a=dense(source);ht,b=dense(target); rows=[]
 for i in range(0,len(hs),row_block_size):
  for j in range(0,len(ht),target_block_size):
   scores=blocked_js(a[i:i+row_block_size],b[j:j+target_block_size])
   for r,x in enumerate(hs[i:i+row_block_size]):
    for c,y in enumerate(ht[j:j+target_block_size]):
     similarity=float(scores[r,c])
     if similarity > threshold: rows.append((city1,x,city2,y,source.model_id.iloc[0],similarity,1-similarity,similarity))
 return pd.DataFrame(rows,columns=["city_1","hex_id_1","city_2","hex_id_2","model_id","js_similarity","js_distance","similarity"])
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--target",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--threshold",type=float,default=-1.0);p.add_argument("--row-block-size",type=int,default=64);p.add_argument("--target-block-size",type=int,default=2048);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);compute_pairwise(pd.read_parquet(a.source),pd.read_parquet(a.target),a.threshold,a.row_block_size,a.target_block_size).to_parquet(a.output,index=False)
if __name__=="__main__":main()
