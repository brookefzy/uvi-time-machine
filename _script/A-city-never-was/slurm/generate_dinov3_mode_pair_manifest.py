#!/usr/bin/env python3
"""Create deterministic cross-city mode-histogram pair manifests."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument("--city-meta",type=Path,required=True);p.add_argument("--histogram-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 cities=[];model_id=None
 for city in sorted(pd.read_csv(a.city_meta)["City"].dropna().unique()):
  path=a.histogram_root/f"city={city}.parquet"
  if not path.exists(): continue
  frame=pd.read_parquet(path)
  if frame.empty or frame.res.nunique()!=1 or int(frame.res.iloc[0])!=8 or frame.model_id.nunique()!=1: raise ValueError(f"invalid histogram: {path}")
  current=frame.model_id.iloc[0]
  if model_id is not None and current!=model_id: raise ValueError("histogram model IDs differ")
  model_id=current;cities.append(city)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text("".join(f"{left}|{right}\n" for i,left in enumerate(cities) for right in cities[i+1:]))
if __name__=="__main__":main()
