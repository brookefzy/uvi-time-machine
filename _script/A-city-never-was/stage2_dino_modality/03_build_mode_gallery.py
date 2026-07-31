#!/usr/bin/env python3
"""Render a portable HTML review gallery for global DINO mode representatives."""
from __future__ import annotations
from html import escape
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from shutil import copy2

def build_representatives(sampled:pd.DataFrame,centroids:pd.DataFrame,index:pd.DataFrame,images_per_mode:int=20)->pd.DataFrame:
 columns=[c for c in centroids if c.startswith("e_")]; scores=sampled[columns].to_numpy("float32") @ centroids[columns].to_numpy("float32").T
 work=sampled[[c for c in sampled if not c.startswith("e_")]].copy();work["mode_id"]=scores.argmax(1);work["assignment_cosine"]=scores.max(1)
 if "name" not in index: index=index.assign(name=index.path.map(lambda x:Path(x).name))
 return work.merge(index[["name","path"]],on="name",how="inner").sort_values(["mode_id","assignment_cosine"],ascending=[True,False]).groupby("mode_id",group_keys=False).head(images_per_mode)
def render_gallery(rows:pd.DataFrame,output:Path)->None:
 output.parent.mkdir(parents=True,exist_ok=True)
 cards="".join(f"<article><h2>Mode {escape(str(r.mode_id))}</h2><p>{escape(str(getattr(r,'city','')))} · {escape(str(getattr(r,'hex_id','')))} · cosine {float(getattr(r,'assignment_cosine',0)):.4f}</p><img src='{escape(str(r.path))}'></article>" for r in rows.itertuples())
 output.write_text(f"<!doctype html><meta charset='utf-8'><title>Global DINO modes</title><style>article{{display:inline-block;width:280px;vertical-align:top;margin:8px}}img{{max-width:100%;height:180px;object-fit:contain}}</style>{cards}")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--representatives",type=Path);p.add_argument("--sampled",type=Path);p.add_argument("--centroids",type=Path);p.add_argument("--image-index",type=Path);p.add_argument("--images-per-mode",type=int,default=20);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.representatives: rows=pd.read_parquet(a.representatives)
 elif a.sampled and a.centroids and a.image_index: rows=build_representatives(pd.read_parquet(a.sampled),pd.read_parquet(a.centroids),pd.read_parquet(a.image_index),a.images_per_mode)
 else: p.error("supply --representatives or --sampled --centroids --image-index")
 render_gallery(rows,a.output)
if __name__=="__main__":main()
