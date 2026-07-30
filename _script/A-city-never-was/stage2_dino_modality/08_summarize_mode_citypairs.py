#!/usr/bin/env python3
"""Summarize exact H3 mode similarities by unordered city pair."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def summarize_city_pairs(frame:pd.DataFrame)->pd.DataFrame:
 if frame.empty:return pd.DataFrame(columns=["city_1","city_2","js_similarity_avg","p50","p90","p95","max","pair_count_observed"])
 return frame.groupby(["city_1","city_2"],as_index=False).agg(js_similarity_avg=("js_similarity","mean"),p50=("js_similarity",lambda x:x.quantile(.5)),p90=("js_similarity",lambda x:x.quantile(.9)),p95=("js_similarity",lambda x:x.quantile(.95)),max=("js_similarity","max"),pair_count_observed=("js_similarity","size"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);summarize_city_pairs(pd.read_parquet(a.input)).to_parquet(a.output,index=False)
if __name__=="__main__":main()
