#!/usr/bin/env python3
"""Summarize exact H3 mode similarities by unordered city pair."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import pandas as pd
def summarize_city_pairs(frame:pd.DataFrame)->pd.DataFrame:
 if frame.empty:return pd.DataFrame(columns=["city_1","city_2","js_similarity_avg","p50","p90","p95","max","pair_count_observed"])
 return frame.groupby(["city_1","city_2"],as_index=False).agg(js_similarity_avg=("js_similarity","mean"),p50=("js_similarity",lambda x:x.quantile(.5)),p90=("js_similarity",lambda x:x.quantile(.9)),p95=("js_similarity",lambda x:x.quantile(.95)),max=("js_similarity","max"),pair_count_observed=("js_similarity","size"))


def read_similarity_input(path: Path) -> pd.DataFrame:
 if path.is_file(): return pd.read_parquet(path)
 files=sorted(path.rglob("*.parquet"))
 if not files: raise ValueError(f"no similarity Parquet files under {path}")
 return pd.concat([pd.read_parquet(file) for file in files],ignore_index=True)


def validate_expected_pairs(frame: pd.DataFrame, cities: list[str]) -> None:
 expected=set(combinations(sorted(set(cities)),2))
 observed=set(map(tuple, frame[["city_1","city_2"]].drop_duplicates().itertuples(index=False,name=None)))
 if observed != expected: raise ValueError(f"expected {len(expected)} unordered city pairs, observed {len(observed)}")
 if "model_id" in frame and frame.model_id.nunique()!=1: raise ValueError("similarity shards contain multiple model IDs")

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--city-meta",type=Path);a=p.parse_args();frame=read_similarity_input(a.input)
 if a.city_meta: validate_expected_pairs(frame,pd.read_csv(a.city_meta)["City"].dropna().tolist())
 a.output.parent.mkdir(parents=True,exist_ok=True);summarize_city_pairs(frame).to_parquet(a.output,index=False)
if __name__=="__main__":main()
