#!/usr/bin/env python3
"""Build validated sparse H3 mode histograms."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import build_histogram
from stage2_dino_modality.common import validate_sparse_histogram
def make_histogram(assignments:pd.DataFrame)->pd.DataFrame:
 result=build_histogram(assignments);validate_sparse_histogram(result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);make_histogram(pd.read_parquet(a.input)).to_parquet(a.output,index=False)
if __name__=="__main__":main()
