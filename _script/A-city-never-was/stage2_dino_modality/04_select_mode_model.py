#!/usr/bin/env python3
"""Select an immutable global DINOv3 codebook model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import select_model

def build_selected_model(scorecard: pd.DataFrame, selected_k: int|None=None, model_id: str|None=None) -> dict:
    available=set(scorecard.loc[scorecard.status=="ok","k"])
    if selected_k is not None:
        if selected_k not in available: raise ValueError(f"selected k={selected_k} is unsupported")
        result={"selected_k":int(selected_k),"selection_rule":"explicit_override"}
    else:
        selected=select_model(scorecard); result={"selected_k":selected["selected_k"],"selection_rule":selected["rule"]}
    if model_id: result["model_id"]=model_id
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--scorecard",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--selected-k",type=int); p.add_argument("--model-id"); a=p.parse_args()
    result=build_selected_model(pd.read_parquet(a.scorecard),a.selected_k,a.model_id); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,sort_keys=True,indent=2))
if __name__=="__main__": main()
