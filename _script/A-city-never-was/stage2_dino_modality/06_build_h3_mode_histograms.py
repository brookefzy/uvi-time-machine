#!/usr/bin/env python3
"""Build validated sparse H3 mode histograms."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2_dino_modality.mode_ops import build_histogram
from stage2_dino_modality.common import validate_sparse_histogram
def make_histogram(assignments:pd.DataFrame)->pd.DataFrame:
 result=build_histogram(assignments);validate_sparse_histogram(result);return result
