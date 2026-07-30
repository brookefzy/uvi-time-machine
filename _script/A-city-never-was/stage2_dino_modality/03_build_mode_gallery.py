#!/usr/bin/env python3
"""Render a portable HTML review gallery for global DINO mode representatives."""
from __future__ import annotations
from html import escape
from pathlib import Path
import pandas as pd
def render_gallery(rows:pd.DataFrame,output:Path)->None:
 output.parent.mkdir(parents=True,exist_ok=True)
 cards="".join(f"<article><h2>Mode {escape(str(r.mode_id))}</h2><p>{escape(str(getattr(r,'city','')))} · {escape(str(getattr(r,'hex_id','')))} · cosine {float(getattr(r,'assignment_cosine',0)):.4f}</p><img src='{escape(str(r.path))}'></article>" for r in rows.itertuples())
 output.write_text(f"<!doctype html><meta charset='utf-8'><title>Global DINO modes</title><style>article{{display:inline-block;width:280px;vertical-align:top;margin:8px}}img{{max-width:100%;height:180px;object-fit:contain}}</style>{cards}")
