import importlib.util
from pathlib import Path
import pandas as pd

SCRIPT=Path(__file__).with_name("04_select_mode_model.py")
def load():
    spec=importlib.util.spec_from_file_location("select",SCRIPT); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_build_selected_model_honors_explicit_override():
    scorecard=pd.DataFrame({"k":[64,128],"status":["ok","ok"],"stability":[.95,.95],"min_mode_share":[.1,.1],"held_out_mean_cohesion":[.8,.81],"model_id":["k=64-a","k=128-b"]})
    result=load().build_selected_model(scorecard, selected_k=64)
    assert result["selected_k"]==64 and result["selection_rule"]=="explicit_override"

def test_selected_model_requires_model_id_from_scorecard():
    scorecard=pd.DataFrame({"k":[64],"status":["ok"],"stability":[.95],"min_mode_share":[.1],"held_out_mean_cohesion":[.8],"model_id":["k=64-abc"]})
    assert load().build_selected_model(scorecard, selected_k=64)["model_id"] == "k=64-abc"
