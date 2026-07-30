import importlib.util
from pathlib import Path
import pandas as pd
import pytest
SCRIPT=Path(__file__).with_name("07_compute_h3_mode_js_similarity.py")
def load():
 s=importlib.util.spec_from_file_location("js",SCRIPT);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_pairwise_rejects_same_city_and_model_mismatch():
 frame=pd.DataFrame({"city":["A"],"hex_id":["h"],"res":[8],"mode_id":[0],"mode_fraction":[1.],"model_id":["m"]})
 with pytest.raises(ValueError,match="distinct"): load().compute_pairwise(frame,frame)
 other=frame.assign(city="B",model_id="other")
 with pytest.raises(ValueError,match="model IDs"): load().compute_pairwise(frame,other)
