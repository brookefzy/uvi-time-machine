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


def test_pairwise_writer_publishes_a_single_atomic_parquet_shard(tmp_path):
 module=load()
 source=pd.DataFrame({"city":["A"],"hex_id":["h1"],"res":[8],"mode_id":[0],"mode_fraction":[1.],"model_id":["m"]})
 target=source.assign(city="B",hex_id="h2")
 output=tmp_path / "part.parquet"
 module.write_pairwise(source,target,output)
 assert output.exists()
 assert pd.read_parquet(output).js_similarity.iloc[0] == pytest.approx(1.)
