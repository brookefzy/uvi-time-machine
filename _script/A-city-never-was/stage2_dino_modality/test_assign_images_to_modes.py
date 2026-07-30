import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT=Path(__file__).with_name("05_assign_images_to_modes.py")
def load():
 s=importlib.util.spec_from_file_location("assign",SCRIPT); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_assign_sampled_images_rejects_model_id_mismatch():
 rows=pd.DataFrame({"city":["A"],"hex_id":["h"],"res":[8],"name":["x"]})
 with __import__("pytest").raises(ValueError,match="model ID"):
  load().assign_sampled_images(rows,np.array([[1,0]],np.float32),np.array([[1,0]],np.float32),"expected","wrong")
