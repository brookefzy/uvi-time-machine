import importlib.util
from pathlib import Path
import pandas as pd
def load():
 p=Path(__file__).with_name("08_summarize_mode_citypairs.py");s=importlib.util.spec_from_file_location("s",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_summary_has_one_unordered_city_pair_row():
 result=load().summarize_city_pairs(pd.DataFrame({"city_1":["A","A"],"city_2":["B","B"],"js_similarity":[.5,1.]}))
 assert result.iloc[0].pair_count_observed==2 and result.iloc[0].js_similarity_avg==.75
