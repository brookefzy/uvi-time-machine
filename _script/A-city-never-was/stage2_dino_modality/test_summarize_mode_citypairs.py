import importlib.util
from pathlib import Path
import pandas as pd
def load():
 p=Path(__file__).with_name("08_summarize_mode_citypairs.py");s=importlib.util.spec_from_file_location("s",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_summary_has_one_unordered_city_pair_row():
 result=load().summarize_city_pairs(pd.DataFrame({"city_1":["A","A"],"city_2":["B","B"],"js_similarity":[.5,1.]}))
 assert result.iloc[0].pair_count_observed==2 and result.iloc[0].js_similarity_avg==.75


def test_read_similarity_input_concatenates_partitioned_pair_files(tmp_path):
 module = load()
 first = tmp_path / "city_1=A" / "city_2=B.parquet"
 second = tmp_path / "city_1=A" / "city_2=C.parquet"
 first.parent.mkdir()
 second.parent.mkdir(exist_ok=True)
 pd.DataFrame({"city_1":["A"],"city_2":["B"],"js_similarity":[.5]}).to_parquet(first)
 pd.DataFrame({"city_1":["A"],"city_2":["C"],"js_similarity":[.7]}).to_parquet(second)
 assert len(module.read_similarity_input(tmp_path)) == 2


def test_manifest_reader_ignores_stale_pairs_outside_current_manifest(tmp_path):
 module = load()
 current = tmp_path / "city_1=A" / "city_2=B" / "part_res=8.parquet"
 stale = tmp_path / "city_1=A" / "city_2=C" / "part_res=8.parquet"
 current.parent.mkdir(parents=True)
 stale.parent.mkdir(parents=True)
 pd.DataFrame({"city_1":["A"],"city_2":["B"],"js_similarity":[.5]}).to_parquet(current)
 pd.DataFrame({"city_1":["A"],"city_2":["C"],"js_similarity":[.7]}).to_parquet(stale)

 result = module.read_manifest_similarity_input(tmp_path, [("A", "B")])

 assert result[["city_1", "city_2"]].values.tolist() == [["A", "B"]]
