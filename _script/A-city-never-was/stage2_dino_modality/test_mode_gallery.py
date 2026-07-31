import importlib.util
from pathlib import Path

import pandas as pd


def load():
    script = Path(__file__).with_name("03_build_mode_gallery.py")
    spec = importlib.util.spec_from_file_location("gallery", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gallery_reads_partitioned_datasets_and_joins_on_city_and_name(tmp_path):
    module = load()
    sampled = tmp_path / "sampled"
    index = tmp_path / "index"
    sampled.mkdir(); index.mkdir()
    pd.DataFrame({"city":["A"],"name":["same.jpg"],"hex_id":["h"],"e_0000":[1.]}).to_parquet(sampled / "city=A.parquet")
    pd.DataFrame({"city":["B"],"name":["same.jpg"],"hex_id":["wrong"],"e_0000":[1.]}).to_parquet(sampled / "city=B.parquet")
    pd.DataFrame({"city":["A"],"name":["same.jpg"],"path":["/a.jpg"]}).to_parquet(index / "city=A.parquet")
    pd.DataFrame({"city":["B"],"name":["same.jpg"],"path":["/b.jpg"]}).to_parquet(index / "city=B.parquet")
    centroids = pd.DataFrame({"mode_id":[0],"e_0000":[1.]})
    rows = module.build_representatives(module.read_parquet_dataset(sampled), centroids, module.read_parquet_dataset(index))
    assert set(rows.path) == {"/a.jpg", "/b.jpg"}


def test_gallery_derives_city_from_partition_filename_when_index_omits_it(tmp_path):
    module = load()
    path = tmp_path / "city=Paris.parquet"
    pd.DataFrame({"name":["image.jpg"], "path":["/image.jpg"]}).to_parquet(path)
    assert module.read_parquet_dataset(path).city.tolist() == ["Paris"]
