# %%
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import numpy as np
import gc
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Constants
ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
VALFOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir"
)
CURATED_FOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob"
)
TRAIN_TEST_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"
RAW_PATH = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{city}/gsvmeta/{city}_meta.csv"

PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
PATH_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"

CURATE_FOLDER_SOURCE = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary"
CURATE_FOLDER_EXPORT = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity"

if not os.path.exists(CURATE_FOLDER_EXPORT):
    os.makedirs(CURATE_FOLDER_EXPORT)
    
vector_ls = [str(x) for x in range(0, 127)]


RES_EXCLUDE = 11

TODAY = pd.Timestamp.today().strftime("%Y%m%d")
EXPORT_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_similarity_{TODAY}"
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)
    print("Created folder: ", EXPORT_FOLDER)

# %%
gc.collect()
res_exclude = 11
res_sel = 7
city_meta = pd.read_csv("../city_meta.csv")
files = glob(CURATE_FOLDER_SOURCE + f"/*res_exclude={res_exclude}.parquet")
print(len(files))
df_all = []
for f in files:
    temp = pd.read_parquet(f)
    temp = temp[temp.res == res_sel].reset_index(drop=True)
    temp["city"] = os.path.basename(f).split("_")[1].replace("city=", "")
    df_all.append(temp)
df_all = pd.concat(df_all).drop_duplicates("hex_id").reset_index(drop=True)
df_all = df_all.drop(columns=["res"])
df_map = df_all[['hex_id', 'city']].drop_duplicates().reset_index(drop=True)
print("The total hexagons to process: ", df_all.shape)


city_ls = city_meta.City.values
CURATE_FOLDER_EXPORT2 = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair"

for city in city_ls:
    print(f"Processing city: {city}")
    filename = f"similarity_city={city}_res={res_sel}.parquet"
    temp = pd.read_parquet(
        os.path.join(CURATE_FOLDER_EXPORT2, filename)
    )
    print("This raw file contains: ",temp.shape)
    temp["key"] = temp.apply(
        lambda row: "_".join(sorted([row["hex_id1"], row["hex_id2"]])), axis=1
    )
    temp = temp.groupby("key").agg({"similarity": "max"}).reset_index()
    print(temp.shape)
    temp["hex_id1"] = temp["key"].apply(lambda x: x.split("_")[0])
    temp["hex_id2"] = temp["key"].apply(lambda x: x.split("_")[1])
    temp = temp.drop(columns=["key"])
    temp = (
        temp.merge(df_map, left_on="hex_id1", right_on="hex_id")
        .drop(["hex_id"], axis=1)
        .merge(
            df_map,
            left_on="hex_id2",
            right_on="hex_id",
            suffixes=["_1", "_2"],
        )
        .drop(["hex_id"], axis=1)
    )
    innercity = temp[(temp["city_1"] == temp["city_2"])].reset_index(drop=True)
    print("This innercity file contains: ",innercity.shape)
    intracity = temp[(temp["city_1"] != temp["city_2"])].reset_index(drop=True)
    print("This intracity file contains: ",intracity.shape)
    intracity.to_parquet(
        os.path.join(
            EXPORT_FOLDER, f"similarity_intracity_city={city}_res={res_sel}.parquet"
        )
    )
    print(f"Finished city: {city}")
