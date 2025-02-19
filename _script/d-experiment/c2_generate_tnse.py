import os
import pandas as pd
from sklearn import manifold
from sklearn.preprocessing import StandardScaler


def get_std(df_seg_update, variables_remain):
    # first convert all variables into ratio
    df_seg_update['total_pixel'] = df_seg_update[variables_remain].sum(axis = 1)
    for v in variables_remain:
        df_seg_update[v] = df_seg_update[v]/df_seg_update['total_pixel']
    scaler = StandardScaler().fit(df_seg_update[variables_remain])
    data = scaler.transform(df_seg_update[variables_remain])
    return data


def get_tsne(data, n_components=2):

    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
DATA_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_hex"
FILES = os.listdir(DATA_FOLDER)

FILENAME = "c_seg_cat={n_cat}_res={res}.parquet"
FILENAME_WITHIN = "c_seg_cat={n_cat}_res={res}_withincity.parquet"
# PREFIX = "_built_environment"
PREFIX = ""
# FILENAME = "c_seg_long_cat=31_res={res}.parquet"
# FILENAME_WITHIN = "c_seg_long_cat=31_res={res}_withincity.parquet"

variables = [
    "bike",
    "building",
    "bus",
    "car",
    "grass",
    # "house",
    "installation",
    "lake+waterboday",
    "light",
    "mountain+hill",
    "other",
    "person",
    "pole",
    "railing",
    "road",
    "shrub",
    "sidewalk",
    "signage",
    "sky",
    "sportsfield",
    # "table+chair",
    "tower",
    "traffic light",
    "trashcan",
    "tree",
    "truck",
    "van",
    # "wall",
    "window",
]
index_cols_long = ["year_group", "city_lower", "hex_id", "img_count", "res"]
index_cols_cross = ["city_lower", "hex_id", "img_count", "res"]


def export_tnse(res, filename, variables=variables):

    file_name = filename.format(res=res, n_cat=N_CAT)
    df = pd.read_parquet(os.path.join(DATA_FOLDER, file_name))
    print("Current dataset length: ", df.shape[0])
    # standardize the data

    data = get_std(df, variables)
    print("finish basic standardization")
    # get tsne
    tsne_data = get_tsne(data)  # this takes very long time.
    print("finish tsne")
    # save the tsne_data once done
    tsne_df = pd.DataFrame(tsne_data, columns=["tsne_1", "tsne_2"]).reset_index(
        drop=True
    )
    print(tsne_df.head())
    index_cols = [x for x in df.columns if x in index_cols_long]
    tsne_df = pd.concat([df[index_cols], tsne_df], axis=1)
    print("finished concatination.")
    print(tsne_df.head())

    tsne_df.to_parquet(
        os.path.join(
            DATA_FOLDER,
            file_name.replace(".parquet", f"{PREFIX}_tsne.parquet"),
        ),
        index=False,
    )
    print(f"finish {res} {filename}")
    return tsne_df


# for res in [8, 9,12]:
#     for filename in [FILENAME_WITHIN]:
#         export_tnse(res=res, filename=filename)
N_CAT = 27
for res in [8, 9]:
    for filename in [FILENAME_WITHIN]:
        export_tnse(res=res, filename=filename)

# /home/yuanzf/uvi-time-machine/_script/d-experiment/c2_generate_tnse.py
