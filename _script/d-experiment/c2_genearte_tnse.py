import os
import pandas as pd
import numpy as np
from glob import glob
import gspread
import h3
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns


def get_std(df_seg_update, variables_remain):
    scaler = StandardScaler().fit(df_seg_update[variables_remain])
    data = scaler.transform(df_seg_update[variables_remain])
    return data


def get_tsne(data, n_components=2):

    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


DATA_FOLDER = "/group/geog_pyloo/08_GSV/data/_curated/c_seg_hex/c_seg_hex"
FILES = os.listdir(DATA_FOLDER)

FILENMAE = "c_seg_cat=31_res={res}.parquet"

variables = [
    "bike",
    "building",
    "bus",
    "car",
    "grass",
    "house",
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
    "table+chair",
    "tower",
    "traffic light",
    "trashcan",
    "tree",
    "truck",
    "van",
    "wall",
    "window",
]
index_cols = ["city_lower", "hex_id", "img_count", "res"]


def export_tnse(res, variables=variables):

    df = pd.read_parquet(os.path.join(DATA_FOLDER, FILENMAE.format(res=res)))
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
    tsne_df = pd.concat([df[index_cols], tsne_df], axis=1)
    print("finished concatination.")
    print(tsne_df.head())

    tsne_df.to_parquet(
        os.path.join(
            DATA_FOLDER,
            FILENMAE.replace(".parquet", "").format(res=res) + "_tsne.parquet",
        ),
        index=False,
    )

    return tsne_df


for res in [8, 9, 12]:
    export_tnse(res=res)
    print(f"finish {res}")
