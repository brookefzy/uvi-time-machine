"""This code select sample images from five cities to visualize the cluster types"""

import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
import h3


ROOT = "/lustre1/g/geog_pyloo/05_timemachine"
CURATED_FOLDER = f"{ROOT}/_curated"
EXPORT_FOLDER = f"{ROOT}/_curated/c_analysis"

# IMG_SAMPLE_FOLDER = (
#     "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_analysis/img_sample"
# )
DATA_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_seg_hex"
PANO_PATH = "{ROOT}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
CURATED_TARGET = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_seg_hex"
META_PATH = "{ROOT}/GSV/gsv_rgb/{cityabbr}/gsvmeta/{cityabbr}_meta.csv"


N = 7
H3_RES = [9]
IMG_SAMPLE_FOLDER = (
    f"/lustre1/g/geog_pyloo/05_timemachine/_curated/c_analysis/img_sample_cluster={N}_restandardized"
)
if not os.path.exists(IMG_SAMPLE_FOLDER):
    os.makedirs(IMG_SAMPLE_FOLDER)


def get_std(df_seg_update, variables_remain):
    scaler = StandardScaler().fit(df_seg_update[variables_remain])
    data = scaler.transform(df_seg_update[variables_remain])
    return data


def load_class():
    serviceaccount = "/home/yuanzf/uvi-time-machine/google_drive_personal.json"
    import gspread

    # from oauth2client.service_account import ServiceAccountCredentials
    gc = gspread.service_account(filename=serviceaccount)

    def read_url(url, SHEET_NAME):
        SHEET_ID = url.split("/")[5]
        spreadsheet = gc.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        rows = worksheet.get_all_records()
        df_spread = pd.DataFrame(rows)
        return df_spread, worksheet

    url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
    SHEETNAME = "object150"
    obj_meta, other_worksheet = read_url(url, SHEETNAME)
    return obj_meta


def get_result(cityabbr, curated_folder, f_suffixes="*panoptic.csv"):
    outfolder = f"{curated_folder}/{cityabbr}"
    seg_file = glob(os.path.join(outfolder, f_suffixes))
    panoptic_df = []
    for p in seg_file:
        temp = pd.read_csv(p)
        panoptic_df.append(temp)
    panoptic_df = pd.concat(panoptic_df).reset_index(drop=True)
    return panoptic_df


def clean_seg(seg_df, pano_df, meta_df):

    seg_df_filtered = seg_df.merge(meta_df, on="img")
    seg_df_filtered = seg_df_filtered[seg_df_filtered["size"] >= 10000].reset_index(
        drop=True
    )
    print("Segmentation shape after filter: ", seg_df_filtered.shape[0])
    # count number of unique labels per image and drop it if image has fewer than 3
    seg_df_filtered["label_c"] = seg_df_filtered.groupby("img")["labels"].transform(
        "nunique"
    )
    seg_df_filtered = seg_df_filtered[seg_df_filtered["label_c"] >= 3].reset_index(
        drop=True
    )
    print(
        "Segmentation shape after filter images with problems: ",
        seg_df_filtered.shape[0],
    )

    seg_df_summary = (
        seg_df_filtered.groupby(["img", "labels"]).agg({"areas": "sum"}).reset_index()
    )
    seg_df_summary["panoid"] = seg_df_summary["img"].apply(lambda x: x[:22])

    col_cols = ["labels"]
    index_cols = ["img", "year", "h3_9"]
    seg_df_summary_pano = seg_df_summary.merge(pano_df, on=["panoid"])

    if seg_df_summary_pano.shape[0] < seg_df_summary.shape[0]:
        print("data missing after data join.")
        print("Before join: ", seg_df_summary.shape[0])
        print("After join: ", seg_df_summary_pano.shape[0])
    else:
        print("data consistent")

    seg_df_summary = seg_df_summary_pano.drop_duplicates(index_cols + col_cols)
    print("Segmentation shape: ", seg_df_summary.shape[0])
    seg_df_pivot = (
        seg_df_summary.pivot(columns=col_cols, index=index_cols, values="areas")
        .reset_index()
        .fillna(0)
    )
    return seg_df_pivot


def load_raw(city):
    cityabbr = city.lower().replace(" ", "")
    curate_folder = CURATED_FOLDER.format(ROOTFOLDER=ROOT)

    seg_df = get_result(cityabbr, CURATED_FOLDER, f_suffixes="*seg.csv")

    pano_df = pd.read_csv(PANO_PATH.format(ROOT=ROOT, cityabbr=cityabbr))[
        ["panoid", "lat", "lon", "year", "month"]
    ]

    for res in [9]:
        pano_df[f"h3_{res}"] = pano_df.apply(
            lambda x: h3.geo_to_h3(x.lat, x.lon, res), axis=1
        )

    meta_df = pd.read_csv(META_PATH.format(ROOT=ROOT, cityabbr=cityabbr))
    meta_df["img"] = meta_df["path"].apply(lambda x: x.split("/")[-1].split(".")[0])
    # here make sure
    meta_df = meta_df[["img", "size", "path"]]
    return seg_df, meta_df, pano_df


def get_seg_data(city, hex_detail_cluster):
    cityabbr = city.lower().replace(" ", "")
    seg_df, meta_df, pano_df = load_raw(city)
    seg_df_pivot = clean_seg(seg_df, pano_df, meta_df)
    print(f"city {cityabbr} saved")
    print("*" * 50)

    hex_detail_cluster_city_sel = hex_detail_cluster[
        hex_detail_cluster["city_lower"] == cityabbr
    ].copy()
    seg_df_pivot_sel = seg_df_pivot.merge(
        hex_detail_cluster_city_sel[["hex_id", f"cluster_{N}"]],
        left_on="h3_9",
        right_on="hex_id",
    ).drop("hex_id", axis=1)

    obj_meta = load_class()
    obj_meta["id"] = obj_meta["id"].astype(str)
    ADE_CATEGORIES_DICT = dict(zip(obj_meta["id"].values, obj_meta["category"].values))
    new_cols = []
    for x in seg_df_pivot_sel.columns:
        if str(x) in obj_meta["id"].values:
            new_cols.append(ADE_CATEGORIES_DICT[str(x)])
        else:
            new_cols.append(str(x))
    seg_df_pivot_sel.columns = new_cols

    # drop the columns if all value are 0
    variables = set(
        [v for v in seg_df_pivot_sel.columns if v in obj_meta["category"].unique()]
    )
    print("Variables original: ", len(variables))
    to_drop = ["other"]
    variables_remain = [v for v in variables if not v in to_drop]
    print("Variables kept: ", len(variables_remain))
    seg_df_pivot_sel_stack = (
        seg_df_pivot_sel.set_index(["img", "year", "h3_9", f"cluster_{N}"])
        .stack()
        .reset_index()
        .fillna(0)
        .groupby(["img", "year", "h3_9", f"cluster_{N}", "level_4"])
        .sum()
        .reset_index()
        .pivot_table(
            index=["img", "year", "h3_9", f"cluster_{N}"], columns="level_4", values=0
        )
        .reset_index()
        .fillna(0)
        .drop(columns=["other"], axis=1)
    )
    # standardize the data
    seg_df_pivot_sel_stack_scaled = get_std(seg_df_pivot_sel_stack, feature_ls)
    seg_df_pivot_sel_stack_scaled_keys = seg_df_pivot_sel_stack[
        ["img", "year", "h3_9", f"cluster_{N}"]
    ]
    seg_df_pivot_sel_stack = pd.concat(
        [
            seg_df_pivot_sel_stack_scaled_keys,
            pd.DataFrame(seg_df_pivot_sel_stack_scaled, columns=feature_ls),
        ],
        axis=1,
    )
    #  remove the image if it only has fewer than 3 types of elements presented in the image

    return seg_df_pivot_sel_stack, meta_df


def sel_cluster_sample(city, cluster_mean, seg_df_i, meta_df, cluster_i):
    cityabbr = city.lower().replace(" ", "")
    cluster_mean_i = cluster_mean[feature_ls].iloc[cluster_i].values
    # remove the image if it only has fewer than 3 types of elements presented in the image

    seg_df_i = (
        seg_df_pivot_sel_stack[seg_df_pivot_sel_stack[f"cluster_{N}"] == cluster_i]
        .set_index(["img"])
        .drop([f"cluster_{N}", "year", "h3_9"], axis=1)
    )
    seg_df_i["distance_to_mean"] = np.linalg.norm(
        seg_df_i[feature_ls].values - cluster_mean_i, axis=1
    )
    # use cosine similarity
    # seg_df_i['distance_to_mean'] = seg_df_i[feature_ls].apply(lambda x:
    #     np.dot(x, cluster_mean_i)/(np.linalg.norm(x)*np.linalg.norm(cluster_mean_i)), axis = 1)
    seg_df_i_sel = seg_df_i.sort_values("distance_to_mean").head(20)

    imgsel = seg_df_i_sel.index
    # find the path of these images
    img_folder = os.path.join(IMG_SAMPLE_FOLDER, "round" + prefixfull)
    os.makedirs(img_folder, exist_ok=True)
    export_folder = os.path.join(img_folder, "cluster_" + str(cluster_i), cityabbr)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    gsv_df_sel = meta_df[meta_df["img"].isin(imgsel)].reset_index(drop=True)
    import shutil
    from tqdm import tqdm

    for i, row in tqdm(gsv_df_sel.iterrows()):
        shutil.copy(row["path"], os.path.join(export_folder, row["img"] + ".jpg"))


#############################################
# STEP 1. Load the hex cluster mean ##################
# HEX_LEVEL_FILE = f"c_hex_full_cluster={N}.csv"
prefixfull = "_built_environment"
# prefixfull = ""
# N_CAT = 30
N_CAT = 27
HEX_LEVEL_FILE = (
    # "c_seg_cat=31_res=9_withincity_built_environment_tsne_cluster_range.csv"
    # f"c_seg_cat={N_CAT}_res=9_withincity{prefixfull}_tsne_cluster_range.csv"
    "c_seg_cat=27_res=9_withincity_built_environment_tsne_restandardized_cluster_range.csv"
    # "c_seg_cat=30_res=9_withincity_tsne_cluster_range.csv"
)
hex_df = pd.read_csv(os.path.join(DATA_FOLDER, HEX_LEVEL_FILE))
print("Data 1 with cluster loaded.")

# feature_ls = [ 'bike', 'building', 'bus', 'car',
#        'grass', 'ground', 'house', 'installation', 'lake+waterboday', 'light',
#        'mountain+hill', 'person', 'pole', 'railing', 'road',
#        'shrub', 'sidewalk', 'signage', 'sky', 'skyscraper', 'sportsfield',
#        'table+chair', 'tower', 'traffic light', 'trashcan', 'tree', 'truck',
#        'van', 'wall', 'window']
feature_ls = [
    "skyscraper",
    "light",
    "road",
    "sidewalk",
    "traffic light",
    "railing",
    "window",
    "building",
    "signage",
    "pole",
    # "table+chair",
    # "house",
    "trashcan",
    "installation",
    "shrub",
    "grass",
    "tree",
    "sky",
    "lake+waterboday",
    "sportsfield",
    "mountain+hill",
]
feature_dynamics = ["bike", "person", "bus", "car", "van", "truck"]
if prefixfull == "":
    feature_ls = feature_ls + feature_dynamics
    print("feature includes dynamic features")
else:
    print("feature only includes built environment")

hex_detail = pd.read_parquet(
    os.path.join(DATA_FOLDER, f"c_seg_cat={N_CAT}_res=9.parquet")
)
hex_detail_hex = hex_detail[["hex_id"]]
hex_detail_scaled = get_std(hex_detail, feature_ls)
hex_detail_scaled = pd.DataFrame(hex_detail_scaled, columns=feature_ls)
print("Hex Detail Loaded:", hex_detail.shape)
hex_detail = pd.concat([hex_detail_hex, hex_detail_scaled], axis=1)

hex_detail_cluster = hex_detail.merge(
    hex_df[["hex_id", f"cluster_{N}", "city_lower"]], on="hex_id"
)
hex_detail_cluster.shape[0], hex_detail.shape[0]
# find the cluster mean and find sample images that are close to the sample mean

cluster_mean = (
    hex_detail_cluster[feature_ls + [f"cluster_{N}"]].groupby(f"cluster_{N}").mean()
)
print("Cluster Mean Calculated:", cluster_mean.shape)

#######################################################################
# 2. Load City img data and sample the useful ones ##################

# For each cluster,
# 1. sample 50 images within each N clusters' hexagon from these cities
# ['New York','Singapore', 'Hong Kong', 'London','Bangkok','Nairobi']
# 2. for each 50 images, compute the proportion of the features
# 3. calculate the distance between the images and the cluster mean
# 4. find the images that are closest to the cluster mean from each city
CITY_LS = ["New York", "London", "Bangkok", "Nairobi"]
for city in CITY_LS[:]:
    print(f"city {city} started")
    seg_df_pivot_sel_stack, meta_df = get_seg_data(city, hex_detail_cluster)
    print(f"city {city} loaded")
    for cluster_i in range(N):
        sel_cluster_sample(
            city, cluster_mean, seg_df_pivot_sel_stack, meta_df, cluster_i
        )
    print(f"city {city} saved")
    print("*" * 50)

# cd /scr/u/yuanzf/anaconda3/envs/py312/lib
# python /home/yuanzf/uvi-time-machine/_script/d-experiment/11_viz_sample_image_cluster.py
