import os
import pandas as pd
import numpy as np
from glob import glob
import gspread
import h3
from tqdm import tqdm
import gc
##################### NOTES ###################################################
## FILE TOO LARGE, needs to process city by city
##############################################################################
year_groups = {
    "2017-2019":[2017,2018,2019],
    "2022-2024":[2024,2022,2023],
}

##################### HELPER ###################################################
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


def construct_cat(df_seg, obj_meta):
    
    new_cols = []
    for x in df_seg.columns:
        if x in obj_meta["id"].values:
            new_cols.append(ADE_CATEGORIES_DICT[x])
        else:
            new_cols.append(x)
    df_seg.columns = new_cols
    print(df_seg.columns)
    print("Before filter to years:", df_seg.shape[0])
    df_seg = df_seg[df_seg['year'].isin([2017,2018,2019,2022,2023,2024])].reset_index(drop = True)
    print("After filter to years:", df_seg.shape[0])
    df_seg['year_group'] = df_seg['year'].apply(lambda x: [k for k,v in year_groups.items() if x in v][0])

    # drop the columns if all value are 0
    variables = set([v for v in df_seg.columns if v in obj_meta["category"].unique()])
    print("Variables original: ", len(variables))
    to_drop = ["other"]
    variables_remain = [v for v in variables if not v in to_drop]
    print("Variables kept: ", len(variables_remain))

    # combine categories and transform
    df_long = (
        df_seg.set_index(["city_lower", "hex_id", "img_count","year_group"]).stack().reset_index()
    )
    # print(df_long.columns)
    df_long.rename(columns={"level_4": "category", 0: "value"}, inplace=True)
    df_long["value"] = df_long["value"].fillna(0).astype(float)

    df_seg_update = (
        df_long.groupby(["city_lower", "hex_id", "img_count", "category", "year_group"])["value"]
        .sum()
        .reset_index()
        .pivot(
            columns="category",
            index=["year_group","city_lower", "hex_id", "img_count"],
            values="value",
        )
        .reset_index()
    )
    return df_seg_update, variables_remain

def get_cross(curated_cross, obj_meta, res):
    segfiles = glob(curated_cross + "/*.parquet")
    df_seg = []
    for f in tqdm(segfiles):
        temp = pd.read_parquet(f)
        temp["city_lower"] = f.split("/")[-1].split(".")[0]
        temp_filter = temp[temp['res']==res].reset_index(drop = True)
        temp_update, variables_remain = construct_cat(temp_filter, obj_meta)
        print(temp_update.shape)
        df_seg.append(temp_update)
    df_seg = pd.concat(df_seg).reset_index(drop=True)
    return df_seg

############################################# SET UP CONSTANT ############################################################
# CURATED_FOLDER_LONG = (
#     "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_seg_longitudinal_all"
# )
CURATED_FOLDER_LONG = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_seg_longitudinal_year"

RAW_PATH = (
    "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{city}/gsvmeta/{city}_meta.csv"
)

CURATED_TARGET = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_seg_hex"
if not os.path.exists(CURATED_TARGET):
    os.makedirs(CURATED_TARGET)
GRAPHIC_PATH = "/lustre1/g/geog_pyloo/05_timemachine/_graphic"
if not os.path.exists(GRAPHIC_PATH):
    os.makedirs(GRAPHIC_PATH)

##################### EXPORT STAGING FILES FOR LATER ANALYSIS############################################################
obj_meta = load_class()
print("Loaded: ", obj_meta.shape[0])
n_cat = len(obj_meta["category"].unique())
print("Number of categories: ", n_cat)
obj_meta["id"] = obj_meta["id"].astype(str)
ADE_CATEGORIES_DICT = dict(zip(obj_meta["id"].values, obj_meta["category"].values))
print("Exporting staging files for later analysis")

for res in [8,9]:
    print("Now processing resoluation: ", res)
    df_seg = get_cross(CURATED_FOLDER_LONG, obj_meta, res)
    df_seg.to_parquet(
        CURATED_TARGET + f"/c_seg_long_cat={n_cat}_res={res}.parquet", index=False
    )
    
# !python /home/yuanzf/uvi-time-machine/_script/d-experiment/c1_combine_seg_cat-long.py
    

