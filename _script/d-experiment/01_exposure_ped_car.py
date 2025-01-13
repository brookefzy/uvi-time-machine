import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import glob
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import h3
import geopandas as gpd
import argparse

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
PATH_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"
CURATED_FOLDER = f"{ROOTFOLDER}/_curated"
META_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/{cityabbr}_meta.csv"
EXFOLDER = os.path.join(CURATED_FOLDER, "c_object_crossectional")
EXFOLDER_LONG = os.path.join(CURATED_FOLDER, "c_object_longitudinal")
EXFOLDER_SEG_LONG = os.path.join(CURATED_FOLDER, "c_seg_longitudinal_all")
if not os.path.exists(EXFOLDER):
    os.makedirs(EXFOLDER)
if not os.path.exists(EXFOLDER_LONG):
    os.makedirs(EXFOLDER_LONG)
# YEAR_GROUP = ["2015-2018", "2020-2023"]
YEAR_GROUP = ["<2014", "2014-2015", "2016-2017", "2018-2019", "2020-2021", "2022-2023"]
column_map = {
    "<2014": [
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
    ],
    "2014-2015": [2014, 2015],
    "2016-2017": [2016, 2017],
    "2018-2019": [2018, 2019],
    "2020-2021": [2020, 2021],
    "2022-2023": [2022, 2023],
}

OBJECT_SOURCE_FOLDER = "{CURATED_FOLDER}/{city_abbr}/*_objects.parquet"

# variables may change later
# res = 9  # resolution of hexagon
min_num_pano = 2  # minimum number of panoid per each hexagon to avoid sampling bias
exportfolder = f"{EXFOLDER}/exposure_measure"
os.makedirs(exportfolder, exist_ok=True)

def get_hex_basics(city_abbr, res):
    # read all object files and concat them into one df
    df_pano = pd.read_csv(PANO_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=city_abbr))
    df_path = pd.read_csv(PATH_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=city_abbr))
    # only keep the panoid that has a path
    df_pano_inner = df_pano[
        df_pano["panoid"].isin(df_path["panoid"].unique())
    ].reset_index(drop=True)
    print(df_pano_inner.shape[0], "out of ", df_pano.shape[0], "panoids have path")
    df_pano_inner["hex_id"] = df_pano_inner.apply(
        lambda x: h3.geo_to_h3(x["lat"], x["lon"], res), axis=1
    )

    # keep these hex
    df_all_keep = df_pano_inner[(df_pano_inner["year"] >= 2014)].reset_index(drop=True)
    print(df_all_keep.shape[0], "panoids are kept")
    # assign the year_group; skip for now

    # get number of panoid per hex per year
    df_all_keep_hex = (
        df_all_keep.groupby(["hex_id", "year"])["panoid"]
        .nunique()
        .reset_index(name="panoid_n")
    )
    return df_all_keep_hex, df_all_keep
## Incorporate into the 02b_object_detect_summary.py

def get_exposure(city, res):
    city_abbr = city.lower().replace(" ", "")
    df_all_keep_hex, df_all_keep = get_hex_basics(city_abbr, res)
    print("Done with basic hexagon setup")

    # process the prediction results
    objfiles = glob.glob(
        OBJECT_SOURCE_FOLDER.format(CURATED_FOLDER=CURATED_FOLDER, city_abbr=city_abbr)
    )
    if len(objfiles) == 0:
        print("No object files found for", city)
    df = pd.concat([pd.read_parquet(f) for f in objfiles])
    df["panoid"] = df["img"].apply(lambda x: x[:22])
    # object specific processing
    variable_ls_kep = [
        "person",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "train",
        "fire hydrant",
        "van",
        "bench",
        "chair",
        "table",
        "traffic light",
        "stop sign",
    ]

    df_sel = (
        df[df["object_name"].isin(variable_ls_kep)]
        .groupby(["panoid", "object_name"])
        .size()
        .reset_index(name="n")
    )
    # get the panoid that have car/truck/bus and person observed at the same time
    df_sel['with_vehicle'] = df_sel.groupby('panoid')['object_name'].transform(lambda x: x.isin(['car', 'truck', 'bus','van']).any())

    object_summary = \
        df_all_keep[["panoid", "year", "hex_id"]]\
        .merge(df_sel[df_sel['object_name'].isin(['person','motorcycle', 'bicycle'])], on="panoid", how="inner")
    object_summary_update = object_summary.groupby(['hex_id', 'object_name','with_vehicle'])['n'].sum().reset_index()
    object_summary_wide = object_summary_update.pivot(columns = ['object_name','with_vehicle'], index = 'hex_id', values = 'n').reset_index().fillna(0)
    # # flatten the column names
    object_summary_wide.columns = ['_'.join([str(x) for x in col]).strip() for col in object_summary_wide.columns.values]
    object_summary_wide['person_total'] = object_summary_wide['person_False'] + object_summary_wide['person_True']
    object_summary_wide['bicycle_total'] = object_summary_wide['bicycle_False'] + object_summary_wide['bicycle_True']
    object_summary_wide['motorcycle_total'] = object_summary_wide['motorcycle_False'] + object_summary_wide['motorcycle_True']
    object_summary_wide['person_exposure'] = object_summary_wide['person_True'] / object_summary_wide['person_total']
    object_summary_wide['bicycle_exposure'] = object_summary_wide['bicycle_True'] / object_summary_wide['bicycle_total']
    object_summary_wide['motorcycle_exposure'] = object_summary_wide['motorcycle_True'] / object_summary_wide['motorcycle_total']
    # object_summary_wide.to_parquet(f"{exportfolder}/{city_abbr}_res={res}.parquet")
    print("Done with crossectional object exposure: ", city)
    return object_summary_wide
    
def load_all_cities():
    # import multiprocessing
    city_ls = (
            pd.read_csv("/home/yuanzf/uvi-time-machine/_script/city_meta.csv")["City"]
            .unique()
            .tolist()
        )
    
    for res in [8,9,12]:
        fulldf = []
        for city in city_ls:
            print("Processing", city)
            try:
                object_summary_wide = get_exposure(city, res)
            except:
                print("Error in processing", city)
            object_summary_wide['city'] = city
        
            fulldf.append(object_summary_wide)
        fulldf = pd.concat(fulldf).reset_index(drop = True)
        fulldf.to_parquet(f"{exportfolder}/exposure_res={res}.parquet")
def main():
    load_all_cities()
    
if __name__ == "__main__":
    main()
    
# python /home/yuanzf/uvi-time-machine/_script/d-experiment/01_exposure_ped_car.py