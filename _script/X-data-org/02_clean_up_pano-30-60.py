"""
1. Merge the gsv_pano_label with gsv_path file >> {city}_meta.csv
2. Calculate the image size per image
3. find the development ring for each city
# run this until the 65 cities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from math import sin, cos, sqrt, atan2, radians
import random
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import glob
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from utils.gsvload import GSVSummary
import concurrent.futures
import gspread

import geopandas as gpd
from shapely.geometry import Point

## DEFINE CONSTANT
TGT_FILE = "{cityabbrlower}_meta.csv"
PATH_FILE = "gsv_path.csv"
PANO_FILE = "gsv_pano_label.csv"
RAW_FOLDER_ROOT = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb"
META_FOLDER = "{RAW_FOLDER_ROOT}/{cityabbrlower}/gsvmeta/"
PFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_urban_expansion/t_urban_expansion"
meta = {
    "path":"path to the gsv",
    "panoid":"unique identifier",
    "angle":"image angle",
    "size":"image size",
    "lat":"latitude",
    "lon":"longitude",
    "year":"year taken",
    "month":"month",
    # "id":"sent id",
    "dist_hav":"distance from the city center", 
    "h3_res8":"h3 level 8",
    "h3_res9":"h3 level 9",
    "ring":"development ring"
}


# LOAD ALL cities

serviceaccount = "../../google_drive_personal.json"
# from oauth2client.service_account import ServiceAccountCredentials
gc = gspread.service_account(filename=serviceaccount)
GLOBAL_CRS = "EPSG:4326"

def read_url(url, SHEET_NAME):
    SHEET_ID = url.split("/")[5]
    spreadsheet = gc.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    rows = worksheet.get_all_records()
    df_spread = pd.DataFrame(rows)
    return df_spread, worksheet

def list_city():
    url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
    SHEETNAME = "select_city"
    city_meta, other_worksheet = read_url(url, SHEETNAME)
    city_meta = city_meta[city_meta['City']!=''].reset_index(drop = True)
    cityls = city_meta[city_meta['GSV Downloaded']>0]['City'].unique()
    return cityls

def get_gsv_file_size(citylower, meta_folder):
    """get the file size of each image"""
    print("Now processing: ", citylower)

    path_df = pd.read_csv(os.path.join(meta_folder, PATH_FILE))

    def get_file_size(file):
        try:
            return file, os.path.getsize(file)
        except OSError as e:
            print(f"Error: {e}")
            return file, None

    if './data/' in path_df["path"].values[0]:
        path_df['path'] = path_df['path'].apply(lambda x: x.replace("./data/", "/lustre1/g/geog_pyloo/05_timemachine/GSV/"))
        print("path fixed")
        
    files = path_df["path"].values
    if "size" in path_df.columns:
        print("size finished, no need to calculate. continue next")
        return path_df
    else:
        # Store file sizes in a dictionary
        print("size calculation starts")
        file_sizes = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # The map method helps maintain the order of results
            results = executor.map(get_file_size, files)

            for file, size in results:
                if size is not None:
                    file_sizes[file] = size
                else:
                    file_sizes[file] = 0  # the file is removed

        path_df["size"] = path_df["path"].apply(lambda x: file_sizes[x])
        path_df.to_csv(os.path.join(meta_folder, PATH_FILE), index = False)
        print("Finished calculating the file size")
        return path_df


def get_urban_expansion_ring(citylower, meta_folder):
    """use the panoid file to find the associated ring of development for each city. 
    For cities do not exist in the urban expansion project, move to next"""

    meta_folder = META_FOLDER.format(
        cityabbrlower = citylower,
        RAW_FOLDER_ROOT = RAW_FOLDER_ROOT
    )
    
    
    if citylower == "hong_kong":
            geofilename = 'hong_kong,_hong_kong.geojson' # edge case
    elif citylower == "taipei":
        geofilename = 'taipei,_taiwan.geojson'
    else:
        geofilename = citylower + ".geojson"
        
    pano_df = pd.read_csv(os.path.join(meta_folder, PANO_FILE))
    
    if os.path.exists(os.path.join(PFOLDER, geofilename)):
        area = gpd.read_file(os.path.join(PFOLDER, geofilename))

        
        meta_gdf = gpd.GeoDataFrame(pano_df, 
                                    geometry = [Point(x,y) for x,y in zip(pano_df['lon'], pano_df['lat'])])
        meta_gdf.crs = GLOBAL_CRS
        meta_intersect = gpd.sjoin(
            meta_gdf[['panoid', 'geometry']], 
            area[["geometry", "ring"]],
        )
        meta_update = pano_df.merge(meta_intersect[['panoid', 'ring']], on = 'panoid', how = 'left')
        print(meta_update.shape[0])
        print("Down joining the urban expansion project for city: ", citylower)
        return meta_update.drop("geometry", axis = 1)

    else:
        pano_df['ring'] = np.nan
        return pano_df


def run_city(city):
    citylower= city.lower().replace(" ", "")
    meta_folder = META_FOLDER.format(
        cityabbrlower = citylower,
        RAW_FOLDER_ROOT = RAW_FOLDER_ROOT
    )
    
    path_df = get_gsv_file_size(citylower, meta_folder)
    print("Finished calculating the file size")
    pano_update = get_urban_expansion_ring(citylower, meta_folder)
    print ("Finished updating the development ring")
    
    # merge two dataframe
    meta_df = path_df.merge(pano_update, on = 'panoid', how = 'left')
    print("Total rows: ", meta_df.shape[0])
    tgt_file = TGT_FILE.format(cityabbrlower = city.lower().replace(" ", ""))
    tgt_path = os.path.join(meta_folder, tgt_file)
    if "geometry" in meta_df.columns:
        meta_df = meta_df.drop_duplicates("path").drop("geometry", axis = 1)
    meta_df.to_csv(tgt_path, index = False)
    meta_df['angle'] = meta_df['path'].apply(lambda x: x.split("/")[-1].split(".")[0][23:])
    print(meta_df['angle'].value_counts()) # for checking the angle
    # assert all columns are in the meta
    assert all([x in meta_df.columns for x in meta.keys()])
    return None
    

def main():
    cityls = list_city()
    for city in cityls[30:60]:
        run_city(city)
        print("Finished processing: ", city)
        print("*"*100)
    
    print("All cities are processed till number 80")
    
if __name__ == "__main__":
    main()
    