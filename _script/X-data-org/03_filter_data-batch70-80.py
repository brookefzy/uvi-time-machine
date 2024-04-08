"""
This notebook further filter the data by criteria:
1. Calculate the image sizes. smaller than 2000 b is not considered (too few information)
"""

import geopandas as gpd
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from utils.gsvload import GSVSummary
from shapely.geometry import Point
from multiprocessing import Pool
import concurrent.futures
import gspread


RAW_FOLDER_ROOT = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb"
ROAD_PTH = "{RAW_FOLDER_ROOT}/{cityabbrlower}/road/osm.geojson"
# TGT_FILE = "gsv_pano_label.csv" 
TGT_FILE = "{cityabbrlower}_meta.csv"

# Get city list
serviceaccount = "../../google_drive_personal.json"

# from oauth2client.service_account import ServiceAccountCredentials
gc = gspread.service_account(filename=serviceaccount)

def read_url(url, SHEET_NAME):
    SHEET_ID = url.split("/")[5]
    spreadsheet = gc.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    rows = worksheet.get_all_records()
    df_spread = pd.DataFrame(rows)
    return df_spread, worksheet


def get_gsv_file_size(city):
    citylower = city.lower().replace(" ", "")
    print("Now processing: ", city)
    meta_path = "{RAW_FOLDER_ROOT}/{cityabbrlower}/gsvmeta/{tgt_tile}".format(
        RAW_FOLDER_ROOT = RAW_FOLDER_ROOT,
        cityabbrlower = citylower,
        tgt_tile = TGT_FILE.format(cityabbrlower = citylower)
    )

    metadf = pd.read_csv(meta_path)
    
    def get_file_size(file):
        try:
            return file, os.path.getsize(file)
        except OSError as e:
            print(f"Error: {e}")
            return file, None

    if './data/' in metadf["path"].values[0]:
        metadf['path'] = metadf['path'].apply(lambda x: x.replace("./data/", "/lustre1/g/geog_pyloo/05_timemachine/GSV/"))
        print("path fixed")
    files = metadf["path"].values
    if "size" in metadf.columns:
        print("size finished, no need to calculate. continue next")
        return None
    else:
        
        # Store file sizes in a dictionary
        file_sizes = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # The map method helps maintain the order of results
            results = executor.map(get_file_size, files)

            for file, size in results:
                if size is not None:
                    file_sizes[file] = size
                else:
                    file_sizes[file] = 0  # the file is removed

        metadf["size"] = metadf["path"].apply(lambda x: file_sizes[x])
        metadf.to_csv(meta_path, index = False)
        return metadf
    
url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
SHEETNAME = "select_city"
city_meta, other_worksheet = read_url(url, SHEETNAME)
city_meta = city_meta[city_meta['City']!=''].reset_index(drop = True)
cityls = city_meta[city_meta['GSV Downloaded']>0]['City'].unique()
print("city list loaded")


for city in tqdm(cityls[70:80]):
    get_gsv_file_size(city)
    print("*"*100)