import os
import gspread
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import time

def read_url(url, SHEET_NAME):
    SHEET_ID = url.split("/")[5]
    spreadsheet = gc.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    rows = worksheet.get_all_records()
    df_spread = pd.DataFrame(rows)
    return df_spread, worksheet




GC_URL = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
SHEETNAME = "select_city_classifier"
RAW_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_raw/r_boundary_osm"
if not os.path.exists(RAW_FOLDER):
    os.makedirs(RAW_FOLDER)

serviceaccount = "../../google_drive_personal.json"
gc = gspread.service_account(filename=serviceaccount)
city_meta, other_worksheet = read_url(GC_URL, SHEETNAME)
city_meta = city_meta[city_meta['City']!=''].reset_index(drop = True)
city_meta["country_clean"] = np.where(city_meta["Country"].isin(["USA", "United States"]), 
                                      "United States of America",
                                      city_meta["Country"]
                                     )

# construct queries from the csv
city_meta["city_clean"] = city_meta["City"].apply(lambda x: x.split(",")[0])
city_meta["level_0_admin"] = np.where(city_meta["city_clean"]=="Taipei", "Taiwan", city_meta["country_clean"])
city_meta["level_0_admin"] = np.where(city_meta["country_clean"]=="United States of America", 
                                      city_meta["State/Province"], 
                                      city_meta["level_0_admin"])

city_meta["city_query"] = city_meta.apply(lambda x: x["city_clean"]+", " + x["level_0_admin"], axis = 1)
city_meta["city_query"] = np.where(city_meta["country_clean"]=="United States of America", 
                                   city_meta["city_query"]+", USA",
                                   city_meta["city_query"])

def get_finished():
    files = os.listdir(RAW_FOLDER)
    city_finished = [x.split(".")[0] for x in files]
    return city_finished

city_finished = get_finished()

for i, q in enumerate(city_meta["city_query"].values):
    cityabbrlower = city_meta.at[i, "city_clean"].lower().replace(" ", "")
    if cityabbrlower in city_finished:
        continue
    else:
        try:
            boundary = ox.geocoder.geocode_to_gdf(q)
            print("Finished download")
            print(boundary.crs)
            if "4326" not in str(boundary.crs):
                boundary = boundary.to_crs("EPSG:4326")
                print("Finished reprojection")
            boundary.to_file(os.path.join(RAW_FOLDER, f"{cityabbrlower}.geojson"), 
                                driver = "GeoJSON")
            print(cityabbrlower, ": DONE")
            print("*"*100)
        except:
            print("Error with current city: ", cityabbrlower)
            time.sleep(10)