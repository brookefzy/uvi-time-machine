import os
import pandas as pd
import numpy as np
from glob import glob
import gspread
from haversine import haversine, Unit
import h3
from tqdm import tqdm
from shapely.geometry import Polygon
import geopandas as gpd

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
DATA_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_hex"
BOUND_FOLDER = f"{ROOTFOLDER}/_raw/r_boundary_osm"

# FILENAME = "c_seg_cat={n_cat}_res={res}.parquet"
# FILENAME_WITHIN = "c_seg_cat=31_res={res}_withincity.parquet"
FILENAME = "c_seg_long_cat={n_cat}_res={res}.parquet"
FILENAME_WITHIN = "c_seg_long_cat=31_res={res}_withincity.parquet"

def cell_to_shapely(cell):
    coords = h3.h3_to_geo_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)

def get_data_within_bound(df):
    df_within = []
    fullcity = df["city_lower"].unique().tolist()
    for cityabbr in tqdm(fullcity):
        if "," in cityabbr:
            city_short = cityabbr.split(",")[0]
        else:
            city_short = cityabbr
        bound = gpd.read_file(os.path.join(BOUND_FOLDER, f"{city_short}.geojson"))
        sample = df[df["city_lower"]==cityabbr].reset_index(drop = True)
        h3_geoms = sample['hex_id'].apply(lambda x: cell_to_shapely(x))
        df_sel_gdf = gpd.GeoDataFrame(sample, geometry = h3_geoms)
        df_sel_gdf.crs = "EPSG:4326"
        df_sel_gdf_within = gpd.sjoin(df_sel_gdf, bound[['geometry']])
        df_sel_gdf_within['city_lower'] = cityabbr
        df_within.append(df_sel_gdf_within.drop(['index_right','geometry'], axis=1))
    df_within = pd.concat(df_within).reset_index(drop = True)
    df_within.to_parquet(os.path.join(DATA_FOLDER, FILENAME_WITHIN.format(res = res)))
    print("Done saving within city data")
    return df_within

for res in [12]:
    df = pd.read_parquet(os.path.join(DATA_FOLDER, FILENAME.format(n_cat = 31, res = res)))
    df_within = get_data_within_bound(df)
    df_within.to_parquet(
        os.path.join(DATA_FOLDER, FILENAME_WITHIN.format(res = res)), 
                         index = False)
    print("Done saving within city data for resolution: ", res)

# check missing cities
# df = pd.read_parquet(os.path.join(DATA_FOLDER, FILENAME.format(n_cat = 31, res = 9)))
# fullcity = df["city_lower"].unique().tolist()
# original_city_ls = 
