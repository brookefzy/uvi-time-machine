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
import argparse

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
DATA_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_hex"
BOUND_FOLDER = f"{ROOTFOLDER}/_raw/r_boundary_osm"

FILENAME = "c_seg_cat={n_cat}_res={res}.parquet"
FILENAME_WITHIN = "c_seg_cat={n_cat}_res={res}_withincity.parquet"

FILENAME_LONG = "c_seg_long_cat={n_cat}_res={res}.parquet"
FILENAME_WITHIN_LONG = "c_seg_long_cat={n_cat}_res={res}_withincity.parquet"
# FILENAME = "c_seg_long_cat={n_cat}_res={res}.parquet"
# FILENAME_WITHIN = "c_seg_long_cat={n_cat}_res={res}_withincity.parquet"


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
        sample = df[df["city_lower"] == cityabbr].reset_index(drop=True)
        h3_geoms = sample["hex_id"].apply(lambda x: cell_to_shapely(x))
        df_sel_gdf = gpd.GeoDataFrame(sample, geometry=h3_geoms)
        df_sel_gdf.crs = "EPSG:4326"
        df_sel_gdf_within = gpd.sjoin(df_sel_gdf, bound[["geometry"]])
        df_sel_gdf_within["city_lower"] = cityabbr
        df_within.append(df_sel_gdf_within.drop(["index_right", "geometry"], axis=1))
    df_within = pd.concat(df_within).reset_index(drop=True)
    print("Done saving within city data")
    return df_within


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", 
                        "-l",
                        type=bool, default=False)
    process_long = parser.parse_args().long
    N_CAT = 27 # this is a constant
    
    for res in [8, 9]:
        if process_long:
            filename = FILENAME_LONG.format(n_cat=N_CAT, res=res)
            filename_within = FILENAME_WITHIN_LONG.format(n_cat=N_CAT, res=res)
        else:
            filename = FILENAME.format(n_cat=N_CAT, res=res)
            filename_within = FILENAME_WITHIN.format(n_cat=N_CAT, res=res)
        df = pd.read_parquet(
            os.path.join(DATA_FOLDER, filename)
        )
        df_within = get_data_within_bound(df)
        df_within.to_parquet(
            os.path.join(DATA_FOLDER, filename_within),
            index=False,
        )
        print("Done saving within city data for resolution: ", res)

if __name__ == "__main__":
    main()
#python /home/yuanzf/uvi-time-machine/_script/d-experiment/c1b_filter_to_city.py