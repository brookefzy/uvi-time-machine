import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import h3
import geopandas as gpd
import argparse
from shapely.geometry import Polygon
import h3

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
GSVROOT = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb"
ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
DATA_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_hex"


# res = 9
N_CAT = 27
BOUND_FOLDER = f"{ROOTFOLDER}/_raw/r_boundary_osm"
# FILENAME = "c_seg_cat={n_cat}_res={res}.parquet"
FILENAME_WITHIN = "c_seg_cat={n_cat}_res={res}_withincity.parquet"

def load_hex(res, n_cat):
    df = pd.read_parquet(f"{DATA_FOLDER}/{FILENAME_WITHIN.format(n_cat=n_cat, res=res)}")
    return df

def get_road_segment_one_city(citylower):
    road = gpd.read_file(f"{GSVROOT}/{citylower}/road/osm.geojson")
    road = road.to_crs(epsg=4326)
    # load h3 index with panoid downloaded
    hexdf = load_hex(res, N_CAT)
    hexdf_sel = hexdf[hexdf['city_lower']==citylower].reset_index(drop = True)[['hex_id','city_lower']]
    hexgdf_sel = gpd.GeoDataFrame(hexdf_sel, 
                                geometry= hexdf_sel['hex_id'].apply(lambda x: Polygon(h3.h3_to_geo_boundary(x,geo_json=True))),
                                    crs = 'EPSG:4326')
    road_sel = gpd.overlay(road, hexgdf_sel, how='intersection')
    # get length after intersection into meter
    road_sel = gpd.overlay(road, hexgdf_sel[['geometry','hex_id']], how='intersection')
    road_sel = road_sel.to_crs(epsg=3857)
    road_sel['length_intersection_meter'] = road_sel['geometry'].length
    # road_sel['proportion'] = road_sel['length_intersection_meter']/road_sel['length_meter']

    road_sel_h3 = road_sel.groupby('hex_id').agg({'length_intersection_meter':'sum'}).reset_index()
    
    road_sel_h3.to_csv(os.path.join(EXPORT_FOLDER,f"{citylower}_road_length.csv"), index=False)

for res in [8,9]:
    hexdf = load_hex(res, N_CAT)
    EXPORT_FOLDER = f"{ROOTFOLDER}/_curated/c_hex_road/res={res}"
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    for citylower in tqdm(hexdf['city_lower'].unique()):
        get_road_segment_one_city(citylower)
        print(citylower)
    
# python /home/yuanzf/uvi-time-machine/_script/d-experiment/05_urban_sprawl_road_density.py
