import pandas as pd
import numpy as np
import os
import sys
import geopandas as gpd

ROOT_DIR = "../../"
sys.path.append(ROOT_DIR)
from utils.citymeta import load_all

city_meta = load_all()

ROOTFOLDER = (
    "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2023/01_city-never-was/_data"
)
RAWFOLDER = f"{ROOTFOLDER}/_raw/_road_fatality/_fatality"
TRANSFORM_FOLDER = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=country"
TRANSFORM_FOLDER_CITY = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=city"
TRANSFORM_FOLDER_STATE = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=state"

# test using spatial boundary to query
shp_folder = f"{ROOTFOLDER}/_raw/r_boundary_osm"
year = 2019
uk = pd.read_csv(f"{RAWFOLDER}/UK/dft-road-casualty-statistics-collision-{year}.csv")

cityls = ["london", "manchester", "sheffield"]
city_pop = city_meta[city_meta["city_lower"].isin(cityls)][["city_lower", "urban_pop"]]

shp_gdf = []
for city in cityls:
    shp = gpd.read_file(f"{shp_folder}/{city}.geojson")
    shp["city"] = city
    shp_gdf.append(shp)
shp = pd.concat(shp_gdf)
uk_accident_gdf = gpd.GeoDataFrame(
    uk, geometry=gpd.points_from_xy(uk["longitude"], uk["latitude"]), crs="EPSG:4326"
)
city_save = gpd.sjoin(uk_accident_gdf, shp, op="within")
city_summary = (
    city_save[city_save["accident_severity"] == 1]
    .groupby(["city"])["accident_reference"]
    .nunique()
    .reset_index()
    .rename(columns={"accident_reference": "num_fatal"})
)
city_summary = city_summary.merge(city_pop, left_on="city", right_on="city_lower").drop(
    columns=["city_lower"]
)
city_summary["num_person_killed_per_lakh"] = (
    city_summary["num_fatal"] / city_summary["urban_pop"] * 100000
)
city_summary.to_csv(
    os.path.join(TRANSFORM_FOLDER_CITY, "t_road_fatality_uk.csv"), index=False
)
print(city_summary.head(5))
