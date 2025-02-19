import os
import pandas as pd
import numpy as np
from glob import glob
import gspread
import h3
from tqdm import tqdm

# from fcmeans import FCM
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
from shapely.geometry import Polygon
import geopandas as gpd

# import kmean
from sklearn.cluster import KMeans
from scipy.spatial import distance
import os

# load tsne data
DATA_FOLDER = "/group/geog_pyloo/08_GSV/data/_curated/c_seg_hex/c_seg_hex"
BOUNDARY_FOLDER = "/group/geog_pyloo/08_GSV/data/_raw/r_boundary_osm"

GRAPHIC_PATH = "/group/geog_pyloo/08_GSV/_graphic/cluster/allcities_c={n}"
GRAPHIC_PATH_INNER = "/group/geog_pyloo/08_GSV/_graphic/cluster/allcities_inner_c={n}"
FILENAME = "c_seg_cat={N_CAT}_res={res}_tsne.parquet"


def cell_to_shapely(cell):
    coords = h3.h3_to_geo_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)


# loop through all cities and save the graphic and data
def get_result(df, city, n):
    # make folder
    if not os.path.exists(GRAPHIC_PATH.format(n=n)):
        os.makedirs(GRAPHIC_PATH.format(n=n))
    if not os.path.exists(GRAPHIC_PATH_INNER.format(n=n)):
        os.makedirs(GRAPHIC_PATH_INNER.format(n=n))
    print("folder generated")
    cityabbr = city.lower().replace(" ", "")

    sample = df[df["city_lower"] == cityabbr].reset_index(drop=True)
    h3_geoms = sample["hex_id"].apply(lambda x: cell_to_shapely(x))
    df_sel_gdf = gpd.GeoDataFrame(sample[["hex_id", f"cluster_{n}"]], geometry=h3_geoms)
    df_sel_gdf.crs = "EPSG:4326"

    df_sel_gdf[f"cluster_{n}"] = df_sel_gdf[f"cluster_{n}"].astype(str)

    # load boundary
    cityabbrshort = cityabbr.split(",")[0]
    boundary = gpd.read_file(os.path.join(BOUNDARY_FOLDER, f"{cityabbrshort}.geojson"))
    boundary = boundary.to_crs("EPSG:4326")
    old_count = df_sel_gdf.shape[0]
    print("number of hex: ", old_count)
    df_sel_gdf_intersect = gpd.sjoin(df_sel_gdf, boundary[["geometry"]], how="inner")

    new_count = df_sel_gdf_intersect.shape[0]
    print("number of hex after limiting: ", new_count)
    if new_count / old_count < 0.4:
        print("Too few hex in the city")
        with open("problem_city.txt", "a") as f:
            f.write(city + ": too few sample problem" + "\n")
    df_sel_gdf.plot(figsize=(10, 10), column=f"cluster_{n}", legend=True, linewidth=0.1)
    plt.title(city)
    plt.savefig(
        os.path.join(GRAPHIC_PATH, f"{city}_cluster={n}-tsn-res=9.png"),
        dpi=200,
        bbox_inches="tight",
    )

    df_sel_gdf_intersect.plot(
        figsize=(10, 10), column=f"cluster_{n}", legend=True, linewidth=0.1
    )
    plt.title(city)
    # hide the plot grid
    # plt.grid(False)
    # hide the axis
    plt.axis("off")
    plt.savefig(
        os.path.join(GRAPHIC_PATH_INNER, f"{city}_cluster={n}-tsn-res=9.png"),
        dpi=200,
        bbox_inches="tight",
    )

    return df_sel_gdf, df_sel_gdf_intersect


def generate_cluster(df, n):
    res = 9

    data = df[["tsne_1", "tsne_2"]].copy()

    km = KMeans(n_clusters=n, random_state=0)
    km.fit(data)
    df[f"cluster_{n}"] = km.labels_
    df[f"cluster_{n}"] = df[f"cluster_{n}"].astype(str)

    allgdf = []
    allgdf_intersect = []
    for city in tqdm(df["city_lower"].unique()):
        print(city)
        # try:
        df_sel_gdf, df_sel_gdf_intersect = get_result(df, city, n)
        df_sel_gdf["city_lower"] = city
        df_sel_gdf_intersect["city_lower"] = city
        allgdf.append(df_sel_gdf.drop(columns=["geometry"], axis=1))
        allgdf_intersect.append(df_sel_gdf_intersect.drop(columns=["geometry"], axis=1))
        print("*" * 100)
    allgdf = pd.concat(allgdf).reset_index(drop=True)
    allgdf_intersect = pd.concat(allgdf_intersect).reset_index(drop=True)
    allgdf.to_csv(
        os.path.join(DATA_FOLDER, f"allcity_cluster={n}-tsn-res={res}.csv"), index=False
    )
    allgdf_intersect.to_csv(
        os.path.join(DATA_FOLDER, f"allcity_cluster={n}-tsn-res={res}_inner.csv"),
        index=False,
    )


def main():
    res = 9
    df = pd.read_parquet(os.path.join(DATA_FOLDER, FILENAME.format(res=res)))
    print(df.head())
    for n in [8, 10, 13, 15]:

        generate_cluster(df, n)
        print(f"finish {n}")
    print("finish all")


main()
