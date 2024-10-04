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
import shapely
import argparse

################################################
# basic set ups
################################################
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
YEAR_GROUP = ['<2014','2014-2015','2016-2017','2018-2019', '2020-2021','2022-2023']
column_map = {
    "<2014":[2007,   2008,   2009,   2010,   2011,   2012,   2013,],
    "2014-2015":[2014, 2015],
    "2016-2017":[2016, 2017],
    "2018-2019":[2018, 2019],
    "2020-2021":[2020, 2021],
    "2022-2023":[2022, 2023]
    
}

segfiles = glob.glob(f"{EXFOLDER_SEG_LONG}/*.parquet")
OBJECT_SOURCE_FOLDER = "{CURATED_FOLDER}/{city_abbr}/*_objects.parquet"

# variables may change later
# res = 9  # resolution of hexagon
min_num_pano = 2  # minimum number of panoid per each hexagon to avoid sampling bias
# year_to_exclude = 2019  # don't use this since we want to measure the change


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
    df_all_keep = df_pano_inner[(df_pano_inner["year"] >= 2015)
    ].reset_index(drop=True)
    print(df_all_keep.shape[0], "panoids are kept")
    # assign the year_group; skip for now
    
    # get number of panoid per hex per year
    df_all_keep_hex = (
        df_all_keep.groupby(["hex_id", "year"])["panoid"]
        .nunique()
        .reset_index(name="panoid_n")
    )
    return df_all_keep_hex, df_all_keep


# def viz_result(df_all_keep_hex, object_summary, viz_variable):
#     """Optional function to visualize the result"""
#     import seaborn as sns

#     fig, ax = plt.subplots(figsize=(4, 4))
#     sns.kdeplot(
#         data=object_summary[object_summary["object_name"] == viz_variable],
#         hue="year_group",
#         x="n_per_pano",
#         fill=True,
#         palette=sns.color_palette("husl", n_colors=2),
#     )
#     plt.title(f"{viz_variable.title()} Density per Image Sample Location in a Hex")
#     sns.despine()
#     df_hex_keep = df_all_keep_hex.merge(
#         object_summary[object_summary["object_name"] == viz_variable],
#         on=["hex_id", "year_group"],
#         how="left",
#     ).fillna(0)
#     df_hex_keep[df_hex_keep["year_group"] == YEAR_GROUP[0]].plot(
#         column="n_per_pano",
#         legend=True,
#         figsize=(8, 8),
#         legend_kwds={
#             "label": f"Average {viz_variable} count per sampled images in {YEAR_GROUP[0]}",
#             "orientation": "horizontal",
#         },
#         colormap="coolwarm",
#     )
#     # remove the axis
#     plt.axis("off")
#     plt.savefig(f"{EXFOLDER_LONG}/c_object_{viz_variable}_hex.png", dpi=300)
#     plt.close()
#     # visualize the delta
#     df_hex_keep = df_hex_keep.sort_values(["hex_id", "year_group"]).reset_index(
#         drop=True
#     )
#     # calculate difference of the log transformed value
#     df_hex_keep[f"log_delta_{viz_variable}"] = df_hex_keep.groupby("hex_id")[
#         "n_per_pano"
#     ].transform(lambda x: np.log(x + 1).diff().iloc[-1])

#     # df_hex_keep['delta'] = df_hex_keep.groupby('hex_id')[f'{variable}_n'].transform(lambda x: x.diff().iloc[-1])
#     df_hex_keep[df_hex_keep["year_group"] == YEAR_GROUP[0]].plot(
#         column=f"log_delta_{viz_variable}",
#         legend=True,
#         figsize=(8, 8),
#         legend_kwds={
#             "label": f"Delta of Log({viz_variable} count per sampled images)",
#             "orientation": "horizontal",
#         },
#         colormap="coolwarm",
#         # center the colorbar to 0
#         vmin=-1.5,
#         vmax=1.5,
#     )

#     plt.axis("off")
#     plt.savefig(f"{EXFOLDER_LONG}/c_object_{viz_variable}_delta_hex.png", dpi=300)

#     plt.close()


def summarize_objects(city, res, viz=False):
    city_abbr = city.lower().replace(" ", "")
    df_all_keep_hex, df_all_keep = get_hex_basics(city_abbr, res)
    print("Done with basic hexagon setup")

    # process the prediction results
    objfiles = glob.glob(
        OBJECT_SOURCE_FOLDER.format(CURATED_FOLDER=CURATED_FOLDER, city_abbr=city_abbr)
    )
    if len(objfiles) == 0:
        print("No object files found for", city)
        return
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
    # count all observed objects across all panoids
    object_summary = (
        df_all_keep[["panoid", "year", "hex_id"]]
        .merge(df_sel, on="panoid", how="left")
        .groupby(["hex_id", "year", "object_name"])["n"]
        .sum()
        .reset_index()
        .merge(
            df_all_keep_hex[["hex_id", "panoid_n", "year"]],
            on=["hex_id", "year"],
            how="right",
        )
        .fillna(0)
    )
    object_summary.to_csv(f"{EXFOLDER_LONG}/c_object_res={res}_{city_abbr}.csv", index=False)
    # export the complete hexagon id for comparison (this is the full sample to use)
    df_all_keep_hex[["hex_id", "panoid_n", "year"]].to_csv(
        f"{EXFOLDER_LONG}/c_hex_res={res}_{city_abbr}.csv", index=False
    )
    print("Done with", city)
    # if viz:
    #     viz_result(df_all_keep_hex, object_summary, "person")
    # else:
    #     print("No visualization requested")


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--city",
        help="City name to process",
        type=str,
        default="All",
    )
    argparser.add_argument(
        "--viz",
        help="Variable to visualize",
        type=str,
        default="None",
    )
    # check data finished summary in the export folder
    args = argparser.parse_args()
    if args.viz == "None":
        viz_bool = False
    else:
        viz_bool = True
    if args.city == "All":
        city_ls = pd.read_csv("../city_meta.csv")["City"].unique().tolist()
    else:
        city_ls = [args.city]
    for res in [8, 9, 12]:
        print("Now processing resolution: ", res)
        # default is 9, so when res = 9, there is no res=9 in the file name. Currently skip 9 since it is already done
        finished_ls = glob.glob(f"{EXFOLDER_LONG}/c_hex_res={res}_*.csv")
        finished_city = [
            x.split("/")[-1].split(".")[0].split(f"c_hex_res={res}_")[1] for x in finished_ls
        ]
        city_ls = [x for x in city_ls if x.lower().replace(" ", "") not in finished_city]
        print("Number of cities to process", len(city_ls))
        for city in city_ls:
            summarize_objects(city, res, viz_bool)


if __name__ == "__main__":
    main()
# buenosaires needs to be reprocessed
# check data in Seoul and Munich