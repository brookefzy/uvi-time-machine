import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import entropy

ROOT_DIR = "../"
sys.path.append(ROOT_DIR)
from utils.citymeta import load_all

N = 7  # 8
res = 9  # need to test the 8 later
N_CAT = 27
prefix_ls = ["_built_environment", ""]
ROOT = "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2023"
# ROOT = "D:/Dropbox (Personal)/Personal Work/_Projects2023"
CURATED_FOLDER = f"{ROOT}/01_city-never-was/_data/_curated/"
EXPORT_FOLDER = f"{ROOT}/01_city-never-was/_data/_curated/c_analysis"
# DATA_FOLDER = f"{CURATED_FOLDER}/c_seg_hex"
DATA_EXPORT = f"{ROOT}/01_city-never-was/_data/_curated/c_hex_cluster"
TRANSFORM_FOLDER = f"{ROOT}/01_city-never-was/_data/_transformed/t_city_profiles"
FILENAME_CROSS = "01_seg_dalys_cross.csv"
COMMENT = "_no_exposure_constrain"
y = [
    "diabetes_mellitus_cap",
    "mental_and_substance_use_disorders_cap",
    "cardiovascular_diseases_cap",
    "road_injury_cap",
    "road_injury_5-14_cap",
    "road_injury_15-29_cap",
    "num_person_killed_per_lakh",
]
exposure_cols = [
    "h3_9",
    "sidewalk_presence",
    "exposure_presence",
    "obj_bicycle",
    "obj_bus",
    "obj_car",
    "obj_motorcycle",
    "obj_person",
    "obj_train",
    "obj_truck",
    # 'obj_van',
    "city_lower",
]
featurecols = [
    "bike",
    "building",
    "bus",
    "car",
    "grass",
    "installation",
    "lake+waterboday",
    "light",
    "mountain+hill",
    "other",
    "person",
    "pole",
    "railing",
    "res",
    "road",
    "shrub",
    "sidewalk",
    "signage",
    "sky",
    "skyscraper",
    "sportsfield",
    "tower",
    "traffic light",
    "trashcan",
    "tree",
    "truck",
    "van",
    "window",
]


# cluster_df = pd.read_parquet(os.path.join(DATA_FOLDER, f"all_city_within_boundary_res=9_cluster={N}.parquet"))
# withincity = True
def get_cluster_summary(N, prefix):
    #    "c_seg_cat=31_res={res}_withincity_tsne_cluster={n}.csv".format(res = res, n = N)))
    cluster_df = pd.read_csv(
        os.path.join(DATA_EXPORT, f"allcity{prefix}_cluster={N}_res={res}{COMMENT}.csv")
    )
    cluster_df["city_lower"] = cluster_df["city_lower"].apply(
        lambda x: x.lower().replace(" ", "").split(",")[0]
    )
    # cluster_df.head()

    cluster_df_summary = (
        cluster_df.groupby(["city_lower", f"cluster_{N}"])
        .size()
        .reset_index(name="count")
        .pivot(index="city_lower", columns=f"cluster_{N}", values="count")
        .reset_index()
        .fillna(0)
    )
    # convert to wide format
    cluster_df_summary["total"] = cluster_df_summary.sum(axis=1)
    for c in cluster_df_summary.columns[1:-1]:
        cluster_df_summary[c] = cluster_df_summary[c] / cluster_df_summary["total"]
        cluster_df_summary.rename(columns={c: f"cluster_{c}"}, inplace=True)

    entropy_variables = ["cluster_" + str(i) for i in range(N)]
    cluster_df_summary["diversity"] = cluster_df_summary[entropy_variables].apply(
        lambda x: entropy(x) / np.log(N), axis=1
    )
    return cluster_df, cluster_df_summary


# cluster_df_summary.head()
def get_exposure_summary(cluster_df, exposure_df, exposure_df_2):
    for c in exposure_df.columns:
        exposure_df[c] = exposure_df[c].fillna(0)
    exposure_df_summary = (
        cluster_df[["hex_id"]]
        .merge(
            exposure_df,
            left_on="hex_id",
            right_on="h3_9",
        )
        .merge(
            exposure_df_2[
                ["hex_id", "person_exposure", "bicycle_exposure", "motorcycle_exposure"]
            ],
            right_on="hex_id",
            left_on="h3_9",
        )
        .fillna(0)
        .groupby(["city_lower"])
        .agg(
            {
                "h3_9": "nunique",
                "img": "sum",
                "panoid": "sum",
                "sidewalk_presence": "mean",
                "exposure_presence": "mean",
                "obj_bicycle": "mean",
                "obj_motorcycle": "mean",
                "obj_person": "mean",
                "obj_truck": "mean",
                "obj_car": "mean",
                "obj_bus": "mean",
                "person_exposure": "mean",
                "bicycle_exposure": "mean",
                "motorcycle_exposure": "mean",
                # 'obj_van':'mean',
            }
        )
        .reset_index()
        .rename(
            columns={"h3_9": "num_hex", "img": "img_count", "panoid": "panoid_count"}
        )
    )
    return exposure_df_summary


def main(N):
    gdp = pd.read_csv(os.path.join(CURATED_FOLDER, "c_city_profiles", "c_city_gdp.csv"))
    flux = pd.read_csv(os.path.join(TRANSFORM_FOLDER, "t_ffdas_flux_2015.csv"))
    gdp["city_lower"] = gdp["City"].apply(
        lambda x: x.lower().replace(" ", "").split(",")[0]
    )
    profiledf = flux.merge(gdp, on="city_lower", how="inner")
    exposure_df = pd.read_csv(os.path.join(EXPORT_FOLDER, "c_exposure_sidewalk_h3.csv"))

    waze_df = pd.read_csv(
        os.path.join(f"{CURATED_FOLDER}/c_waze/", "c_waze_accident_alerts_h3_9.csv")
    )
    waze_df_all = pd.read_csv(
        os.path.join(f"{CURATED_FOLDER}/c_waze/", "c_waze_major_accident_h3_9.csv")
    ).rename(
        columns={
            "major_accident_count": "waze_major_count",
            "city": "city_lower",
            f"h3_{res}": "hex_id",
        }
    )
    waze_df.rename(
        columns={
            "city": "city_lower",
            "h3_9": "hex_id",
            "count": "waze_accident_count",
        },
        inplace=True,
    )

    waze_count_df = (
        waze_df.groupby("city_lower")["waze_accident_count"]
        .sum()
        .reset_index(name="waze_accident_count")
    )
    waze_df_all_count_df = (
        waze_df_all.groupby("city_lower")["waze_major_count"]
        .sum()
        .reset_index(name="waze_major_count")
    )

    # waze_count_df
    print(profiledf["city_lower"].nunique(), " city profiles found")
    # city_meta = load_all()
    # tmp solutions here
    city_df_basics = pd.read_csv(
        os.path.join(EXPORT_FOLDER, f"c_city_full_cluster=8_basics.csv")
    )  # these are the old data
    urban_sprawl = pd.read_csv(os.path.join(EXPORT_FOLDER, "c_urban_sprawl.csv"))
    urban_sprawl = (
        urban_sprawl[["city_lower", "Sprawl_SNDi_’00-13"]]
        .reset_index(drop=True)
        .rename(columns={"Sprawl_SNDi_’00-13": "sprawl_sndi"})
    )
    urban_sprawl["city_lower"] = np.where(
        urban_sprawl["city_lower"] == "bogota", "bogotá", urban_sprawl["city_lower"]
    )
    urban_sprawl_2 = pd.read_csv(
        os.path.join(EXPORT_FOLDER, f"c_road_length_res={res}.csv")
    )
    urban_sprawl_2["city_lower"] = np.where(
        urban_sprawl_2["city_lower"] == "bogotá", "bogotá", urban_sprawl_2["city_lower"]
    )
    # urban_sprawl.head()
    urban_sprawl_2_city = (
        urban_sprawl_2.groupby("city_lower")["length_intersection_meter"]
        .mean()
        .reset_index()
    )

    exposure_path = f"{ROOT}/01_city-never-was/_data/_curated/c_object_crossectional/exposure_measure/exposure_res={res}.parquet"
    exposure_df_2 = pd.read_parquet(exposure_path)
    exposure_df_2 = exposure_df_2.fillna(0).rename(columns={"hex_id_": "hex_id"})

    def get_export_for_analysis(N):
        crossdf = pd.read_csv(
            os.path.join(CURATED_FOLDER, "c_analysis", FILENAME_CROSS)
        ).drop(["urban_pop"], axis=1)
        crossdf["city_lower"] = crossdf["city_lower"].apply(
            lambda x: x.lower().replace(" ", "").split(",")[0]
        )
        crossdf = crossdf[crossdf["res"] == 9].reset_index(drop=True)
        y_crossed = crossdf[y + ["city_lower"]].drop_duplicates()

        # cluster_df, cluster_df_summary = get_cluster_summary(N, prefixfull)
        # for prefix in prefix_ls:
        ROOTFOLDER = "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2023/01_city-never-was"
        DATA_FOLDER = f"{ROOTFOLDER}/_data/_curated/c_seg_hex"
        FILENAME_WITHIN = (
            "c_seg_cat={n_cat}_res={res}_withincity{prefixfull}_tsne"  # temporary test
        )

        for prefix in ["_built_environment", ""]:
            cluster_df, cluster_df_summary = get_cluster_summary(N, prefix)
            df_ori_within = pd.read_csv(
                os.path.join(
                    DATA_FOLDER,
                    f"{FILENAME_WITHIN}_cluster_range{COMMENT}.csv".format(
                        prefixfull=prefix, res=res, n_cat=N_CAT
                    ),
                )
            )
            df_ori_within_city = (
                df_ori_within[featurecols + ["city_lower"]]
                .fillna(0)
                .groupby("city_lower")
                .mean()
                .reset_index()
            )
            exposure_df_summary = get_exposure_summary(
                cluster_df, exposure_df, exposure_df_2
            )
            city_df = (
                cluster_df_summary.merge(profiledf, on="city_lower", how="left")
                .merge(city_df_basics, on="city_lower", how="inner")
                .merge(
                    y_crossed,
                    on="city_lower",
                    how="inner",
                    suffixes=("_city", "_country"),
                )
                .merge(waze_count_df, on="city_lower", how="left")
                .merge(exposure_df_summary, on="city_lower")
                .drop("Unnamed: 0", axis=1)
                .merge(waze_df_all_count_df, on="city_lower", how="left")
                .drop_duplicates("City")
                .reset_index(drop=True)
                .merge(df_ori_within_city, on="city_lower", how="left")
                .merge(urban_sprawl, on="city_lower", how="left")
                .merge(urban_sprawl_2_city, on="city_lower", how="left")
            )
            # country level merge
            hex_mergedf = (
                cluster_df.merge(profiledf, on="city_lower", how="left")
                .merge(
                    city_df_basics.drop("Unnamed: 0", axis=1),
                    on="city_lower",
                    how="inner",
                )
                .drop("cluster_7", axis=1)
                .merge(df_ori_within, on=["city_lower", "hex_id"])
                .merge(waze_df, on=["city_lower", "hex_id"], how="left")
                .merge(waze_df_all, on=["city_lower", "hex_id"], how="left")
                .merge(
                    exposure_df[exposure_cols],
                    right_on=["city_lower", f"h3_{res}"],
                    left_on=["city_lower", "hex_id"],
                )
                .drop_duplicates(["city_lower", "hex_id"])
                .reset_index(drop=True)
                .merge(
                    exposure_df_2[
                        [
                            "hex_id",
                            "person_exposure",
                            "bicycle_exposure",
                            "motorcycle_exposure",
                        ]
                    ],
                    left_on="hex_id",
                    right_on="hex_id",
                    how="left",
                )
                .merge(urban_sprawl_2, on=["city_lower", "hex_id"], how="left")
            )
            for c in exposure_cols + [
                "person_exposure",
                "bicycle_exposure",
                "motorcycle_exposure",
            ]:
                hex_mergedf[c] = hex_mergedf[c].fillna(0)

            hex_mergedf["waze_accident_count"] = hex_mergedf[
                "waze_accident_count"
            ].fillna(0)
            hex_mergedf["waze_total_accident_city"] = hex_mergedf.groupby("city_lower")[
                "waze_accident_count"
            ].transform("sum")
            hex_mergedf["include"] = np.where(
                hex_mergedf["waze_total_accident_city"] > 0, 1, 0
            )

            assert (
                hex_mergedf[hex_mergedf["center_lat"].isna()]["city_lower"].nunique()
                == 0
            )
            hex_mergedf.to_csv(
                os.path.join(
                    EXPORT_FOLDER,
                    f"c_hex{prefix}_full_cluster={N}_ncat={N_CAT}{COMMENT}.csv",
                ),
                index=False,
            )
            city_df.to_csv(
                os.path.join(
                    EXPORT_FOLDER,
                    f"c_city{prefix}_full_cluster={N}_ncat={N_CAT}{COMMENT}.csv",
                ),
                index=False,
            )
            print(f"Exported {N} clusters for {prefix}")
            print("*" * 100)

    get_export_for_analysis(N)


main(N)
