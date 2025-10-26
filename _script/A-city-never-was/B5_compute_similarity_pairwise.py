from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import numpy as np
import gc
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gc

RES_EXCLUDE = 11

# Constants
ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
CURATED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob"

RAW_PATH = (
    "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{city}/gsvmeta/{city}_meta.csv"
)


CURATE_FOLDER_SOURCE = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary"
)
# CURATE_FOLDER_EXPORT = (
#     "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity"
# )
CURATE_FOLDER_EXPORT = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair"

filename = "prob_city={city_name}_res_exclude={res_exclude}.parquet"  # to load

if not os.path.exists(CURATE_FOLDER_EXPORT):
    os.makedirs(CURATE_FOLDER_EXPORT)

vector_ls = [str(x) for x in range(0, 127)]


def create_city_pair_ls():
    city_meta = pd.read_csv("/home/yuanzf/uvi-time-machine/_script/city_meta.csv")
    city_ls = city_meta.City.values
    pair_ls = np.array(np.meshgrid(city_ls, city_ls)).T.reshape(-1, 2)
    pair_ls = pair_ls[pair_ls[:, 0] != pair_ls[:, 1]]  # exclude same city
    return pair_ls


def compute_similarity(city_1, city_2, res_sel=6):
    """
    Compute the cosine similarity between two cities at a given resolution.
    include_self: whether to include the similarity within same city cells.
    """

    print(f"Processing res_sel={res_sel}")
    files = glob(
        CURATE_FOLDER_SOURCE
        + "/"
        + filename.format(city_name=city_1, res_exclude=RES_EXCLUDE)
    ) + glob(
        CURATE_FOLDER_SOURCE
        + "/"
        + filename.format(city_name=city_2, res_exclude=RES_EXCLUDE)
    )
    print(len(files))
    df_all = []
    for f in files:
        temp = pd.read_parquet(f)
        temp = temp[temp.res == res_sel].reset_index(drop=True)
        temp["city"] = os.path.basename(f).split("_")[1].replace("city=", "")
        df_all.append(temp)
    df_all = pd.concat(df_all).drop_duplicates("hex_id").reset_index(drop=True)
    df_all = df_all.drop(columns=["res"])
    print("Data loaded", df_all.shape[0])
    n_cells = df_all.shape[0]
    X = df_all[vector_ls].values
    # create a new dataframe that has shape of (n_cells, 2) to store the similarity matrix
    # compute the similarity matrix
    similarity_matrix = cosine_similarity(X)
    print("Similarity matrix computed", similarity_matrix.shape)

    # only keep the upper triangle of the matrix
    similarity_matrix = np.triu(similarity_matrix, k=1)
    print("Upper triangle extracted", similarity_matrix.shape)

    gc.collect()
    hex_ls = df_all.hex_id.values
    similarity_df = pd.DataFrame(similarity_matrix, index=hex_ls, columns=hex_ls)
    gc.collect()
    similarity_df = similarity_df.stack()
    similarity_df = pd.DataFrame(similarity_df).reset_index()
    similarity_df.columns = ["hex_id1", "hex_id2", "similarity"]
    city1_folder = "temp_city1=" + city_1
    os.makedirs(os.path.join(CURATE_FOLDER_EXPORT, city1_folder), exist_ok=True)
    similarity_df.to_parquet(
        os.path.join(
            CURATE_FOLDER_EXPORT,
            city1_folder,
            f"similarity_city2={city_2}_res={res_sel}.parquet",
        )
    )
    print("City similarity computed and saved for ", city_1, city_2)


def remove_dups(city1, res_sel=6):
    # read all files that start with temp_similarity_city1={city1}
    files = glob(
        CURATE_FOLDER_EXPORT
        + "/temp_city1="
        + city1
        + "/similarity_city2=*_res="
        + str(res_sel)
        + ".parquet"
    )
    df_all = []
    for f in files:
        temp = pd.read_parquet(f)
        df_all.append(temp)
    df_all = pd.concat(df_all).reset_index(drop=True)
    # remove duplicates
    df_all = (
        df_all[df_all.hex_id1 != df_all.hex_id2]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_all.to_parquet(
        os.path.join(
            CURATE_FOLDER_EXPORT,
            f"similarity_city={city1}_res={res_sel}.parquet",
        )
    )
    print("Final similarity saved for ", city1)
    # remove temp folder
    temp_folder = os.path.join(CURATE_FOLDER_EXPORT, "temp_city1=" + city1)
    for f in glob(temp_folder + "/*.parquet"):
        os.remove(f)
    os.rmdir(temp_folder)
    print("Temp folder removed for ", city1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res_sel",
        type=int,
        default=6,
        help="Resolution to compute similarity",
    )
    args = parser.parse_args()
    res_sel = args.res_sel

    city_pair_ls = create_city_pair_ls()
    n_pairs = city_pair_ls.shape[0]
    print(f"Total city pairs to process: {n_pairs}")

    for i in tqdm(range(n_pairs)):
        city_1 = city_pair_ls[i, 0]
        city_2 = city_pair_ls[i, 1]
        compute_similarity(city_1, city_2, res_sel=res_sel)

    # After all pairs are processed, remove duplicates for each city
    unique_cities = np.unique(city_pair_ls)
    for city in unique_cities:
        remove_dups(city, res_sel=res_sel)


if __name__ == "__main__":
    main()
