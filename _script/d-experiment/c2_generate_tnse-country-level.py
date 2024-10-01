"""This code uses each country internal data to make TNSE rather than global TNSE.
Focusing on level 12"""
import os
import pandas as pd
from tqdm import tqdm
from sklearn import manifold
from sklearn.preprocessing import StandardScaler


def get_std(df_seg_update, variables_remain):
    scaler = StandardScaler().fit(df_seg_update[variables_remain])
    data = scaler.transform(df_seg_update[variables_remain])
    return data


def get_tsne(data, n_components=2):

    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
DATA_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_hex"
FILES = os.listdir(DATA_FOLDER)
# EXPORT_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_long_hex_country_level" # this is only the within city version
EXPORT_FOLDER = f"{ROOTFOLDER}/_curated/c_seg_long_hex_country_level_all"
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)
    print("folder made")

# FILENAME = "c_seg_cat=31_res={res}.parquet"
# FILENAME_WITHIN = "c_seg_cat=31_res={res}_withincity.parquet"

FILENAME = "c_seg_long_cat=31_res={res}.parquet"
FILENAME_WITHIN = "c_seg_long_cat=31_res={res}_withincity.parquet"

variables = [
    "bike",
    "building",
    "bus",
    "car",
    "grass",
    "house",
    "installation",
    "lake+waterboday",
    "light",
    "mountain+hill",
    "other",
    "person",
    "pole",
    "railing",
    "road",
    "shrub",
    "sidewalk",
    "signage",
    "sky",
    "sportsfield",
    "table+chair",
    "tower",
    "traffic light",
    "trashcan",
    "tree",
    "truck",
    "van",
    "wall",
    "window",
]
index_cols = ["year_group","city_lower", "hex_id", "img_count", "res"]


def export_tnse(res, df, variables=variables):
    print("Current dataset length: ", df.shape[0])
    # standardize the data

    data = get_std(df, variables)
    print("finish basic standardization")
    # get tsne
    tsne_data = get_tsne(data)  # this takes very long time.
    print("finish tsne")
    # save the tsne_data once done
    tsne_df = pd.DataFrame(tsne_data, columns=["tsne_1", "tsne_2"]).reset_index(
        drop=True
    )
    print(tsne_df.head())
    tsne_df = pd.concat([df[index_cols], tsne_df], axis=1)
    print("finished concatination.")
    print(tsne_df.head())

    return tsne_df

# load the meta file first
meta = pd.read_csv("../city_meta.csv")
meta['city_lower'] = meta['City'].apply(lambda x: x.lower().replace(" ", ""))
country_list = meta['country_clean'].unique()


for filename in [FILENAME]:
    for res in [8]:
        print("Now processing resolution: ", res)
        file_name = filename.format(res=res)
        df_all = pd.read_parquet(os.path.join(DATA_FOLDER, file_name))
        for country in tqdm(country_list):
            city_to_work = meta[meta['country_clean']==country]['city_lower'].unique()
            print(city_to_work)
            temp = df_all[df_all['city_lower'].isin(city_to_work)].reset_index(drop = True)                        
            tsne_df = export_tnse(res=res, df=temp)
            tsne_df.to_parquet(os.path.join(EXPORT_FOLDER, "c_{country}_hex={res}_tsne.parquet".format(
                country = country,
                res = res)))
        print("resoulation: ", res, " done")
        print("*"*100)
            
                               
                                            
# for res in [8,9,12]:
#     for filename in [FILENAME]:
#         export_tnse(res=res, filename=filename)
        
