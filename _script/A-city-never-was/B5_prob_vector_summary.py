# %% conda activate py312
#  cd /scr/u/yuanzf/anaconda3/envs/py312/lib
# limited by the h3 version
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import numpy as np
import gc
import datetime
import argparse
import h3
# get h3 version
print(h3.__version__)

# %%
# Constants
ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
VALFOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir"
)
CURATED_FOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob"
)
TRAIN_TEST_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"
RAW_PATH = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{city}/gsvmeta/{city}_meta.csv"

PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
PATH_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"

CURATE_FOLDER_EXPORT = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary"
if not os.path.exists(CURATE_FOLDER_EXPORT):
    os.makedirs(CURATE_FOLDER_EXPORT)
OUTPUT_FILE_NAME = "prob_city={city}_res_exclude={res_exclude}.parquet"
    
vector_ls = [str(x) for x in range(0, 127)]


# %%
# For each image, load their lat lon and find their associated hex level 6, 7, 8 H3 id
def load_source(city):
    city_sel = city.lower().replace(" ", "")
    df_pano = pd.read_csv(PANO_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=city_sel))
    df_path = pd.read_csv(PATH_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=city_sel))
    df_pano_inner = df_pano[
            df_pano["panoid"].isin(df_path["panoid"].unique())
        ].reset_index(drop=True)
    print(df_pano_inner.shape[0], "out of ", df_pano.shape[0], "panoids have path")
    # 6,7,8 for data summary, 11, 12, 13 for data robustness
    for res in [6,7,8, 11, 12, 13]:
        try:
            df_pano_inner[f"hex_{res}"] = df_pano_inner.apply(
                lambda x: h3.geo_to_h3(x["lat"], x["lon"], res), axis=1
            )
        except:
            print("version {} does not support, geo_to_h3, use latlng_to_cell instead".format(h3.__version__))
            df_pano_inner[f"hex_{res}"] = df_pano_inner.apply(
                lambda x: h3.latlng_to_cell(x["lat"], x["lon"], res), axis=1
            )
    return df_pano_inner.drop(columns=["lat", "lon"])

def prepare_data(city):
    city_sel = city.lower().replace(" ", "")
    files = glob(os.path.join(CURATED_FOLDER, f'*/{city_sel}*.parquet'), recursive= True)
    print(len(files), "files already inferred")
    if len(files)==0:
        return None, None
    result_sel = pd.DataFrame({
        "file_path": files,
        "city": [os.path.basename(os.path.dirname(x)) for x in files]
    })
    print(len(result_sel), "files already inferred for", city_sel)
    
    df_pano_inner = load_source(city)
    
    temp = pd.concat(pd.read_parquet(result_sel.file_path.values[i]) for i in range(len(result_sel.file_path.values)))
    temp['panoid'] = temp['name'].apply(lambda x: x[:22])
    # get all train and test image names
    train_test = glob(os.path.join(TRAIN_TEST_FOLDER, f'*/{city}/*.jpg'), recursive= True)
    train_test = [os.path.basename(x) for x in train_test]
    train_test_panoid = [x[:22] for x in train_test]
    print(len(train_test), "train and test images for", city)
    
    # merge the prediction
    temp_merge = temp.merge(df_pano_inner.drop('id', axis = 1), on="panoid", how="inner")
    # merge the train and test for robustness
    train_test_merge = df_pano_inner[df_pano_inner["panoid"].isin(train_test_panoid)].reset_index(drop=True)
    return temp_merge, train_test_merge
    
    
def robust_exclusion(df, train_test_merge, res_exclude= None):
    """
    excluding the prediction results if the validation hex at the resolution of res_exclude sharing same as the training hex at the resolution of res_summary
    this method avoid the data leakage
    """
    if res_exclude is None:
        print("do not exclude any data.")
        df_clean = df.copy()
    else:
        print("Exclude validation images if they share the same hex as training images at resolution: ", res_exclude)
        df_clean = df[~df[f'hex_{res_exclude}'].isin(train_test_merge[f'hex_{res_exclude}'])].reset_index(drop = True)
    resulthex = []
    for res in [6,7,8]:
        
        summary = df_clean[vector_ls+[f"hex_{res}"]].groupby(f"hex_{res}").mean().reset_index()
        summary.rename(columns = {f"hex_{res}":"hex_id"}, inplace = True)
        summary['res'] = res
        resulthex.append(summary)
        print("hex", res, "done")
    resulthex = pd.concat(resulthex).reset_index(drop = True)
    gc.collect()
    return resulthex

def get_viz(resulthex, res = 8):
    viz = resulthex[resulthex.res == res].reset_index(drop = True)
    # for each hex, pick the column with the highest probability
    # viz['max'] = viz[vector_ls].max(axis = 1)
    viz['max_col'] = viz[vector_ls].idxmax(axis = 1)
    # get the second highest probability column
    viz['second_col'] = viz[vector_ls].apply(lambda x: x.drop(viz['max_col']).idxmax(), axis = 1)
    print("Top 10 classes with the highest probability")
    print(viz['second_col'].value_counts()[0:10])
    print("Top 10 classes with the second highest probability")
    print(viz['max_col'].value_counts()[0:10])
    return viz
def get_result(city):
    temp_merge, train_test_merge = prepare_data(city)
    if temp_merge is None:
        print("No data for", city)
        return None
    # result1 = robust_exclusion(temp_merge, train_test_merge, res_exclude= None)
    # result13 = robust_exclusion(temp_merge, train_test_merge, res_exclude= 13)
    for res_exclude in [None, 11, 12, 13]:
        result = robust_exclusion(temp_merge, train_test_merge, res_exclude= res_exclude)
        result.to_parquet(os.path.join(CURATE_FOLDER_EXPORT, OUTPUT_FILE_NAME.format(city=city, res_exclude=str(res_exclude))))
        print("done for", city, "res_exclude", res_exclude)
    

# %%

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--city", type=str, default="Hong Kong")
    args = args.parse_args()
    city = args.city
    finished = glob(os.path.join(CURATE_FOLDER_EXPORT, '*.parquet'), recursive= True)
    finished_city_ls = set([x.split("/")[-1].split("_")[1].replace("city=", "") for x in finished])
    print("Finished cities: ", finished_city_ls)
    if city == "all":
        city_meta = pd.read_csv("/home/yuanzf/uvi-time-machine/_script/city_meta.csv")
        city_ls = city_meta.City.values
        city_ls = [x for x in city_ls if x not in finished_city_ls]
        print("Start processing all cities")
        
        for city in tqdm(city_ls):
            # Check again here in case the processing is done by other process
            finished = glob(os.path.join(CURATE_FOLDER_EXPORT, '*.parquet'), recursive= True)
            finished_city_ls = set([x.split("/")[-1].split("_")[1].replace("city=", "") for x in finished])
            if city in finished_city_ls:
                print(city, "is already processed")
            elif city == "London":
                continue # temporary skip London, need to check thessaloniki, yamaguchi, nairobi, dubai, sanfrancisco later
            else:
                get_result(city)
    else:
        if city in finished_city_ls:
            print(city, "is already processed")
        else:
            get_result(city)

# %%

if __name__ == "__main__":
    main()
    
# to run the script, use the following command:
# python /home/yuanzf/uvi-time-machine/_script/A-city-never-was/B5_prob_vector_summary.py --city all

