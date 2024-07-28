import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

CURATED_FOLDER_TARGET = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifier_agg"
if not os.path.exists(CURATED_FOLDER_TARGET):
    os.makedirs(CURATED_FOLDER_TARGET)
# CURATED_FOLDER_SOURCE = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifier_infer"


TRANSFORMED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier"

# TO_PRED_CITYLS = os.listdir(TRANSFORMED_FOLDER)
# CITYLS = os.listdir(CURATED_FOLDER_SOURCE)
# ALLPRED = glob(CURATED_FOLDER_SOURCE+"/*/*.parquet")
# print("Total cities: ", len(CITYLS))
# print("Total files: ", len(ALLPRED))

# TO_PRED_CITYLS = [x[:-4] for x in TO_PRED_CITYLS]
# REMAIN = [x for x in TO_PRED_CITYLS if not x in CITYLS]
# loop through all cities' results and summarize the prediction
# remove the images share same panoid with training set
# def clean_city(city):
#     """Remove the test data that share same panoid from the training set to avoid data leakage"""
#     cityabbr = city.lower().replace(" ", "")
#     yolofolder = f"/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8/train/{city}"
#     allfiles = glob(yolofolder+"/*.jpg")
#     df_train = pd.DataFrame(allfiles, columns = ["path"])
#     df_train['name'] = df_train["path"].apply(lambda x: x.split("/")[-1])
#     df_train['panoid'] = df_train["name"].apply(lambda x: x[:22])
#     return df_train

city_meta = pd.read_csv("../city_meta.csv")
cityls = city_meta['City'].unique()
# allresults = []
# for city in tqdm(cityls):
#     cityabbr = city.lower().replace(" ", "")
#     city_pred = glob(CURATED_FOLDER_SOURCE +"/"+cityabbr+"/*.parquet")
#     temp_df = []
#     for p in tqdm(city_pred):
#         temp = pd.read_parquet(p)
#         temp_df.append(temp)
#     temp_df = pd.concat(temp_df).reset_index(drop = True)
#     print("Current data shape: ", temp_df.shape[0])
#     df_train = clean_city(city)
    
#     temp_df["panoid"] = temp_df["name"].apply(lambda x: x[:22])
#     temp_df = temp_df[~temp_df['panoid'].isin(df_train['panoid'].values)]
    
#     temp_df.to_parquet(os.path.join(CURATED_FOLDER_TARGET, f"c_{cityabbr}.parquet"), index = False)
#     print("city saved: ", city)
#     print("city remaining shape: ", temp_df.shape[0])
#     print("*"*100)
#     temp_sum = temp_df.groupby('top_1').size().reset_index().rename(columns = {0:"count"})
#     temp_sum['city'] = cityabbr
#     allresults.append(temp_sum)
# allresults = pd.concat(allresults).reset_index(drop = True)
# allresults.to_parquet(os.path.join(CURATED_FOLDER_TARGET, "c_classifier_city_agg.parquet"), index = False)


#### TEMPORARY FIX #####
CURATED_FOLDER_SOURCE = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifier_agg"
def clean_city(city):
    
    """Remove the test data that share same panoid from the training set to avoid data leakage"""
    cityabbr = city.lower().replace(" ", "")
    yolofolder = f"/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8/train/{city}"
    allfiles = glob(yolofolder+"/*.jpg")
    df_train = pd.DataFrame(allfiles, columns = ["path"])
    df_train['name'] = df_train["path"].apply(lambda x: x.split("/")[-1])
    df_train['panoid'] = df_train["name"].apply(lambda x: x[:22])
    pred_df = pd.read_parquet(os.path.join(CURATED_FOLDER_SOURCE, "c_"+cityabbr+".parquet"))
    print("Original data shape: ", pred_df.shape[0])
    pred_df["panoid"] = pred_df["name"].apply(lambda x: x[:22])
    pred_df = pred_df[~pred_df['panoid'].isin(df_train['panoid'].values)]
    print("remaining data shape: ", pred_df.shape[0])
    pred_df.to_parquet(os.path.join(CURATED_FOLDER_SOURCE, "c_"+cityabbr+".parquet"))
    print("*"*100)
    return pred_df

allresults = []
for city in tqdm(city_meta["City"].unique()[1:]):
    print("City is: ", city)
    pred_df = clean_city(city)
    pred_sum = pred_df.groupby("top_1").size().reset_index().rename(columns = {0:"count"})
    allresults.append(pred_sum)
    
# allresults = pd.concat(allresults).reset_index(drop = True)
# allresults["top_1_pre"] = allresults["top_1"].apply(lambda x: label[x])
# allresults["total_count"] = allresults.groupby("city")["count"].transform("sum")
# allresults["pre_pro"] = allresults["count"]/allresults["total_count"]

# allresults.to_parquet(os.path.join(CURATED_FOLDER_TARGET, "c_classifier_city_agg.parquet"), index = False)