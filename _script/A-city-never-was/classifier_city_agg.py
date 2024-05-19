import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

CURATED_FOLDER_TARGET = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifier_agg"
if not os.path.exists(CURATED_FOLDER_TARGET):
    os.makedirs(CURATED_FOLDER_TARGET)
CURATED_FOLDER_SOURCE = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_infer"
TRANSFORMED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier"

TO_PRED_CITYLS = os.listdir(TRANSFORMED_FOLDER)
CITYLS = os.listdir(CURATED_FOLDER_SOURCE)
ALLPRED = glob(CURATED_FOLDER_SOURCE+"/*/*.parquet")
print("Total cities: ", len(CITYLS))
print("Total files: ", len(ALLPRED))

TO_PRED_CITYLS = [x[:-4] for x in TO_PRED_CITYLS]
REMAIN = [x for x in TO_PRED_CITYLS if not x in CITYLS]
# loop through all cities' results and summarize the prediction

allresults = []
for city in tqdm(CITYLS):
    city_pred = glob(CURATED_FOLDER_SOURCE +"/"+city+"/*.parquet")
    temp_df = []
    for p in tqdm(city_pred):
        temp = pd.read_parquet(p)
        temp_df.append(temp)
    temp_df = pd.concat(temp_df).reset_index(drop = True)
    temp_df.to_parquet(os.path.join(CURATED_FOLDER_TARGET, f"c_{city}.parquet"), index = False)
    print("city saved: ", city)
    print("*"*100)
    temp_sum = temp_df.groupby('top_1').size().reset_index().rename(columns = {0:"count"})
    temp_sum['city'] = city
    allresults.append(temp_sum)
allresults = pd.concat(allresults).reset_index(drop = True)
allresults.to_parquet(os.path.join(CURATED_FOLDER_TARGET, "c_classifier_city_agg.parquet"), index = False)