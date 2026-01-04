import os
import pandas as pd
import numpy as np
from glob import glob
import gspread
from haversine import haversine, Unit
import h3
from tqdm import tqdm

CURATED_FOLDER_TARGET = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifier_agg_hex"
if not os.path.exists(CURATED_FOLDER_TARGET):
    os.makedirs(CURATED_FOLDER_TARGET)
CURATED_FOLDER_SOURCE = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifier_agg"
RAW_PATH = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{city}/gsvmeta/{city}_meta.csv"

label = {0: 'Accra', 1: 'Amsterdam', 2: 'Antwerp', 3: 'Astrakhan', 4: 'Athens', 5: 'Auckland', 6: 'Bacolod', 7: 'Bangalore', 8: 'Bangkok', 9: 'Belgrade', 10: 'Belo Horizonte', 11: 'Berezniki', 12: 'Berlin', 13: 'Bogotá', 14: 'Boston', 15: 'Brussels', 16: 'Budapest', 17: 'Buenos Aires', 18: 'Capetown', 19: 'Cebu City', 20: 'Chicago', 21: 'Cirebon', 22: 'Cleveland', 23: 'Cochabamba', 24: 'Copenhagen', 25: 'Culiacan', 26: 'Curitiba', 27: 'Delhi', 28: 'Denver', 29: 'Detroit', 30: 'Dhaka', 31: 'Dubai', 32: 'Dzerzhinsk', 33: 'Florianopolis', 34: 'Fukuoka', 35: 'Gaborone', 36: 'Gainesville, FL', 37: 'Gombe', 38: 'Guadalajara', 39: 'Guatemala City', 40: 'Hindupur', 41: 'Hong Kong', 42: 'Houston', 43: 'Hyderabad', 44: 'Ilheus', 45: 'Istanbul', 46: 'Jaipur', 47: 'Jakarta', 48: 'Jalna', 49: 'Jequie', 50: 'Jerusalem', 51: 'Johannesburg', 52: 'Kampala', 53: 'Kanpur', 54: 'Kaunas', 55: 'Kigali', 56: 'Killeen', 57: 'Kozhikode', 58: 'Kuala Lumpur', 59: 'Kyiv', 60: 'Lagos', 61: 'Le Mans', 62: 'Lima', 63: 'London', 64: 'Los Angeles', 65: 'Madrid', 66: 'Malegaon', 67: 'Manchester', 68: 'Manila', 69: 'Medan', 70: 'Metro Manila', 71: 'Mexico City', 72: 'Miami', 73: 'Milan', 74: 'Minneapolis', 75: 'Modesto', 76: 'Montreal', 77: 'Moscow', 78: 'Mumbai', 79: 'Munich', 80: 'Nagoya', 81: 'Nairobi', 82: 'New York', 83: 'Okayama', 84: 'Palembang', 85: 'Palermo', 86: 'Palmas', 87: 'Parbhani', 88: 'Parepare', 89: 'Paris', 90: 'Philadelphia', 91: 'Portland, OR', 92: 'Pune', 93: 'Quito', 94: 'Rajshahi', 95: 'Raleigh', 96: 'Reynosa', 97: 'Ribeirao Preto', 98: 'Rio de Janeiro', 99: 'Rome', 100: 'Rovno', 101: 'Saidpur', 102: 'Saint Petersburg', 103: 'San Francisco', 104: 'Santiago', 105: 'Sao Paulo', 106: 'Seoul', 107: 'Sheffield', 108: 'Singapore', 109: 'Sitapur', 110: 'Stockholm', 111: 'Sydney', 112: 'Taipei', 113: 'Tel Aviv', 114: 'Thessaloniki', 115: 'Tokyo', 116: 'Toledo', 117: 'Toronto', 118: 'Tyumen', 119: 'Valledupar', 120: 'Victoria', 121: 'Vienna', 122: 'Vijayawada', 123: 'Warsaw', 124: 'Wellington', 125: 'Yamaguchi', 126: 'Zwolle'}
label_r = {}
for k,v in label.items():
    label_r[v] = k
def get_one_city(city_upper):
    print(city_upper)
    city = city_upper.lower().replace(" ", "")
    file_name = f"c_{city}.parquet"
    pred_df = pd.read_parquet(os.path.join(CURATED_FOLDER_SOURCE, file_name))

    # load the meta data
    df_meta = pd.read_csv(RAW_PATH.format(city = city))
    df_meta['name'] = df_meta["path"].apply(lambda x: x.split("/")[-1])
    print(pred_df.shape[0])
    for res in [8,9,12]:
        df_meta[f'h3_{res}'] = df_meta.apply(lambda x: h3.geo_to_h3(x.lat, x.lon, res), axis=1)

    df_temp = pred_df.merge(df_meta.drop("panoid", axis = 1), on = "name")
    #STEP 1: summarize all
    res = 9
    df_agg = df_temp.groupby(['top_1',f'h3_{res}']).size().reset_index().rename(columns = {0:"count"})
    df_agg["img_count"] = df_agg.groupby(f"h3_{res}")["count"].transform('sum')

    df_sel = df_agg[df_agg["top_1"]==label_r[city_upper]].reset_index(drop = True)
    df_sel["correct_class_prop"] = df_sel["count"]/df_sel["img_count"]
    df_agg['max_count'] = df_agg.groupby(f"h3_{res}")["count"].transform('max')

    
    # STEP 2: identify the top 1 classification per hexagon (both true and false)
    df_model = df_agg[df_agg["max_count"]==df_agg["count"]].reset_index(drop = True)
    df_model["sel_class"] = df_model["top_1"].apply(lambda x: label[x])
    df_model['winning_prop'] = df_model['max_count']/df_model['img_count'] # if <0.5, meaning rather random

    # STEP 3: identify the top 1 false classification per hexagon
    df_agg_false = df_agg[df_agg['top_1']!=label_r[city_upper]].reset_index(drop = True)
    df_agg_false["max_count_false"] = df_agg_false.groupby("h3_9")["count"].transform("max")
    df_agg_false_top1 = df_agg_false[df_agg_false["max_count_false"]==df_agg_false["count"]].reset_index(drop = True)
    df_agg_false_top1["top_1_false_prop"] = df_agg_false_top1["max_count_false"]/df_agg_false_top1["img_count"]
    df_agg_false_top1['top_1_false'] = df_agg_false_top1['top_1'].apply(lambda x: label[x])
    df_agg_false_top1 = df_agg_false_top1[['h3_9','max_count_false', 'top_1_false_prop','top_1_false']].reset_index(drop = True)

    # step 4: get all basics
    df_basics = df_meta[['h3_9','dist_hav','ring','panoid']].drop_duplicates().dropna(subset = "h3_9")
    df_basics = df_basics[df_basics["h3_9"]!=0].reset_index(drop = True)
    df_basics = df_basics[["dist_hav", "ring", "h3_9"]].groupby("h3_9").mean().reset_index()
    # summarize all hex level information
    df_summary = df_model[["h3_9", "img_count","sel_class", "winning_prop"]]\
    .merge(df_sel[["h3_9", "correct_class_prop"]], on = "h3_9", how = "left")\
    .merge(df_basics, on = "h3_9", how = "left")\
    .merge(df_agg_false_top1, on = "h3_9", how = "left")


    df_summary["correct_class_prop"] = df_summary["correct_class_prop"].fillna(0) # meaning this hexagon has never been correctly classified
    df_summary["top_1_false_prop"] = df_summary["top_1_false_prop"].fillna(0) # when all are correct, nothing wrong
    df_summary['ring'] = df_summary['ring'].fillna(9)
    return df_summary



city_meta = pd.read_csv("../city_meta.csv")
cityls = city_meta['City'].unique()

# allcity_result = []
for cityupper in tqdm(cityls[:60]):
    df_summary = get_one_city(cityupper)
    city = cityupper.lower().replace(" ", "")
    df_summary.to_parquet(os.path.join(CURATED_FOLDER_TARGET, f"c_{city}_hex.parquet"), index = False)
    # df_summary["source_city"] = city
    print("Done with City: ", city)
    print("*"*100)
    # allcity_result.append(df_summary)
    
# allcity_result = pd.concat(allcity_result).reset_index(drop = True)
# allcity_result.to_parquet(os.path.join(CURATED_FOLDER_TARGET, "c_hex_agg_all.parquet"), index = False)