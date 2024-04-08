import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import gspread
import logging
logging.basicConfig(filename='label_data.log', format='%(asctime)s %(message)s', filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 


ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/"
subs = os.listdir(ROOTFOLDER)
curated_folder = "/lustre1/g/geog_pyloo/05_timemachine/_curated"

meta_folder = "gsvmeta"
transformed_folder = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier"
if not os.path.exists(transformed_folder):
    os.makedirs(transformed_folder)
serviceaccount = "../../google_drive_personal.json"
## Constant set up
dist_thred = 25_000
train_n = 10000
test_n = 2000

sthres = 15000

# from oauth2client.service_account import ServiceAccountCredentials
gc = gspread.service_account(filename=serviceaccount)

def read_url(url, SHEET_NAME):
    SHEET_ID = url.split("/")[5]
    spreadsheet = gc.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    rows = worksheet.get_all_records()
    df_spread = pd.DataFrame(rows)
    return df_spread, worksheet

def get_dist(city, df, dist_thred):
    return df[df['dist_hav'] < dist_thred].reset_index(drop = True)

def load_data(city):
    cityabbr = city.replace(" ", "").lower()
    META_FILE = "{cityabbr}_meta.csv"

    metafiles = os.listdir(os.path.join(ROOTFOLDER, cityabbr, meta_folder))
    meta_file = META_FILE.format(cityabbr=cityabbr)
    meta_file_path = os.path.join(ROOTFOLDER, cityabbr, meta_folder, meta_file)
    df = pd.read_csv(meta_file_path)
    df = df[df['lat'].notnull()].reset_index(drop = True)
    
    return df, meta_file_path

def sel_data(city):
    print("Now Processing: ", city)
    logger.info("Now Processing: "+city)
    cityabbr = city.replace(" ", "").lower()
    df, meta_file_path = load_data(city)
    output_file_path = os.path.join(transformed_folder, f'{cityabbr}.csv')
#     if os.path.exists(output_file_path):
#         return print("city ", city, "done")
    
#     else:
    df_sel = get_dist(city, df, dist_thred)

    # create h3 index
    print("total rows before dropping small images: ", df.shape[0])
    df = df[df['size']>=sthres].reset_index(drop = True) # drop th einvalid data
    print("total rows after dropping the small images: ", df.shape[0])
    if not 'data_group' in df.columns:
        message = "City "+ city+ " has invalid data"
        logger.error(message)
        return print(message)
    train_test_pool = df[df['data_group'].isin(['test', 'val'])].reset_index(drop = True)
    if train_test_pool.shape[0]>train_n+test_n:
        train_sel = train_test_pool.reset_index(drop = True).sample(n = train_n).reset_index(drop = True)
        remain = train_test_pool[~train_test_pool['path'].isin(train_sel['path'])].reset_index(drop = True)
        test_sel = remain.sample(n = test_n)
    elif df.shape[0]>train_n*2+test_n: # validation set needs to be at least the same as the training for inference use
        logger.warning("Not enough training data. break the rule here.")
        train_sel = df.sample(n = train_n).reset_index(drop = True)
        remain = df[~df['path'].isin(train_sel['path'])].reset_index(drop = True)
        test_sel = remain.sample(n = test_n)
    else:
        logger.warning("Not enough total data. use fraction sample methods")
        train_sel = df.sample(frac = 0.4).reset_index(drop = True)
        remain = df[~df['path'].isin(train_sel['path'])].reset_index(drop = True)
        test_sel = remain.sample(frac = 0.33)

    # relabel the data and save the csv
    df['data_group'] = np.where(df['path'].isin(train_sel['path']), 'train', 'val')
    df['data_group'] = np.where(df['path'].isin(test_sel['path']), 'test', df['data_group'])
    # resave the data into a new folder for train and test
    df['city'] = city
    df['label'] = df['city'].apply(lambda x: city_meta_label[x])
    print(df.groupby("data_group").size())
    df.to_csv(os.path.join(transformed_folder, f'{cityabbr}.csv'), index = False)
    print("data saved.")
    print("*"*100)
    return df[['path', 'label','data_group','city']]

# get all cities
url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
SHEETNAME = "select_city_classifier"
city_meta, other_worksheet = read_url(url, SHEETNAME)
city_meta_label = dict(zip(city_meta['City'].values,city_meta['label'].values))

for city in tqdm(["London", "Johannesburg", "Seoul", "Singapore", 
                 "Taipei"]):
    sel_data(city)


    