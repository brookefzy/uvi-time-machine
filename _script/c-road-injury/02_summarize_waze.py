
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import gspread
import gc
import json

serviceaccount = "/home/yuanzf/google_drive_personal.json"
gs = gspread.service_account(filename=serviceaccount)

ROOTFOLDER = "/group/geog_pyloo/08_GSV/data/_raw/waze"
EXPORTFOLDER = "/group/geog_pyloo/08_GSV/data/_curated/waze"
if not os.path.exists(EXPORTFOLDER):
    os.makedirs(EXPORTFOLDER)
files = os.listdir(ROOTFOLDER)

cols_keep = ['country', 'inscale', 'city', 'reportRating',
       'reportByMunicipalityUser', 'confidence', 'reliability', 'type', 'uuid',
       'speed', 'reportMood', 'roadType', 'magvar', 'subtype', 'street',
       'additionalInfo', 
       'id', 
       'date', 'lat', 'lon']

def read_url(url, SHEET_NAME):
    SHEET_ID = url.split("/")[5]
    spreadsheet = gs.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    rows = worksheet.get_all_records()
    df_spread = pd.DataFrame(rows)
    return df_spread, worksheet

def get_city_meta():
    url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
    SHEETNAME = "select_city_classifier"
    city_meta, other_worksheet = read_url(url, SHEETNAME)
    city_meta = city_meta[city_meta['City']!=''].reset_index(drop = True)
    city_meta = city_meta[city_meta['GSV Downloaded']>0].reset_index(drop = True)
    city_meta['city_lower'] = city_meta['City'].apply(lambda x: x.replace(" ", "").lower())
    return city_meta, other_worksheet

def clean_subfolder(subfolder):
    
    files = glob.glob("{subfolder}/**/*.json".format(subfolder = subfolder), recursive = True)
    if len(files) == 0:
        print("No files found in {subfolder}".format(subfolder = subfolder))
        return None
    alertdf_all = []
    for f in tqdm(files):
        try:
            content = json.load(open(f))
            if "alerts" not in content:
                continue
            alertdf = pd.DataFrame(content['alerts'])
            alertdf_all.append(alertdf)
        except:
            continue
        alertdf = pd.DataFrame(content['alerts'])
        alertdf_all.append(alertdf)
    if len(alertdf_all) == 0:
        return None
    alertdf_all = pd.concat(alertdf_all).reset_index(drop = True)
    return alertdf_all

def get_one_city(sample_city):
    sample_city_lower = sample_city.replace(" ", "").lower()
    # check whehter the city is finished
    if os.path.exists(os.path.join(EXPORTFOLDER,
                                      "{city_lower}_alerts.csv".format(city_lower = sample_city_lower))):
          print("City {sample_city} already processed.".format(sample_city = sample_city))
          return None
    
    subfolder = "{ROOTFOLDER}/city={city_lower}".format(ROOTFOLDER = ROOTFOLDER, city_lower = sample_city_lower)
    alertdf_all_r1 = clean_subfolder(subfolder)
    gc.collect()
    alertdf_all_other = []
    for i in range(6):
        subfolder = "{ROOTFOLDER}_r{i}/city={city_lower}".format(ROOTFOLDER = ROOTFOLDER, 
                                                                city_lower = sample_city_lower,
                                                                i = i)
        alertdf_all = clean_subfolder(subfolder)
        if alertdf_all is None:
            continue
        alertdf_all_other.append(alertdf_all)
    if len(alertdf_all_other) != 0:
        alertdf_all_other = pd.concat(alertdf_all_other).reset_index(drop = True)
    try:
        alertdf_all_combined = pd.concat([alertdf_all_r1, alertdf_all_other]).reset_index(drop = True)
    except:
        print("No alerts found for city: {sample_city}".format(sample_city = sample_city))
        return None
    del alertdf_all_r1, alertdf_all_other, alertdf_all
    gc.collect()
    
    totol_reported = alertdf_all_combined['uuid'].nunique()
    print("Total reported alerts: {totol_reported}".format(totol_reported = totol_reported))

    alertdf_all_combined = alertdf_all_combined.drop_duplicates(subset = ['uuid']).reset_index(drop = True)
    alertdf_all_combined['pubMillis'] = pd.to_datetime(alertdf_all_combined['pubMillis'], unit='ms')
    alertdf_all_combined['date'] = alertdf_all_combined['pubMillis'].dt.date
    # check time range
    print("Time range: {start} - {end}".format(start = alertdf_all_combined['pubMillis'].min(),
                                                end = alertdf_all_combined['pubMillis'].max()))
    
    # find coordinates of all alerts
    alertdf_all_combined['lat'] = alertdf_all_combined['location'].apply(lambda x: x['y'])
    alertdf_all_combined['lon'] = alertdf_all_combined['location'].apply(lambda x: x['x'])
    cols_export = [x for x in cols_keep if x in alertdf_all_combined.columns]
    alertdf_all_combined[cols_export].to_csv(os.path.join(EXPORTFOLDER, 
                                                    "{city_lower}_alerts.csv".format(city_lower = sample_city_lower)), 
                                           index = False)
    print("Exported alerts done for city: {sample_city}".format(sample_city = sample_city))
    return None

city_meta, _ = get_city_meta()
for sample_city in city_meta['City']:
    print("Start processing city: {sample_city}".format(sample_city = sample_city))
    get_one_city(sample_city)
    gc.collect()