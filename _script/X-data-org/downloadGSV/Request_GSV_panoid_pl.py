
import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import copy
import random
import shutil
import argparse
import re
import requests
import GSVdownload
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time


parser = argparse.ArgumentParser(description='GSV panoid retrival and image download')
parser.add_argument('--city', type=str, default='Hong Kong', help='city name')
args = parser.parse_args()


def func_panoids(task):
    lat = task['lat']
    lon = task['lon']
    try:
        thisDF = pd.DataFrame(GSVdownload.panoids(lat, lon, closest=False))
        thisDF['id'] = task['id']
        return thisDF
    except:
        return pd.DataFrame()

def export_panoid(args):
    city = args.city
    cityabbr = city.replace(" ", "").lower()
    print("Now Processing city: ", cityabbr)
    dirsave = f'/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{cityabbr}/gsvmeta'
    pointDF = pd.read_pickle(os.path.join(dirsave,'sentPt.p'))
    
    # sample to 80% of the points
    if pointDF.shape[0]<200000:
        frac = 1
    else:
        frac = 0.8
    pointDF = pointDF.sample(frac=frac, random_state=1).reset_index(drop=True)
    if city == 'springfield':
        pointDF = pointDF[230999:]
    print("Total number of pt sent: ", len(pointDF))
    # Load finished
    try:
        finished = pd.read_csv(os.path.join(dirsave,'gsv_pano.csv'))
        pointDF = pointDF[~pointDF['id'].isin(finished['id'])].reset_index(drop=True)
    except:
        print("No finished file")
    
    print("Total number of pt remain to be sent: ", len(pointDF))
    pool = Pool(8)
    
    taskls = [{'lat':pointDF.lat.values[i], 'lon':pointDF.lon.values[i], 'id':pointDF['id'].values[i]} for i in range(len(pointDF))]
    print("Total number of pt sent: ", len(taskls))
    imageDF = []
    for res in tqdm(pool.imap(func_panoids, taskls), total=len(taskls)):
        imageDF.append(res)
        time.sleep(random.randint(1, 5)/100)
        if len(imageDF)%1000 == 0:
            # save to file
            imageDF = pd.concat(imageDF).reset_index(drop = True)
            imageDF = imageDF.drop_duplicates(subset='panoid')
            print("Total number of pt received:", len(imageDF))
            imageDF.to_csv(os.path.join(dirsave,'gsv_pano.csv'), mode='a', header=False, 
                index = False)
            print("Saved to file")
            imageDF = []
        else:
            pass
    # save the last batch to file
    imageDF = pd.concat(imageDF).reset_index(drop = True)
    imageDF = imageDF.drop_duplicates(subset='panoid')
    print("Total number of pt received:", len(imageDF))
    # imageDF.to_pickle(os.path.join(dirsave,'gsv_pano.p'))
    imageDF.to_csv(os.path.join(dirsave,'gsv_pano.csv'), 
                   mode='a', 
                   header=False,
                index = False)

def main():
    export_panoid(args)

if __name__ == '__main__':
    main()

# export_panoid("London")
