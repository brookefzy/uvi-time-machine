#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu August 23 16:16:00 2023

@author: yuanfan

This script will run after you already have panoid retrieved from google map.
You will need to load your panoid file (imgGDF) to run this code.
"""
import gc
import sys
import datetime
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import time
import os
import copy
import random
import shutil
import sys
import argparse
from PIL import Image

import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib
import glob

import re
import requests
import itertools
from io import BytesIO
from shapely.geometry import Point, Polygon
from fiona.crs import from_epsg
import GSVdownload
# import descartes
import hashlib
from tqdm import tqdm
import time


def api_download(panoid, heading, imgfolder, width=640, height=640, extension='jpg', year='0000'):
    """
    Download an image using the official API. These are not panoramas.

    Params:
        :panoid: the panorama id
        :heading: the heading of the photo. Each photo is taken with a 360
            camera. You need to specify a direction in degrees as the photo
            will only cover a partial region of the panorama. The recommended
            headings to use are 0, 90, 180, or 270.
        :flat_dir: the direction to save the image to.
        :key: your API key.
        :width: downloaded image width (max 640 for non-premium downloads).
        :height: downloaded image height (max 640 for non-premium downloads).
        :fov: image field-of-view.
        :image_format: desired image format.

    You can find instructions to obtain an API key here: https://developers.google.com/maps/documentation/streetview/
    """
    # if fname == '':
    #     fname = "%s_%s" % (panoid, str(heading))
    # else:
    fname = "%s_%s" % (panoid, str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    # REVISED on 2023-08-23
    # img_url = 'https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={panoid:}&cb_client=search.revgeo_and_fetch.gps&w={width:}&h={height:}&yaw={heading:}&pitch=0&thumbfov=100'
    img_url = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={panoid:}&cb_client=search.revgeo_and_fetch.gps&yaw={heading:}&pitch=0&thumbfov=100&w={width:}&h={height:}&quot"
    url = img_url.format(panoid=panoid, width=width, height=height, heading=heading)
    response = requests.get(url, stream=True)
    try:
        img = Image.open(BytesIO(response.content))
        md5name=hashlib.md5(panoid.encode()).hexdigest()
        try:
            flat_dir = os.path.join(imgfolder, md5name[-1]+"_1", md5name[-2], md5name[-3])
            if not os.path.exists(flat_dir):
                os.makedirs(flat_dir)
            filename = '%s/%s.%s' % (flat_dir, fname, extension)
            img.save(filename, image_format)
        except:
            print("Recreate a folder")
            flat_dir = os.path.join(imgfolder, md5name[-1]+"_2", md5name[-2], md5name[-3])
            if not os.path.exists(flat_dir):
                os.makedirs(flat_dir)
            filename = '%s/%s.%s' % (flat_dir, fname, extension)
            img.save(filename, image_format)
    except:
        print("Image not found")
        filename = None
    del response
    return filename


def get_path(subfolder):
    files = glob.glob(os.path.join(subfolder, "*/*/*.jpg"))
    return files
    
def load_data(args):

    city = args.city
    cityabbr = city.replace(" ", "").lower()
    generalgsv = '/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{cityabbr}'.format(cityabbr=cityabbr)
    # load the panoid file
    dirsave = os.path.join(generalgsv, 'gsvmeta')
    imgGDF = pd.read_csv(os.path.join(dirsave,'gsv_pano.csv'))
    print(imgGDF.shape[0])

    imgfolder = os.path.join(generalgsv, 'img_rgb')
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)
        
    
    def load_finshed(imgfolder):
        subfolders = os.listdir(imgfolder)
        subfolders = [os.path.join(imgfolder, sub) for sub in subfolders]
        pool = Pool(8)
        allfiles = []
        for ret in tqdm(pool.imap(get_path, subfolders), total = len(subfolders)):
            allfiles.append(ret)
        allfiles = [item for sublist in allfiles for item in sublist]
        panodf = pd.DataFrame(allfiles, columns = ["path"])
        if panodf.shape[0] == 0:
            print("No images available.")
            return []
        else:
            panodf['panoid'] = panodf['path'].apply(lambda x: x.split("/")[-1].split(".")[0][:22])
        return panodf['panoid'].values
        
    # finished = glob.glob(imgfolder+"/*/*/*/*.jpg")
    # finished_pano = [i.split('/')[-1][:22] for i in finished]
    finished_pano = load_finshed(imgfolder)
    imgGDF = imgGDF[~imgGDF.panoid.isin(finished_pano)].reset_index(drop = True)
    return imgGDF, imgfolder
        
# use multi-threading to download images
def download_img(task):
    panoid, imgfolder = task
    for heading in [0, 90, 180, 270]:
        try:
            api_download(panoid, heading, imgfolder, width=400, height=400,extension='jpg', year="0000")
        # print exception message
        except Exception as e:
            print(e)
            time.sleep(10)
            print("Sleeping for 10 seconds")

            # print error message
            
    time.sleep(random.randint(1, 150)/100)
    
    
def main():
    parser = argparse.ArgumentParser(description='GSV image download')
    parser.add_argument('--city', type=str, default='Hong Kong', help='city name')
    args = parser.parse_args()
    

    pool = Pool(10)
    imgGDF, imgfolder = load_data(args)
    taskls = [(panoid, imgfolder) for panoid in imgGDF.panoid.values]

    for ret in tqdm(pool.imap(download_img, taskls), total=len(taskls)):
        pass


if __name__ == '__main__':
    main()

