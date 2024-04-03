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

import matplotlib

import re
import requests
import itertools
from io import BytesIO
from shapely.geometry import Point, Polygon
from fiona.crs import from_epsg
import GSVdownload
from tqdm import tqdm

#use GSVdownload.py here
# city = 'Hong Kong'
# cityabbr = city.replace(" ", "").lower()
dirsave = f'/group/geog_pyloo/08_GSV/data/gsv_rgb/virginia_random/gsvmeta'
pointDF=pd.read_csv(os.path.join(dirsave,'cai-r-building-code-loc-withlatlon.csv'))
coordsL = zip(pointDF['id'].values, zip(pointDF.lat.values, pointDF.lon.values))

imageDF = pd.DataFrame()
for _id, coord in tqdm(coordsL):
    try:
        thisDF = pd.DataFrame(GSVdownload.panoids(*coord, closest=False))#####
        thisDF['id'] = _id
        imageDF = imageDF.append(thisDF)
    except:
        continue

imageDF=imageDF.drop_duplicates(subset='panoid')
# imageDF.to_pickle(os.path.join(dirsave,'gsv_pano.p'))
imageDF.to_csv(os.path.join(dirsave,'gsv_pano.csv'), 
               index = False)

