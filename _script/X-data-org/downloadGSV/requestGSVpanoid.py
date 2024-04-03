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
dirsave = 'Z:\\GSV\\_EU\\gsv_rgb\\munich\\gsvmeta'
pointDF=pd.read_pickle(os.path.join(dirsave,'sentPt.p'))[150000:]
coordsL = zip(pointDF['id'].values, zip(pointDF.lat.values, pointDF.lon.values))

imageDF = pd.DataFrame()
for _id, coord in tqdm(coordsL):
    try:
        thisDF = pd.DataFrame(GSVdownload.panoids(*coord,closest=False))#####
        thisDF['id'] = _id
        imageDF = imageDF.append(thisDF)
    except:
        break

imageDF=imageDF.drop_duplicates(subset='panoid')
imageDF.to_pickle(os.path.join(dirsave,'gsv_pano_2.p'))
imageDF.to_csv(os.path.join(dirsave,'gsv_pano_2.csv'), index = False)

