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
from tqdm import tqdm

import re
import requests
import itertools
from io import BytesIO
    
#python /media/Green01/data_zhangfan/code/tools/GSVdownload/GSVdownload.py ../../GSVdownload/test_coordinates.p -o ./temp -t 5 -m -s 10 -p
#python /media/Green01/data_zhangfan/code/tools/GSVdownload/GSVdownload.py './temp/pano_2019-05-30 08:35:49.073484_100.p' -o ./temp -t 5 -m -s 300 -g

# backup API
# retrieval panoid:
# http://cbk0.google.com/cbk?output=xml&panoid=r0OsGDONcRjfDqvEHZk7pg
# http://maps.google.com/cbk?output=xml&ll=37.765,-122.4

# Generate and download 338(26*13) tiles, then stich them into a panorama image
# Fan 2019-05-29


parser = argparse.ArgumentParser(description='GSV panoid retrival and image download')

parser.add_argument('path', metavar= 'PATH',
                    help='coordinates file (for panoid retrieval mode) or panoid file (for GSV download mode)')
parser.add_argument('-o','--save-path',  default= './',
                    help='image save path (default: ./)')
parser.add_argument('-p', '--pano', dest = 'pano',  action = 'store_true',
                    help='panoid retrieval mode')
parser.add_argument('-g', '--gsv', dest = 'gsv',  action = 'store_true',
                    help='GSV download mode')
parser.add_argument('-m', '--multiFolders', dest = 'multiFolders',  action='store_true',
                    help='if generate multilevel folders for images (only work in GSV download mode)')
parser.add_argument('-t', '--sleep-time', default = 30, type=int,
                    help='waiting time when failed')
parser.add_argument('-s', '--start-from', default = -1, type=int,
                    help='start index of the dataframe, if -1 start from the first False marker ')
parser.add_argument('-c', '--closest', action = 'store_true',
                    help='Raise the flag to only obain the nearest one (works only if latest flag not raised)')
parser.add_argument('-l', '--latest', action = 'store_true',
                    help='Raise the flag to only obain the latest one')

################################## for retrieval panoid ################################## 

def _panoids_url(lat, lon):
    """
    Builds the URL of the script on Google's servers that returns the closest
    panoramas (ids) to a give GPS coordinate.
    """
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)

def _panoids_data(lat, lon):
    """
    Gets the response of the script on Google's servers that returns the
    closest panoramas (ids) to a give GPS coordinate.
    """
    url = _panoids_url(lat, lon)
    return requests.get(url)


def panoids(lat, lon, closest=False, latest=False, disp=False):
    """
    Gets the closest panoramas (ids) to the GPS coordinates.
    If the 'closest' boolean parameter is set to true, only the closest panorama
    will be gotten (at all the available dates)
    """

    resp = _panoids_data(lat, lon)

    # Get all the panorama ids and coordinates
    # I think the nearest panorama should be the first one. And the previous
    # successive ones ought to be in reverse order from bottom to top. The final
    # images don't seem to correspond to a particular year. So if there is one
    # image per year I expect them to be orded like:
    # 2015
    # XXXX
    # XXXX
    # 2012
    # 2013
    # 2014
    pans = re.findall('\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp.text)
    pans = [{
        "panoid": p[0],
        "lat": float(p[1]),
        "lon": float(p[2])} for p in pans] # Convert to floats
    
    # remove the first redundent one caused by the imprefect regex regex
    if len(pans) > 0:
        if pans[0]!=pans[1]:
            print('The first two panoids are not identical! lat:{0}, lon:{1}'.format(lat, lon))
        else:
            pans = pans[1:]
    


    # Get all the dates
    # The dates seem to be at the end of the file. They have a strange format but
    # are in the same order as the panoids except that the latest date is last
    # instead of first. They also appear to have the index of the panorama before
    # them. However, the last date (which corresponds to the first/main panorama
    # doesn't have an index before it. The following regex just picks out all
    # values that looks like dates and the preceeding index.
    dates = re.findall('([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]', resp.text)
    dates = [list(d) for d in dates]

    # Make sure the month value is between 1-12
    dates = [d for d in dates if int(d[2]) <= 12 and int(d[2])>=1]

    # Make the last value of the dates the index
    if len(dates) > 0 and dates[-1][0] == '':
        dates[-1][0] = '0'
    dates = [[int(v) for v in d] for d in dates] # Convert all values to integers

    # Merge the dates into the panorama dictionaries
    for i, year, month in dates:
        pans[i].update({'year': year, "month": month})
    validList = [index for index, year, month in dates]
    pans = [pans[index]  for index in validList]
    
    # Sort the pans array
    def func(x):
        if 'year'in x:
            return datetime.datetime(year=x['year'], month=x['month'], day=1)
        else:
            return datetime.datetime(year=3000, month=1, day=1)
        
    if latest:
        pans.sort(key=func)
        return pans[-1] 
    elif closest:
        return pans[-1] 
    else:
        return pans
    
    if disp:
        for pan in pans:
            print(pan)



################################## for GSV download current ################################## 

def api_download(panoid, heading, flat_dir, fname = '', width=640, height=640, extension='jpg', year='0000'):
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
    if fname == '':
        fname = "%s_%s" % (panoid, str(heading))
    else:
        fname = "%s_%s" % (fname, str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    url = 'https://geo0.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&output=thumbnail&nbt'
    params = {
        # maximum permitted size for free calls
        "w": width,
        "h": height,
        "yaw": heading,
        "panoid": panoid
    }

    response = requests.get(url, params=params, stream=True)
    try:
        img = Image.open(BytesIO(response.content))
        filename = '%s/%s.%s' % (flat_dir, fname, extension)
        img.save(filename, image_format)
    except:
        print("Image not found")
        filename = None
    del response
    return filename


def download_flats(panoid, flat_dir, fname = '', width=400, height=300,
                  extension='jpg', year='0000'):
    for heading in [0, 90, 180, 270]:
        api_download(panoid, heading, flat_dir, fname, width, height, extension, year)
        
        
################################## for GSV download backup ##################################    
def tiles_info(panoid):
    """
    Generate a list of a panorama's tiles and their position.

    The format is (x, y, filename, fileurl)
    """

    image_url = "http://cbk0.google.com/cbk?output=tile&panoid={0:}&zoom=5&x={1:}&y={2:}"

    # The tiles positions
    coord = list(itertools.product(range(26),range(13)))

    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, x, y)) for x,y in coord]

    return tiles


def download_tiles(tiles, directory, disp=False):
    """
    Downloads all the tiles in a Google Stree View panorama into a directory.

    Params:
        tiles - the list of tiles. This is generated by tiles_info(panoid).
        directory - the directory to dump the tiles to.
    """

    for i, (x, y, fname, url) in enumerate(tiles):

        if disp and i % 20 == 0:
            print("Image %d (%d)" % (i, len(tiles)))

        # Try to download the image file
        while True:
            try:
                response = requests.get(url, stream=True)
                break
            except requests.ConnectionError:
                print("Connection error. Trying again in 2 seconds.")
                time.sleep(2)

        with open(directory + '/' + fname, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

def stitch_tiles(panoid, tiles, directory, final_directory):
    """
    Stiches all the tiles of a panorama together. The tiles are located in
    `directory'.
    """
 
    tile_width = 512
    tile_height = 512 
    
    panorama = Image.new('RGB', (26*tile_width, 13*tile_height))

    for x, y, fname, url in tiles:

        fname = directory + "/" + fname
        tile = Image.open(fname)

        panorama.paste(im=tile, box=(x*tile_width, y*tile_height))

        del tile

    panorama.save(final_directory + ("/%s.jpg" % panoid))
    del panorama    



############################# utils ##############################

def make_folders(args, panoid):
    folder = args.save_path 
    if args.multiFolders:            
        flat_dir = os.path.join(folder, panoid[0], panoid[1])
        if not os.path.exists(flat_dir):
            os.makedirs(flat_dir)
    else:            
        flat_dir = folder

    return flat_dir

################### Main loop functions ##########################
        

def main_retrieval_panoids(args):
    pointDF = pd.read_pickle(args.path).reset_index(drop=True)
    total  = len(pointDF)

    if args.start_from == -1:
        args.start_from = 0

    pointDF = pointDF[args.start_from:] 
    print( 'total: ', total, 'start from: ', args.start_from)
    save_path =  os.path.join(args.save_path, 'pano_{}_{}.p'.format(datetime.datetime.now(), total))
    print( 'save path: ', save_path )
                              
    coordsL = zip(pointDF.id.values, zip(pointDF.lat.values, pointDF.lon.values))
    panoList = []

    for _id, coord in coordsL:
        failed_counter = 0
        while True:        
            try:
                # panoids = panoids(lat=-71.98508, lon=43.41396)
                thisDF = pd.DataFrame(panoids(*coord, closest = args.closest, latest = args.latest))
                break
            except requests.ConnectionError:
                print("{} Connection error at {}, sleep for {} sec".format(
                    str(datetime.datetime.now()),_id, args.sleep_time))
                failed_counter += 1
                time.sleep(args.sleep_time)
            if failed_counter>=10:
                print( 'panoid failed so much! Quit!')
                pd.concat(panoList).reset_index(drop=True).to_pickle(save_path) 
                return
                              
        thisDF['id'] = _id
        panoList.append(thisDF)
        if _id%100 == 0:
            print('[{}|{}]'.format(_id, total),  str(datetime.datetime.now()) ) 
        if _id%1000 == 0:
            pd.concat(panoList).reset_index(drop=True).to_pickle(save_path)    

    pd.concat(panoList).reset_index(drop=True).to_pickle(save_path)    

def main_download_GSVs(args):

    wholeDF = pd.read_pickle(args.path)
    if 'marker' not in wholeDF.columns:
        wholeDF['marker'] = False
        
    total = len(wholeDF)        
    print('size of whole DF: ', total)
    
    if args.start_from == -1:
        sampleDF = wholeDF[wholeDF['marker'] == False]        
    else:
        sampleDF = wholeDF.loc[args.start_from:]   
    print( 'start from: ', sampleDF.index.values[0])
    
    for i, panoid in tqdm(zip(sampleDF.index.values, sampleDF.panoid.values) ):
        
        flat_dir = make_folders(args, panoid)
        failed_counter = 0
        
        while True:  
            try:
                download_flats(panoid, flat_dir, panoid, width=400, height=400, extension='jpg')
                wholeDF.loc[wholeDF.panoid == panoid, 'marker'] =True
                break
            except requests.ConnectionError:
                print("{} Connection error at {}, sleep for {} sec".format(
                    str(datetime.datetime.now()), i, args.sleep_time))
                failed_counter += 1
                time.sleep(args.sleep_time)
                 
            if failed_counter>=20:
                print('gsv failed so much! Quit!') 
                wholeDF.to_pickle(args.path)
                return

        if i%100 == 0:
            print( '[{}|{}]'.format(i, total),  str(datetime.datetime.now()) )
            wholeDF.to_pickle(args.path)
    wholeDF.to_pickle(args.path)
                              
def main_retrieval_download_GSVs(args):
#     pointDF = pd.read_pickle(args.path).reset_index(drop=True)
#     if 'marker' not in pointDF.columns:
#         pointDF['marker'] = False
        
#     total  = len(pointDF)
#     pointDF = pointDF[args.start_from:] 
#     print 'total: ', total, 'start from: ', args.start_from
                    
#     coordsL = zip(pointDF.id.values, zip(pointDF.lat.values, pointDF.lon.values))
#     panoList = []

#     for _id, coord in coordsL:
#         failed_counter = 0
#         while True:        
#             try:
                 
#                 thisDF = pd.DataFrame(panoids(*coord,closest=True))
#                 break
#             except requests.ConnectionError:
#                 print("{} Connection error at {}, sleep for {} sec".format(
#                     str(datetime.datetime.now()),_id, args.sleep_time))
#                 failed_counter += 1
#                 time.sleep(args.sleep_time)
#             if failed_counter>=10:
#                 print 'panoid failed so much! Quit!'
#                 pd.concat(panoList).reset_index(drop=True).to_pickle(save_path) 
#                 return
            
#         while True:  
#             try:
#                 download_flats(panoid, flat_dir, panoid, width=400, height=400, extension='jpg')
#                 imageDF.loc[imageDF.panoid == panoid, 'marker'] =True
#                 break
#             except requests.ConnectionError:
#                 print("{} Connection error at {}, sleep for {} sec".format(
#                     str(datetime.datetime.now()), i, args.sleep_time))
#                 failed_counter += 1
#                 time.sleep(args.sleep_time)
                 
#             if failed_counter>=10:
#                 print 'GSV failed so much! Quit!'
#                 imageDF.to_pickle(args.path)
#                 return         
            
#         flat_dir = make_folders(args, panoid)
        
        
#         thisDF['id'] = _id
#         panoList.append(thisDF)
#         if _id%100 == 0:
#             print('[{}|{}]'.format(_id, total),  str(datetime.datetime.now()) ) 
#         if _id%1000 == 0:
#             pd.concat(panoList).reset_index(drop=True).to_pickle(save_path)    

#     pd.concat(panoList).reset_index(drop=True).to_pickle(save_path)  
    pass


def main():
    
    args = parser.parse_args()
    
    print( 'download from:', args.path )
    print('Sleep time:', args.sleep_time)
    print( 'MultiLevel Folder', args.multiFolders)
    print( 'Retrieval panoid:', args.pano )
    print( 'Download gsv:', args.gsv)
    
    if args.pano is True and args.gsv is False:
        print( 'Retrieval panoid only')
        main_retrieval_panoids(args)
        

    if args.pano is False and args.gsv is True:
        print( 'Download GSV only'         )
        main_download_GSVs(args)
             
    if args.pano is True and args.gsv is True:
        print( 'Retrieval and Download GSV'         )
        main_retrieval_download_GSVs(args)
                              
if __name__ == '__main__':
    main()
    
 
     

