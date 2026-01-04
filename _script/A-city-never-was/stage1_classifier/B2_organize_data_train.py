import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import logging
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logging_file = f"label_data_{now}.log"
logging.basicConfig(filename=logging_file, format='%(asctime)s %(message)s', filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG)

TRANSFORMED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier"
CURATED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_classifier"
if not os.path.exists(CURATED_FOLDER):
    os.makedirs(CURATED_FOLDER)
    
YOLOFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"
TRAIN_FOLDER = os.path.join(YOLOFOLDER, "train")
TEST_FOLDER = os.path.join(YOLOFOLDER, "test")
for fl in [YOLOFOLDER, TRAIN_FOLDER, TEST_FOLDER]:
    if not os.path.exists(fl):
        os.makedirs(fl)
        
        
train_file_name = "c_train.parquet"
test_file_name = "c_test.parquet"
train_df = pd.read_parquet(os.path.join(CURATED_FOLDER, train_file_name))
test_df = pd.read_parquet(os.path.join(CURATED_FOLDER, test_file_name))



        
dataset = {
    "train":train_df,
    "test":test_df}
DATACAP = {
    "train":10000,
    "test":1000
}



for folder in ['train']:
    print(f"working on the {folder} data")
    logger.info(f"working on the {folder} data")
    data = dataset[folder]
    cap = DATACAP[folder]
    
    finished = glob.glob(os.path.join(YOLOFOLDER, folder, "*/*.jpg"))
    finished_names = [x.split("/")[-1] for x in finished]
    finished_df = pd.DataFrame({'path':finished,
                               'name':finished_names})
    finished_df['city'] = finished_df['path'].apply(lambda x: x.split("/")[-2])
    data['name'] = data['path'].apply(lambda x: x.split("/")[-1])
    print("Available to move data ORIGINAL:", data.shape[0])
    data = data.drop_duplicates('name').reset_index(drop = True)
    print("Available to move data AFTER DROPPING Duplicates:", data.shape[0])
    data = data[~data['name'].isin(finished_names)].reset_index(drop = True)
    print("Total finished images: ", len(finished_names))
    print("Available to move data AFTER DROPPING FINISHED:", data.shape[0])
    print(data['city'].unique())
    
    for city in tqdm(train_df['city'].unique()):
        finished_names_city = finished_df[finished_df['city']==city].reset_index(drop = True)
        print("Already moved: ", finished_names_city.shape[0])
        if finished_names_city.shape[0]>=cap:
            continue
            print(f"{city} is done")
            logger.info(f"{city} is done")
        else:
            city_folder = os.path.join(YOLOFOLDER, folder, city)
            print(city_folder)
            if not os.path.exists(city_folder):
                os.makedirs(city_folder)
            print(f"start to copy data for city {city}")
            logger.info(f"start to copy data for city {city}")
            temp = data[data['city']==city].reset_index(drop = True)
            print(f"Available Images for {city}",temp.shape[0])
            remain_to_move = cap-finished_names_city.shape[0]
            print("remain to move:", remain_to_move)
            if temp.shape[0]>=remain_to_move:
                temp = temp[:remain_to_move].reset_index(drop = True)
            for path in tqdm(temp['path'].values):
                shutil.copy(path, city_folder)
        print(f"{city} is done")
        logger.info(f"{city} is done")
        print("*"*100)
    logger.info(f"{folder} is done")
    logger.info("*"*100)