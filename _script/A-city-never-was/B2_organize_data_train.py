import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import logging
logging.basicConfig(filename='label_data.log', format='%(asctime)s %(message)s', filemode='w') 
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



for folder in ['train', 'test']:
    print(f"working on the {folder} data")
    logger.info(f"working on the {folder} data")
    data = dataset[folder]
    finished = glob.glob(os.path.join(YOLOFOLDER, folder, "*/*.jpg"))
    finished_names = [x.split("/")[-1] for x in finished]
    print("Total finished images: ", len(finished_names))
    for city in tqdm(train_df['city'].unique()):
        city_folder = os.path.join(YOLOFOLDER, folder, city)
        if not os.path.exists(city_folder):
            os.makedirs(city_folder)
            print(f"start to copy data for city {city}")
            logger.info(f"start to copy data for city {city}")
            temp = data[data['city']==city].reset_index(drop = True)
            cap = DATACAP[folder]
            if temp.shape[0]>cap:
                temp = temp[:cap].reset_index(drop = True)
            for path in tqdm(temp['path'].values):
                img_name = path.split("/")[-1]
                if img_name in finished_names:
                    continue
                else:
                    shutil.copy(path, city_folder)
            print(f"{city} is done")
            logger.info(f"{city} is done")
            print("*"*100)
    logger.info(f"{folder} is done")
    logger.info("*"*100)