import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
TRANSFORMED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier"
CURATED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_classifier"
if not os.path.exists(CURATED_FOLDER):
    os.makedirs(CURATED_FOLDER)
    
YOLOFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"
TRAIN_FOLDER = os.path.join(YOLOFOLDER, "train")
TEST_FOLDER = os.path.join(YOLOFOLDER, "test")

train_file_name = "c_train.parquet"
test_file_name = "c_test.parquet"

files = glob.glob(TRANSFORMED_FOLDER+"/*.csv")
train_df= []
test_df = []
for f in tqdm(files):
    temp = pd.read_csv(f)
    temp['data_group'] = temp['data_group'].apply(lambda x: x.replace('sel','test'))
    temp['name'] = temp['path'].apply(lambda x: x.split("/")[-1])
    temp = temp.drop_duplicates("name").reset_index(drop = True)
    train_temp = temp[temp['data_group']=='train'].reset_index(drop = True)
    train_count = train_temp.shape[0]
    if train_count<20000:
        delta = 20000-train_count
        # sample additional training from the validation set
        val_temp = temp[temp['data_group']=='val'].reset_index(drop = True)
        try:
            val_to_train = val_temp.sample(n = delta, random_state = 1)
            temp.loc[val_to_train.index, 'data_group'] = 'train'
            temp.drop("name", axis =1).to_csv(f, index = False)
        except:
            temp.drop("name", axis =1).to_csv(f, index = False)
        
    train_df.append(temp[temp['data_group']=='train'])
    test_df.append(temp[temp['data_group']=='test'])

train_df = pd.concat(train_df).reset_index(drop = True)
test_df = pd.concat(test_df).reset_index(drop = True)
train_df[['path', 'data_group','city','label']].to_parquet(os.path.join(CURATED_FOLDER, train_file_name), index = False)
test_df[['path', 'data_group','city','label']].to_parquet(os.path.join(CURATED_FOLDER, test_file_name), index = False)