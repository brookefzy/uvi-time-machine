"""Prepare a validation dataset for inference"""
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import numpy as np

TRANSFORMED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier"
YOLOFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"
VALFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir"
if not os.path.exists(VALFOLDER):
    os.makedirs(VALFOLDER)
    print("Validation folder made")

def load_train_test():
    files = glob(YOLOFOLDER+'/*/*/*')
    df = pd.DataFrame({'path':files})
    df['file_type'] = df['path'].apply(lambda x: x.split("/")[-1].split(".")[-1])
    df['name'] = df['path'].apply(lambda x: x.split("/")[-1])
    return df

def load_all_other(df):
    files = glob(TRANSFORMED_FOLDER+"/*.csv")
    for f in files:
        testdf = pd.read_csv(f)
        testdf['name'] = testdf['path'].apply(lambda x: x.split("/")[-1])
        valdf = testdf[~testdf["name"].isin(df["name"])]\
                .drop_duplicates("name")\
                .reset_index(drop = True)
        n_all = testdf.shape[0]
        n_val = valdf.shape[0]
        print("total images: ", n_all, "total images can be used for validation: ", n_val)
        outf_name = os.path.basename(f).replace(".csv", ".parquet")
        valdf.drop(["name"], axis = 1).to_parquet(os.path.join(VALFOLDER, outf_name), index = False)
        print(valdf.shape[0])
        print(outf_name)
        print("*"*100)

def main():
    df = load_train_test()
    print("done load all existing data")
    print("*"*100)
    load_all_other(df)
    print("done processing all files for inference")
    
main()

    
