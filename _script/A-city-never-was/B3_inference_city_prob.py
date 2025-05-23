"""This script try inference by city names."""

from ultralytics import YOLO
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import numpy as np
import gc
import datetime
import argparse


VALFOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8_inf_dir"
)
CURATED_FOLDER = (
    "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob"
)
if not os.path.exists(CURATED_FOLDER):
    os.makedirs(CURATED_FOLDER)

# load list of files to infer
ALL_TO_INFER = os.listdir(VALFOLDER)
# batch processing in case out of memory
C_SIZE = 500

# START LOAD INFERENCE PARAMETER
model = YOLO(
    "/home/yuanzf/uvi-time-machine/_script/A-city-never-was/runs/classify/train4/weights/best.pt"
)


def inference_prob(paths, model=model, k=1):
    paths = list(paths)
    results = model.predict(paths, verbose=False)
    pred_ls = []
    for i, r in enumerate(results):
        top = r.cpu().probs.data.numpy()
        pred_ls.append(top)
    pred_df = pd.DataFrame(pred_ls)
    pred_df["name"] = [x.split("/")[-1] for x in paths]
    return pred_df



def get_finished(f, curated_city_folder):
    city = f.replace(".parquet", "")
    allfinished = glob(curated_city_folder + "/" + city + "_*.parquet")
    if len(allfinished) > 0:
        p_df = []
        for p in allfinished:
            tmp = pd.read_parquet(p)
            p_df.append(tmp)
        p_df = pd.concat(p_df).reset_index(drop=True)
        return p_df["name"].values
    else:
        return []


def get_result_one(city):
    f = city + ".parquet"
    CURATED_CITY_FOLDER = os.path.join(CURATED_FOLDER, city)
    if not os.path.exists(CURATED_CITY_FOLDER):
        os.makedirs(CURATED_CITY_FOLDER)

    # Load to inference
    df = pd.read_parquet(os.path.join(VALFOLDER, f))
    df["name"] = df["path"].apply(lambda x: x.split("/")[-1])
    print("Total to infer", df.shape[0])

    name_f = get_finished(f, CURATED_CITY_FOLDER)

    df = df[~df["name"].isin(name_f)].reset_index(drop=True)
    print("Remain: ", df.shape[0])
    if df.shape[0] > 0:

        # use chunk size to do batch processing
        task_ls = []
        for i in range(0, df.shape[0], C_SIZE):
            task_ls.append(df.iloc[i : i + C_SIZE]["path"])

        for j, task in enumerate(tqdm(task_ls)):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pred_df = inference_prob(task)
            output_name = f.replace(".parquet", f"_{current_time}.parquet")
            pred_df.to_parquet(
                os.path.join(CURATED_CITY_FOLDER, output_name), index=False
            )
            gc.collect()
        print("Done: ", f)
        print("*" * 100)
    else:
        print("Done: ", f)
        print("*" * 100)


def main():

    parser = argparse.ArgumentParser(description="Inference by city names.")
    parser.add_argument(
        "--city",
        type=str,
        default="hongkong",
        help="city name in lower case and no space",
    )
    args = parser.parse_args()
    city = args.city
    get_result_one(city)


if __name__ == "__main__":
    main()
