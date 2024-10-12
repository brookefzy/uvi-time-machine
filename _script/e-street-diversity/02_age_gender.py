
"""This script uses yolov10 to differentiate the age and gender of people"""

from ultralytics import YOLOv10
import cv2
import supervision as sv
import pandas as pd
from multiprocessing import Pool
import os
from tqdm import tqdm
import argparse


FILE_PATH = "/scr/u/yuanzf/yolov10/runs/detect/train9"
ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
GSV_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"
FOLDER_TO_SAVE = "{ROOTFOLDER}/_transformed/age_gender/{cityabbr}"
FILE_TO_SAVE = "{ROOTFOLDER}/_transformed/age_gender/{cityabbr}/n={part}_objects.parquet"

model = YOLOv10(f'{FILE_PATH}/weights/best.pt')


def load_paths(cityabbr):

    gsv_path = GSV_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr)
    return pd.read_csv(gsv_path)[["path"]]


def inference(img_path):
    """Read image path and generate the results by:
    object_id, object_name, confidence, x1, y1, x2, y2
    """
    # make sure the image exists
    if not os.path.exists(img_path):
        return pd.DataFrame()
    img = cv2.imread(img_path)
    results = model(img)[0]
    # get the results
    detections = sv.Detections.from_ultralytics(results)
    # get the results in a dataframe
    resultdf = pd.concat(
        [
            pd.DataFrame(
                {
                    # "object_id": detections.ids,
                    "object_name": detections.data["class_name"],
                    "confidence": detections.confidence,
                }
            ),
            pd.DataFrame(detections.xyxy, columns=["x1", "y1", "x2", "y2"]),
        ],
        axis=1,
    )
    resultdf["img"] = img_path.split("/")[-1]
    return resultdf


def loop_inference(variables):
    img_paths = variables["img_paths"]
    # check if the paths all finished
    k = variables["i"]
    
    
    file_to_save = FILE_TO_SAVE.format(
        ROOTFOLDER=ROOTFOLDER, cityabbr=variables["cityabbr"], part=k
    )
    if os.path.exists(file_to_save):
        print(file_to_save, "already exists")
        return None
    else:
        resultdf = []
        for path in tqdm(img_paths):
            resultdf.append(inference(path))
        resultdf = pd.concat(resultdf).reset_index(drop=True)
        resultdf.to_parquet(file_to_save, index=False)
        print(
            "saved",
            file_to_save,
        )
        print("*" * 100)


def parallel_inference(cityabbr):
    gsv_path = GSV_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr)
    pathdf = pd.read_csv(gsv_path)
    img_paths = pathdf["path"].tolist()
    print("Total images to inference:", len(img_paths))
    # split the list into 100 parts and save seperately

    NIMAGE = 10_000  # each part save 100_000 images' results
    variable_ls = [
        {
            "img_paths": img_paths[i : i + NIMAGE],
            "i": i,
            "cityabbr": cityabbr,
        }
        for i in range(0, len(img_paths), NIMAGE)
    ]
    pool = Pool(8)
    for res in pool.imap(loop_inference, variable_ls):
        pass


def load_all():

    city_meta = pd.read_csv("../city_meta.csv")
    return city_meta


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--city", type=str, default="Hong Kong")
    city = args.parse_args().city
    
    cityabbr = city.lower().replace(" ", "")
    folder_to_save = FOLDER_TO_SAVE.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr)
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    parallel_inference(cityabbr)


if __name__ == "__main__":
    main()