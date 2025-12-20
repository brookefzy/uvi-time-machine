"""This script uses yolov10 to differentiate the age and gender of people"""

from ultralytics import YOLO
import cv2
import supervision as sv
import pandas as pd
from multiprocessing import Pool
import os
from tqdm import tqdm
import argparse
import glob


# FILE_PATH = "/scr/u/yuanzf/yolov10/runs/detect/train9"
# FILE_PATH = "/scr/u/yuanzf/yolov10/runs/detect/train12" # 6 classes # version 2
# FILE_PATH = "/scr/u/yuanzf/yolov10/runs/detect/train2" # 2 classes # version 3
VERSION = 2
FILE_PATH = "./runs/detect/train"
ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
YEAR_SEL = [2022,2021, 2020]

CURATED_FOLDER = f"{ROOTFOLDER}/_curated"
PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
GSV_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"
FOLDER_TO_SAVE = "{ROOTFOLDER}/_transformed/utility_box_v{version}/{cityabbr}"

FILE_TO_SAVE = (
    "{ROOTFOLDER}/_transformed/utility_box_v{version}/{cityabbr}/n={part}_objects.parquet"
)
TEMP_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_sidewalk_attr/sel_path"
os.makedirs(TEMP_FOLDER, exist_ok=True)
path_with_sidewalk = "{temp_folder}/{cityabbr}.parquet"

model = YOLO(f"{FILE_PATH}/weights/best.pt")

def get_result(cityabbr, curated_folder, f_suffixes = "*panoptic.csv"):
    outfolder = f"{curated_folder}/{cityabbr}"
    seg_file = glob.glob(os.path.join(outfolder, f_suffixes))
    panoptic_df = []
    for p in seg_file:
        temp = pd.read_csv(p)
        panoptic_df.append(temp)
    panoptic_df = pd.concat(panoptic_df).reset_index(drop = True)
    return panoptic_df



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
        version = VERSION,
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
        
def get_path_with_sidewalk(cityabbr):
    seg_df = get_result(cityabbr, CURATED_FOLDER, f_suffixes = "*seg.csv")
    seg_df_sel = seg_df[(seg_df['labels']==11)&(seg_df['areas']>=50)].reset_index(drop = True)[["img"]]
    gsv_path_df = pd.read_csv(GSV_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr))
    pano_df = pd.read_csv(PANO_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr))
    pano_sel = pano_df[pano_df['year'].isin(YEAR_SEL)].reset_index(drop = True)
    seg_df_sel['panoid'] = seg_df_sel['img'].apply(lambda x: x[:22])
    pano_sel = pano_sel[pano_sel['panoid'].isin(seg_df_sel['panoid'].values)].reset_index(drop = True)
    gsv_path_df['panoid'] = gsv_path_df['path'].apply(lambda x: x.split("/")[-1][:22])
    sel_path = gsv_path_df.merge(pano_sel, on="panoid")
    sel_path[['path']].to_parquet(path_with_sidewalk.format(cityabbr=cityabbr, temp_folder = TEMP_FOLDER))
    return sel_path[['path']]



def parallel_inference(cityabbr):
    # gsv_path = GSV_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr)
    gsv_path = path_with_sidewalk.format(cityabbr=cityabbr, temp_folder = TEMP_FOLDER)
    # check if file exists then read directly, otherwise generate the data
    if os.path.exists(gsv_path):
        pathdf = pd.read_parquet(gsv_path)
    else:
        pathdf = get_path_with_sidewalk(cityabbr)
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
    args.add_argument("--city", type=str, default="Detroit")
    city = args.parse_args().city

    cityabbr = city.lower().replace(" ", "")
    folder_to_save = FOLDER_TO_SAVE.format(version = VERSION, ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr)
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    parallel_inference(cityabbr)


if __name__ == "__main__":
    main()
