import os
import glob
import pandas as pd
from PIL import Image
YOLOFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"
files = glob.glob(YOLOFOLDER+'/*/*/*')

df = pd.DataFrame({'path':files})
df['file_type'] = df['path'].apply(lambda x: x.split("/")[-1].split(".")[-1])
df['folder'] = df['path'].apply(lambda x: x.split("/")[-2])
df['group'] = df['path'].apply(lambda x: x.split("/")[-3])
df['name'] = df['path'].apply(lambda x: x.split("/")[-1])
# read each file and calculate the image size and check for corrupted images
for path in df['path'].values:
    try:
        img = Image.open(path)
        img.size
    except:
        print(f"corrupted image found at {path}")
        os.remove(path)
        print(f"{path} is removed")