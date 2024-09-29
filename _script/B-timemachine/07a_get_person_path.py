import pandas as pd
import pickle
import os
import glob

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
PATH_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"
CURATED_FOLDER = f"{ROOTFOLDER}/_curated"
TRANSFOM_FOLDER = f"{ROOTFOLDER}/_transformed"
EXFOLDER_SEG_LONG = os.path.join(CURATED_FOLDER, "c_seg_longitudinal_all")
OBJECT_SOURCE_FOLDER = "{CURATED_FOLDER}/{city_abbr}/*_objects.parquet"
PATH_TRANSFOM_FOLDER = f"{TRANSFOM_FOLDER}/t_human_attr/sel_path"
if not os.path.exists(PATH_TRANSFOM_FOLDER):
    os.makedirs(PATH_TRANSFOM_FOLDER)
PATH_SEL = "{PATH_TRANSFOM_FOLDER}/{city_abbr}.csv"

def get_city_path(city):
    """Get the path of the images with at least one person in the city"""
    cityabbr = city.replace(" ", "").lower()
    df_path = pd.read_csv(PATH_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr))
    objfiles = glob.glob(OBJECT_SOURCE_FOLDER.format(CURATED_FOLDER=CURATED_FOLDER, city_abbr=cityabbr))
    df = pd.concat([pd.read_parquet(f) for f in objfiles])

    df_path['img'] = df_path['path'].apply(lambda x: x.split("/")[-1])
    df_path_sel = df[df['object_name']=='person'].drop_duplicates("img").merge(df_path, on='img')[['path']]

    df_path_sel = df[df['object_name']=='person'].drop_duplicates("img").merge(df_path, on='img')[['path']]

    df_path_sel.to_csv(PATH_SEL.format(PATH_TRANSFOM_FOLDER = PATH_TRANSFOM_FOLDER, city_abbr=cityabbr), 
                    index=False)
    return print(f"Done {city}")

city_ls = pd.read_csv("../city_meta.csv")
city_ls = city_ls["City"].tolist()
for city in city_ls:
    get_city_path(city)