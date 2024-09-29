import pandas as pd
import pickle
import os
import glob
from tqdm import tqdm

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


COLS_NAME = ['gender',
              'age', 
              'side',
              'glasses', 
              'hat', 
              'obs_front',
              'bags',
              'upper',
              'lower',
              'shoes',
              'idx',
            'confidence',
              'x1','y1','x2','y2']


def process_one_img(test_file):
  with open(test_file, 'rb') as f:
      data = pickle.load(f)
  try: 
      df = pd.concat([pd.DataFrame(data[0]['human_attr']), pd.DataFrame(data[0]['boxes'])], axis=1)
      df.columns = COLS_NAME
      df['img'] = data[0]['img']
      return df
  except:
      return test_file


def get_one_city(city):
    full_result = []
    error_files = []
    cityabbr = city.replace(" ", "").lower()
    results_folder = f"/lustre1/g/geog_pyloo/05_timemachine/_curated/c_human_attr/{cityabbr}"
    results_files = glob.glob(f"{results_folder}/human_attr_*.pickle")
    print("Number of files: ", len(results_files))

    for test_file in tqdm(results_files):
        df = process_one_img(test_file)
        if type(df) == str:
            error_files.append(df)
            if len(error_files) % 100 == 0:
                print(f"Error files: {len(error_files)}")
        else:
            full_result.append(df)
    full_result = pd.concat(full_result).reset_index(drop=True)
    full_result.to_csv(f"{results_folder}/human_attr_all.csv", index=False)


city_ls = pd.read_csv("../city_meta.csv")
for city in city_ls["City"].tolist():
    get_one_city(city)
    print(f"Done {city}")