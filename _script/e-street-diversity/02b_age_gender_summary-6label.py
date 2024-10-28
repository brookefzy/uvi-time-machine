"""This script summarizes the labeled data for age and gender across all cities:
1. Summarize the distribution of female, with confidence level over 0.5, 0.2
2. Summarize the distribution of age >60 with confidence level over 0.5, 0.2
3. Summarize the hexagon level of the 1 and 2"""
import pandas as pd
import os
from tqdm import tqdm
import glob
import h3
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
FOLDER_TO_SAVE = "{ROOTFOLDER}/_transformed/age_gender_v2/{cityabbr}"
FILE_TO_SAVE = "{ROOTFOLDER}/_transformed/age_gender_v2/{cityabbr}/n={part}_objects.parquet"
PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
PATH_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_path.csv"
FILE_TO_EXPORT = "{ROOTFOLDER}/_curated/c_age_gender_v2/{cityabbr}.parquet"
FOLDER_TO_EXP = f"{ROOTFOLDER}/_curated/c_age_gender_v2"
if not os.path.exists(FOLDER_TO_EXP):
    os.makedirs(FOLDER_TO_EXP)
    
object_age_dict = {
                'female18-60':'18-60', 
                'male18-60':'18-60', 
                'male-60':'60+', 
                'male-18':'18-', 
                'female-60':'60+',
                'female-18':'18-'
                }

def get_hex_basics(city_abbr, res):
    # read all object files and concat them into one df
    df_pano = pd.read_csv(PANO_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=city_abbr))
    df_path = pd.read_csv(PATH_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=city_abbr))
    # only keep the panoid that has a path
    df_pano_inner = df_pano[
        df_pano["panoid"].isin(df_path["panoid"].unique())
    ].reset_index(drop=True)
    # print(df_pano_inner.shape[0], "out of ", df_pano.shape[0], "panoids have path")
    df_pano_inner["hex_id"] = df_pano_inner.apply(
        lambda x: h3.geo_to_h3(x["lat"], x["lon"], res), axis=1
    )

    # keep these hex
    df_all_keep = df_pano_inner[(df_pano_inner["year"] >= 2014)
    ].reset_index(drop=True)
    print(df_all_keep.shape[0], "panoids are kept")
    # assign the year_group; skip for now
    
    # get number of panoid per hex per year
    df_all_keep_hex = (
        df_all_keep.groupby(["hex_id", "year"])["panoid"]
        .nunique()
        .reset_index(name="panoid_n")
    )
    return df_all_keep_hex, df_all_keep


def load_results(cityabbr):
    
    df_path = pd.read_csv(PATH_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr))
    df_path['img'] = df_path['path'].apply(lambda x: x.split("/")[-1])
    
    # process the prediction results
    objfiles = glob.glob(
        FOLDER_TO_SAVE.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr) + "/*.parquet"
    )
    if len(objfiles) == 0:
        print("No object files found for", cityabbr)
        return pd.DataFrame(), df_path
    df = pd.concat([pd.read_parquet(f) for f in objfiles])
    return df,df_path
    
def get_confidence(df, conf):
    df_con = df_con = df[df['confidence']>=conf].reset_index(drop=True)
    df_con['object_name'] = df_con['object_name'].apply(lambda x: x.lower())
    df_con['gender'] = df_con['object_name'].apply(lambda x: "female" if "female" in x.lower() else "male")
    df_con['age'] = df_con['object_name'].apply(lambda x: object_age_dict[x.lower()] if x.lower() in object_age_dict else "unknown")
    # find an image with most male-60
    df_summary = df_con.groupby(['img','gender', 'age']).size().reset_index().pivot(
        index='img', columns = ['gender', 'age'], values=0
    ).fillna(0)
    df_summary.columns = ['_'.join(x) for x in df_summary.columns]
    df_summary['pedestrian_count'] = df_summary.sum(axis=1)
    df_summary = df_summary.reset_index()
    df_summary['panoid'] = df_summary['img'].apply(lambda x: x[:22])
    
    df_summary_pano = df_summary.groupby(['panoid']).sum().reset_index()
    # add a step to calculate the image-level average
    return df_summary, df_summary_pano

def get_result(city):
    cityabbr = city.lower().replace(" ", "")
    # check if the export exist, then skip:
    file_export = FILE_TO_EXPORT.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr)
    # if os.path.exists(file_export):
    #     print(f"city {city} already exists")
        # return None
    df, df_path = load_results(cityabbr)
    if df.shape[0] == 0:
        print(f"city {city} has no data")
        return None
    alldf = []
    for res in [8,9]:
        df_all_keep_hex, df_all_keep = get_hex_basics(cityabbr, res)
        for conf in [0.2, 0.5, 0.7, 0.8]:
            df_summary, df_summary_pano = get_confidence(df, conf)
            df_summary_merged = df_summary_pano.merge(df_all_keep, on='panoid', how='inner')
            male_stand = ['male_18-60', 'male_60+','male_18-']
            female_stand = ['female_18-60', 'female_60+','female_18-']
            
            male_export = [x for x in male_stand if x in df_summary_merged.columns]
            female_export = [x for x in female_stand if x in df_summary_merged.columns]
            
            df_summary_merged['male_total'] = df_summary_merged[male_export].sum(axis=1)
            df_summary_merged['female_total'] = df_summary_merged[female_export].sum(axis=1)
            df_summary_merged['male_ratio'] = df_summary_merged['male_total']/df_summary_merged['pedestrian_count']
            df_summary_merged['female_ratio'] = df_summary_merged['female_total']/df_summary_merged['pedestrian_count']
            
            standard_cols = ['male_18-60', 'male_60+', 'female_60+', 'female_18-60',
                'male_18-', 'female_18-','pedestrian_count']
            cols_to_export = [x for x in standard_cols if x in df_summary_merged.columns]
            
            df_summary_hex = df_summary_merged.groupby(['hex_id', 'year'])[cols_to_export].sum().reset_index()
            df_summary_hex_mean = df_summary_merged.groupby(['hex_id', 'year']).agg({
                'male_ratio':'mean',
                'female_ratio':'mean'
            }).reset_index().rename(columns={'male_ratio':'male_ratio_img_mean',
                                             'female_ratio':'female_ratio_img_mean'} 
                                             ) # this is the average among images with at least one person
            
            df_summary_hex_pano = df_summary_merged.groupby(['hex_id','year']).agg({'panoid':'nunique'}).reset_index()
            df_summary_hex_pano.rename(columns={'panoid':'panoid_n_with_person'}, inplace=True)

            df_summary_final = df_all_keep_hex.merge(
                df_summary_hex, on=['hex_id', 'year'], how='left'
            ).merge(
                df_summary_hex_pano, on=['hex_id','year'], how='left'
            ).fillna(0)\
                .merge(
                df_summary_hex_mean, on=['hex_id', 'year'], how='left'
                )
            df_summary_final['res'] = res
            df_summary_final['conf'] = conf
            alldf.append(df_summary_final)
    alldf = pd.concat(alldf).reset_index(drop=True)
    alldf.to_parquet(FILE_TO_EXPORT.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr), index=False)
    print(f"city {city} done")
    return alldf

allcity = pd.read_csv("/home/yuanzf/uvi-time-machine/_script/city_meta.csv")
allcity_ls = allcity['City'].values
# allcity_ls = [x for x in allcity['City'].values if x not in ["New York","Gaborone", "Curitiba","Vienna","Sydney", "Rajshahi"]]
# for city in tqdm(allcity_ls):
#     # try:
#     get_result(city)
    #     print(f"city {city} done")
    # except Exception as e:
    #     print(f"city {city} failed with {e}")
# use parallel processing
import multiprocessing
from multiprocessing import Pool
pool = Pool(8)
for i in pool.imap(get_result, allcity_ls):
    pass
    
#python /home/yuanzf/uvi-time-machine/_script/e-street-diversity/02b_age_gender_summary.py