import numpy as np
import os
import pandas as pd
import glob
import h3
from tqdm import tqdm


ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
CURATED_FOLDER = f"{ROOTFOLDER}/_curated"
META_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/{cityabbr}_meta.csv"

EXFOLDER_LONG = os.path.join(CURATED_FOLDER, "c_seg_longitudinal_year")
if not os.path.exists(EXFOLDER_LONG):
    os.makedirs(EXFOLDER_LONG)
    
TOTAL_PX = 160000
H3_RES = [8, 9, 12]

##################################################################
############### CLEAN UP SEG TYPE
##################################################################


def get_result(cityabbr, curated_folder, f_suffixes = "*panoptic.csv"):
    outfolder = f"{curated_folder}/{cityabbr}"
    seg_file = glob.glob(os.path.join(outfolder, f_suffixes))
    panoptic_df = []
    for p in seg_file:
        temp = pd.read_csv(p)
        panoptic_df.append(temp)
    panoptic_df = pd.concat(panoptic_df).reset_index(drop = True)
    return panoptic_df

def clean_seg(seg_df, pano_df, meta_df):

    seg_df_filtered = seg_df.merge(meta_df, on = 'img')
    seg_df_filtered = seg_df_filtered[seg_df_filtered['size']>=10000].reset_index(drop = True)
    print("Segmentation shape after filter: ", seg_df_filtered.shape[0])
    seg_df_summary = seg_df_filtered.groupby(["img", "labels"]).agg({'areas':'sum'}).reset_index()
    seg_df_summary['panoid'] = seg_df_summary['img'].apply(lambda x: x[:22])

    col_cols = ["labels"]
    index_cols = ["img", "year", "h3_8", "h3_9", "h3_12"]
    seg_df_summary_pano = seg_df_summary.merge(pano_df, on = ['panoid'])
    
    
    if seg_df_summary_pano.shape[0]<seg_df_summary.shape[0]:
        print("data missing after data join.")
        print("Before join: ", seg_df_summary.shape[0])
        print("After join: ",seg_df_summary_pano.shape[0])
    else:
        print("data consistent")
    
    seg_df_summary = seg_df_summary_pano.drop_duplicates(index_cols+col_cols)
    print("Segmentation shape: ", seg_df_summary.shape[0])
    seg_df_pivot = seg_df_summary.pivot(
        columns = col_cols,
        index = index_cols,
        values = "areas"
    ).reset_index().fillna(0)
    return seg_df_pivot

def get_opt(seg_df_pivot):
    all_labels = [x for x in seg_df_pivot.columns if str(x) in [str(s) for s in range(150)]]
    print("label length: ", len(all_labels))
    ops = {"img":"count"}
    for o in all_labels:
        # ops[o] = "mean"
        ops[o] = 'sum' # change to sum, and get average later uses the image count
    return ops

# assume the data can be understand every year
def get_longitudinal(seg_df_pivot):
    ops = get_opt(seg_df_pivot)
    h3_summary = []
    for res in H3_RES:
        # for each resolution of h3 id we get a average pixel of one category
        df_h3_summary = seg_df_pivot.groupby([f'h3_{res}','year']).agg(ops).reset_index()\
        .rename(columns = {f'h3_{res}':"hex_id", "img":"img_count"})
        df_h3_summary["res"] = res
        h3_summary.append(df_h3_summary)
    h3_summary = pd.concat(h3_summary).reset_index(drop = True)
    return h3_summary


def load_data(city):
    cityabbr = city.lower().replace(" ", "")
    seg_df = get_result(cityabbr, CURATED_FOLDER, f_suffixes = "*seg.csv")
    pano_df = pd.read_csv(PANO_PATH.format(
    ROOTFOLDER = ROOTFOLDER,
    cityabbr = cityabbr
    ))[['panoid', 'lat', 'lon', 'year', 'month']]

    for res in H3_RES:
        pano_df[f'h3_{res}'] = pano_df.apply(lambda x: h3.geo_to_h3(x.lat, x.lon, res), axis=1)
        
    meta_df = pd.read_csv(META_PATH.format(
        ROOTFOLDER = ROOTFOLDER,
        cityabbr = cityabbr
    ))
    meta_df['img']= meta_df['path'].apply(lambda x: x.split("/")[-1].split(".")[0])
    # here make sure 
    meta_df = meta_df[['img','size']]
    
    seg_df_pivot = clean_seg(seg_df, pano_df, meta_df)
    seg_longitudinal = get_longitudinal(seg_df_pivot)
    seg_longitudinal.columns = [str(x) for x in seg_longitudinal.columns]
    seg_longitudinal.to_parquet(os.path.join(EXFOLDER_LONG, cityabbr+".parquet"), index = False)
    print(f"city {cityabbr} saved")
    print("*"*50)
    
def check_finished():
    finished = [x.split(".")[0] for x in os.listdir(EXFOLDER_LONG)]
    return finished
    
def load_all():

    city_meta = pd.read_csv("../city_meta.csv")
    return city_meta
# step 1

def main():
    city_meta = load_all()
    # finished = check_finished()
    allcity = city_meta["City"].values
    # city_to_process = ["Chicago"]
    for city in tqdm(allcity):
        cityabbr = city.lower().replace(" ", "")

        # try:
        load_data(cityabbr)
        # except:
        #     print(f"check problem for this city {city}")

if __name__ == "__main__":
    main()
    