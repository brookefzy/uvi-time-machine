# %% [markdown]
# # Calculate the hexagon metrics
# 1. Distance from urban center
# 

# %%
import pandas as pd
import os
import h3
from glob import glob
# compute the harvesine distance between the (lat, lon) of the hex_id and the urban center
import haversine as hs
from haversine import Unit

res_exclude = 11
CURATE_FOLDER_SOURCE = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_hex_summary"
CURATE_FOLDER_EXPORT = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity"

files = glob(CURATE_FOLDER_SOURCE + f"/*res_exclude={res_exclude}.parquet")

city_meta = pd.read_csv("/home/yuanzf/uvi-time-machine/_script/city_meta.csv")
city_meta = city_meta[["center_lat", "center_lng", "City"]]
city_meta["city_lower"] = city_meta["City"].str.lower().apply(lambda x: x.replace(" ", ""))
city_meta['city_lower'].unique()

# %%
import gc
gc.collect()

# %% [markdown]
# # load similarity index

# %%
res_sel = 7
# dist_matrix_df = pd.read_parquet(CURATE_FOLDER_EXPORT + f"/h3_hex_pairwise_dist_res={res_sel}.parquet")
similarit_summary_overall = pd.read_csv(os.path.join(CURATE_FOLDER_EXPORT, f'similarity_summary_connection_res={res_sel}.csv'))\
    .rename(columns ={0:'count'})
similarit_summary_overall

# %%

files = glob(CURATE_FOLDER_EXPORT + f"/*res={res_sel}*.parquet")
files = [f for f in files if "similarity" in f]
similarity_df =[]
for f in files:
    temp = pd.read_parquet(os.path.join(CURATE_FOLDER_EXPORT,f))
    similarity_df.append(temp)
similarity_df = pd.concat(similarity_df).reset_index(drop = True)

# %%
def get_city_pair(similarit_summary_overall, city_meta):
    similarit_summary_overall_meta = similarit_summary_overall.merge(city_meta[['center_lat', 'center_lng', 'City']], 
                                                                    left_on = 'city_1', right_on = 'City', how = 'inner')\
                                                                        .drop(columns = ['City'])\
                                                            .merge(city_meta[['center_lat', 'center_lng', 'City']], 
                                                                    left_on = 'city_2', right_on = 'City', how = 'inner')\
                                                                        .drop(columns = ['City'])
                                                                        
    
    similarit_summary_overall_meta['center_dist'] = similarit_summary_overall_meta.apply(lambda x: hs.haversine((x['center_lat_x'], x['center_lng_x']),
                                                                                                            (x['center_lat_y'], x['center_lng_y']),
                                                                                                            unit = Unit.METERS), axis = 1)
    similarit_summary_overall = \
    similarit_summary_overall_meta.groupby(['city_1', 'city_2']).agg({

        'similarity': 'mean',
        'center_dist': 'mean'
    }).reset_index().rename(columns = {"similarity": "mean_similarity"})
    similarit_summary_overall = similarit_summary_overall[similarit_summary_overall['mean_similarity'] > 0].reset_index(drop = True)

    return similarit_summary_overall

# %%
n = 10
print(f"n = {n}")
df_all_update_sel = df_all_update[df_all_update["h3_cbd_dist"]<n*1000]
df_all_update_sel.shape
h3_hex_ids = df_all_update_sel['hex_id'].unique().tolist()
similarity_df_sel = similarity_df[(similarity_df['hex_id1'].isin(h3_hex_ids))&(similarity_df['hex_id2'].isin(h3_hex_ids))]\
    .reset_index(drop = True)
gc.collect()

similarity_summary = get_city_pair(similarity_df_sel, city_meta)
similarity_summary.to_csv(os.path.join(CURATE_FOLDER_EXPORT, f'similarity_summary_connection_res={res_sel}_n={n}.csv'), index = False)


# %%
for n in range(2,21, 2):
    print(f"n = {n}")
    df_all_update_sel = df_all_update[df_all_update["h3_cbd_dist"]<n*1000]
    df_all_update_sel.shape
    h3_hex_ids = df_all_update_sel['hex_id'].unique().tolist()
    similarity_df_sel = similarity_df[(similarity_df['hex_id1'].isin(h3_hex_ids))&(similarity_df['hex_id2'].isin(h3_hex_ids))]\
        .reset_index(drop = True)
    gc.collect()
    
    similarity_summary = get_city_pair(similarity_df_sel, city_meta)
    similarity_summary.to_csv(os.path.join(CURATE_FOLDER_EXPORT, f'similarity_summary_connection_res={res_sel}_n={n}.csv'), index = False)


# %%
similarity_df_sel_meta = get_city_pair(similarity_df_sel, city_meta)

# %%
import gc
gc.collect()

# %%
similarity_df_sel_meta.head()

# %%
# similarit_summary_overall = similarit_summary_overall[similarit_summary_overall['mean_similarity'] > 0.005].reset_index(drop = True)
# print(similarit_summary_overall.shape)
similarit_summary_overall_meta.to_csv(os.path.join(CURATE_FOLDER_EXPORT, f'similarity_summary_connection_res={res_sel}.csv'), index = False)

# %%
similarity_df_sel_meta.to_csv(os.path.join(CURATE_FOLDER_EXPORT, f'similarity_summary_connection_res={res_sel}_distcenter10km.csv'), index = False)




