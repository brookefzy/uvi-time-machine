# load current availiable country level data first
import pandas as pd
import numpy as np
import os

ROOTFOLDER = (
    "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2023/01_city-never-was/_data"
)
RAWFOLDER = f"{ROOTFOLDER}/_raw/_road_fatality/_fatality"
TRANSFORM_FOLDER = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=country"
TRANSFORM_FOLDER_CITY = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=city"
TRANSFORM_FOLDER_STATE = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=state"

df_brazil = pd.read_excel(
    os.path.join(RAWFOLDER, "_brazil/fatality_daly_brazil_2019.xlsx"),
    header=[0, 1],
    #   engine = 'python'
)
columns_keep = [
    ("Unnamed: 0_level_0", "state"),
    ("num_person_killed", "n"),
    ("num_person_killed_per_lakh", "n"),
    ("DALY Rate", "value"),
]
df_brazil = (
    df_brazil[columns_keep]
    .droplevel(1, axis=1)
    .rename(columns={"Unnamed: 0_level_0": "state"})
)
# remove state name encoding, put into clean English name
df_brazil["state_original"] = df_brazil["state"]
encodingmapping = {
    "á": "a",
    "é": "e",
    "ã": "a",
    "í": "i",
}
df_brazil["state"] = df_brazil["state"].apply(
    lambda x: "".join(
        [encodingmapping[y] if y in encodingmapping.keys() else y for y in x]
    )
)
df_brazil["country"] = "Brazil"
df_brazil["year"] = 2019
df_brazil["level"] = "state"
df_brazil[
    [
        "country",
        "state",
        "year",
        "num_person_killed_per_lakh",
    ]
].to_csv(
    os.path.join(TRANSFORM_FOLDER_STATE, "t_road_fatality_brazil.csv"), index=False
)
print(df_brazil.head(5))
