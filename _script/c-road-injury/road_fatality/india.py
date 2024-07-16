india_path = [
    "India_num_person_killed_2019_2022.xlsx",
    "India_total_accidents_2019_2022.xlsx",
    "India_num_fatal_accidents_state_2019_2022.xlsx",
]
columns_mapping = {
    "Cities": "city",
    "Fatal Accidents": "num_fatal_accidents",
    "No. of Persons Killed": "num_person_killed",
    "No of persons Killed": "num_person_killed",
    "Total Accidents": "total_accidents",
    "No. of Persons Injured": "num_person_injured",
    "No of persons injured": "num_person_injured",
    "Injury Accidents": "num_injury_accidents",
    "Severity of Accidents": "severity",
    "States/UTs": "city",
}
india_city = {
    "Bengaluru": "Bangalore",
}
india_mapping = dict(
    zip(
        [
            "State/UT-Wise Total Number of  Persons Killed in Road Accidents during",
            "Share of States/UTs in Total Number of Persons Killed in Road Accidents",
            "Total Number of Persons Killed in Road Accidents Per Lakh Population",
            "Total Number of Persons Killed in Road Accidents per 10,000 Vehicles",
            "Total Number of Persons Killed in Road Accidents per 10,000 Km of Roads",
            "State/UT-Wise Total Number of Road Accidents during",
            "Share of States/UTs in Total Number of Road Accidents",
            "Total Number of Accidents Per Lakh Population",
            "Total Number of  Road Accidents per 10,000\nVehicles",
            "Total Number of  Road Accidents per 10,000 Km of\nRoads",
        ],
        [
            "num_person_killed",
            "num_person_killed_share",
            "num_person_killed_per_lakh",
            "num_person_killed_per_10k_vehicle",
            "num_person_killed_per_10k_km_roads",
            "num_accidents",
            "num_accidents_share",
            "num_accidents_per_lakh",
            "num_accidents_per_10k_vehicle",
            "num_accidents_per_10k_km_roads",
        ],
    )
)
vari_keep = [
    "num_person_killed",
    "num_person_killed_per_lakh",
    "num_fatal_accidents",
    "num_accidents",
    "num_accidents_per_lakh",
    "population",
]

import gspread
import pandas as pd
import numpy as np
import os


def load_all():
    serviceaccount = (
        "/Users/yuan/Dropbox (Personal)/personal files/ssh/google_drive_personal.json"
    )
    import gspread

    # from oauth2client.service_account import ServiceAccountCredentials
    gc = gspread.service_account(filename=serviceaccount)

    def read_url(url, SHEET_NAME):
        SHEET_ID = url.split("/")[5]
        spreadsheet = gc.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        rows = worksheet.get_all_records()
        df_spread = pd.DataFrame(rows)
        return df_spread, worksheet

    url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"
    SHEETNAME = "select_city_classifier"
    city_meta, other_worksheet = read_url(url, SHEETNAME)
    city_meta = city_meta[city_meta["City"] != ""].reset_index(drop=True)
    city_meta["city_lower"] = city_meta["City"].apply(
        lambda x: x.lower().replace(" ", "")
    )
    return city_meta


ROOTFOLDER = (
    "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2023/01_city-never-was/_data"
)
RAWFOLDER = f"{ROOTFOLDER}/_raw/_road_fatality/_fatality"
TRANSFORM_FOLDER = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=country"
TRANSFORM_FOLDER_CITY = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=city"
TRANSFORM_FOLDER_STATE = f"{ROOTFOLDER}/_transformed/t_road_fatality/level=state"
for t in [TRANSFORM_FOLDER, TRANSFORM_FOLDER_CITY, TRANSFORM_FOLDER_STATE]:
    if not os.path.exists(t):
        os.makedirs(t)
city_meta = load_all()
starter_df = city_meta[
    [
        "City",
        "Country",
        "State/Province",
        "city_lower",
        "country_clean",
        "county_ls",
        "urban_pop",
    ]
]


# 2019-2022 India
def get_19_india(india_path):
    india_df = pd.read_excel(
        os.path.join(RAWFOLDER, "India", india_path[0]), header=[0, 1]
    )
    india_df_killed_long = india_df.melt(
        id_vars=[("States/Uts", "Unnamed: 0_level_1")],
        value_vars=list(india_df.columns[1:]),
        var_name=["variables", "year"],
        value_name="values",
    ).rename(columns={("States/Uts", "Unnamed: 0_level_1"): "state"})
    india_df_killed_long["variables"] = india_df_killed_long["variables"].map(
        india_mapping
    )
    india_df_killed_long["year"] = india_df_killed_long["year"].astype(int)
    india_df_killed_long = india_df_killed_long[
        india_df_killed_long["variables"].isin(vari_keep)
    ].reset_index(drop=True)
    india_df_killed = india_df_killed_long.pivot(
        index=["state", "year"], columns="variables", values="values"
    ).reset_index()
    # india_df_killed

    # Number of Fatal Accidents
    india_fatal_df = pd.read_excel(
        os.path.join(RAWFOLDER, "India", india_path[2]), header=[0]
    ).rename(columns={"States/UTs": "state"})
    india_fatal_df = (
        india_fatal_df.set_index("state")
        .stack()
        .reset_index()
        .rename(columns={"level_1": "year", 0: "num_fatal_accidents"})
    )

    # Number of total accidents
    india_accident_df = pd.read_excel(
        os.path.join(RAWFOLDER, "India", india_path[1]), header=[0, 1]
    ).rename(columns={"States/UTs": "state"})
    india_accident_long = india_accident_df.melt(
        id_vars=[("state", "Unnamed: 0_level_1")],
        value_vars=list(india_accident_df.columns[1:]),
        var_name=["variables", "year"],
        value_name="values",
    ).rename(columns={("state", "Unnamed: 0_level_1"): "state"})
    india_accident_long["variables"] = india_accident_long["variables"].map(
        india_mapping
    )
    india_accident_long["year"] = india_accident_long["year"].astype(int)
    india_accident = (
        india_accident_long[india_accident_long["variables"].isin(vari_keep)]
        .reset_index(drop=True)
        .pivot(index=["state", "year"], columns="variables", values="values")
        .reset_index()
    )

    statemapping = {
        "D & N Haveli": "Dadra & Nagar Haveli",
        "A & N Islands": "Andaman & Nicobar Islands",
    }
    india_accident["state"] = india_accident["state"].apply(
        lambda x: statemapping[x] if x in statemapping.keys() else x
    )
    india_df_killed["state"] = india_df_killed["state"].apply(
        lambda x: statemapping[x] if x in statemapping.keys() else x
    )
    india_fatal_df["state"] = india_fatal_df["state"].apply(
        lambda x: statemapping[x] if x in statemapping.keys() else x
    )

    india_df = india_accident.merge(
        india_df_killed, on=["state", "year"], how="outer"
    ).merge(india_fatal_df, on=["state", "year"], how="outer")
    india_df["num_person_killed_per_lakh"] = india_df[
        "num_person_killed_per_lakh"
    ].apply(lambda x: x if x != "#" else np.nan)
    return india_df


def get_india_16(india_path):
    # 2013-2016 India state-level
    india_df_2013_2016 = pd.read_excel(
        os.path.join(
            RAWFOLDER, "India", "Pages from Road_Accidents_in_India_2016-2-state.xlsx"
        ),
        header=[0, 1],
        index_col=0,
    )
    india_df_2013_2016.stack().reset_index().rename(
        columns={
            "level_0": "state",
            "level_1": "year",
        }
    )

    india_2016 = pd.read_excel(
        os.path.join(
            RAWFOLDER,
            "India",
            "Pages from Road_Accidents_in_India_2016_city_level.xlsx",
        )
    )

    india_2016 = (
        india_2016[india_2016["Cities"].notnull()]
        .copy()
        .rename(columns=columns_mapping)
    )
    # india_2019 = india_2019[india_2019['States/UTs'].notnull()].copy().rename(
    #     columns = columns_mapping
    # )
    india_2016["year"] = 2016

    # map city names
    india_2019 = pd.read_excel(
        os.path.join(
            RAWFOLDER,
            "India",
            "Pages from Road_Accidents_in_India_2019_city_level.xlsx",
        ),
        header=[0, 1],
        index_col=[0, 1],
    )
    india_2019 = (
        india_2019.stack()
        .reset_index()
        .rename(columns={"level_1": "city", "level_2": "year"})
        .rename(columns=columns_mapping)
    )
    # india_2019['year'] = 2019
    india_city_df = pd.concat([india_2016, india_2019], axis=0)
    india_city_df["city"] = india_city_df["city"].apply(
        lambda x: india_city[x] if x in india_city.keys() else x
    )
    india_city_df = india_city_df[
        [
            "city",
            "num_fatal_accidents",
            "num_person_killed",
            "total_accidents",
            "num_person_injured",
            "year",
        ]
    ]
    return india_city_df


india_df = get_19_india(india_path)
india_city_df = get_india_16(india_path)

starter_india = starter_df[starter_df["Country"] == "India"].copy()
with_city_india = starter_india.merge(
    india_city_df, left_on="City", right_on="city", how="inner"
)
with_city_india["num_person_killed_per_lakh"] = (
    with_city_india["num_person_killed"] / with_city_india["urban_pop"] * 100000
)
with_city_india["fatality_source"] = "city_level"

no_city_india = starter_india[
    ~starter_india["City"].isin(with_city_india["City"])
].copy()
print(no_city_india["State/Province"].unique())
no_city_india = no_city_india.merge(
    india_df, left_on="State/Province", right_on="state", how="inner"
)
print(no_city_india["State/Province"].unique())
no_city_india["fatality_source"] = "state_level"
india_df.to_csv(
    os.path.join(TRANSFORM_FOLDER_STATE, "t_road_fatality_india.csv"), index=False
)
