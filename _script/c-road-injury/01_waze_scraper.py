import requests
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import gspread
import logging
import geopandas as gpd
import json
import multiprocessing as mp
import argparse

logging.basicConfig(
    filename="waza_scraper.log", format="%(asctime)s %(message)s", filemode="w"
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

serviceaccount = "../../google_drive_personal.json"
gc = gspread.service_account(filename=serviceaccount)
gc_url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"

# ROOTFOLDER = "/group/geog_pyloo/08_GSV" # set as an argument
PATH_TO_ROAD = "{ROOTFOLDER}/gsv_rgb/{citylower}/road/osm.geojson"
# waze url
URL_SCRAPE = """https://www.waze.com/live-map/api/georss?top={top}&bottom={bottom}&left={left}&right={right}&env=row&types=alerts,traffic"""

SHEETNAME = "select_city_classifier"


def read_url(url, SHEET_NAME):
    SHEET_ID = url.split("/")[5]
    spreadsheet = gc.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    rows = worksheet.get_all_records()
    df_spread = pd.DataFrame(rows)
    return df_spread, worksheet


def get_city_meta(gc_url):
    city_meta, other_worksheet = read_url(gc_url, SHEETNAME)
    city_meta = city_meta[city_meta["City"] != ""].reset_index(drop=True)
    city_meta = city_meta[city_meta["GSV Downloaded"] > 0].reset_index(drop=True)
    city_meta["city_lower"] = city_meta["City"].apply(
        lambda x: x.replace(" ", "").lower()
    )
    return city_meta, other_worksheet


# loop through the cities and get road boundaries
def get_bound(test_city, rootfolder, city_meta):
    citylower = test_city["city_lower"]
    path_to_road = PATH_TO_ROAD.format(ROOTFOLDER=rootfolder, citylower=citylower)
    road = gpd.read_file(path_to_road)
    # extract the bounding box
    road.crs = "EPSG:3857"
    road = road.to_crs(epsg=4326)

    left, bottom, right, top = road.total_bounds
    print(left, bottom, right, top)
    logger.info(f"Bounding box for {citylower} is {left, bottom, right, top}")
    # make sure the bounding box is valid coordinates in the world
    if (
        left < right
        and bottom < top
        and -180 <= left <= 180
        and -90 <= bottom <= 90
        and -180 <= right <= 180
        and -90 <= top <= 90
    ):
        return left, bottom, right, top
    else:
        logger.error(f"Bounding box for {citylower} is not valid")
        raise ValueError("Bounding box is not valid")


# loop through all cities and add the bound to the google sheet
def add_bound_to_sheet(city_meta):
    for i in tqdm(range(len(city_meta))):
        print(city_meta.loc[i])
        test_city = city_meta.loc[i]
        left, bottom, right, top = get_bound(test_city, city_meta)
        city_meta.loc[i, "left"] = left
        city_meta.loc[i, "bottom"] = bottom
        city_meta.loc[i, "right"] = right
        city_meta.loc[i, "top"] = top
        print("*" * 20, "done", "*" * 20)
    return city_meta


def scrape_waze_city(city_lower, rootfolder, waze_folder="waze_data"):
    from datetime import datetime

    city_data = city_meta.loc[city_meta["city_lower"] == city_lower]
    left, bottom, right, top = (
        city_data["left"].values[0],
        city_data["bottom"].values[0],
        city_data["right"].values[0],
        city_data["top"].values[0],
    )
    url = URL_SCRAPE.format(left=left, bottom=bottom, right=right, top=top)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        folder_save = (
            """{ROOTFOLDER}/{waze_folder}/city={city_lower}/datetime={now}""".format(
                ROOTFOLDER=rootfolder,
                waze_folder=waze_folder,
                now=now,
                city_lower=city_lower,
            )
        )
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        with open(f"""{folder_save}/waze.json""", "w") as f:
            json.dump(data, f)
        logger.info(f"Scraped {city_lower}")
    else:
        logger.error(f"Failed to scrape {city_lower} at {now}")


def main():
    args = argparse.ArgumentParser()
    rootfolder = args.rootfolder
    city_meta, other_worksheet = get_city_meta(gc_url)
    city_meta = add_bound_to_sheet(city_meta)
    # update sheet
    city_meta = city_meta.drop(columns=["city_lower"])
    other_worksheet.update(
        [city_meta.columns.values.tolist()] + city_meta.values.tolist()
    )
    print("Updated the google sheet")
    print("*" * 50)
    print(
        """Start scraping waze data. 
          Scrape all cities in the google sheet.
          Set jobs in parrellel using pooling
          """
    )
    ## NOT DONE YET
    pool = mp.Pool(mp.cpu_count())
