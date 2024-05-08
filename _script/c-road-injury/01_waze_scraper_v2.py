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
from datetime import datetime
import time
import random
from datetime import timedelta


def get_logging(today):
    logging_file = f"waza_scraper_{today}.log"
    logging.basicConfig(
        filename=logging_file, format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


serviceaccount = "/home/yuanzf/google_drive_personal.json"
gc = gspread.service_account(filename=serviceaccount)
gc_url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"

# waze url
URL_SCRAPE_OTHER = """https://www.waze.com/live-map/api/georss?top={top}&bottom={bottom}&left={left}&right={right}&env=row&types=alerts,traffic,users"""
URL_SCRAPE_US = """https://www.waze.com/live-map/api/georss?top={top}&bottom={bottom}&left={left}&right={right}&env=na&types=alerts,traffic,users"""

SHEETNAME = "select_city_classifier"


def sel_url(city_lower, city_meta):
    # if the city is a US city, use the US url
    us_city = city_meta[
        city_meta["Country"].isin(["USA", "United States", "Canada"])
    ].reset_index(drop=True)
    if city_lower in us_city["city_lower"].values:
        print("American city. Use the American URL")
        return URL_SCRAPE_US
    else:
        print("Other city. Use the other URL")
        return URL_SCRAPE_OTHER


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


def get_response(url):
    payload = {}
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "cookie": "_gcl_au=1.1.1648696555.1713863818; phpbb3_waze_u=1; phpbb3_waze_k=; phpbb3_waze_sid=0f630f02d5914814ea51dbacf461cb0b; _ga_DGC95PYF7W=GS1.1.1713863849.1.1.1713863860.0.0.0; _ga_NNRWG3BV8Y=GS1.1.1713863856.1.0.1713863860.0.0.0; partnerhub_locale=en; _web_visitorid=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiZWM5YTBlOGUtMjc2Zi00MWE1LTllMTEtOThkNGExNjQ5ZDJhIiwiaWF0IjoxNzEzODY0MTYzfQ.EQmKKX7tG5H755BqJ-MG4TKHMpMfRsN3Og6hBFj20k0; _ga=GA1.2.794042959.1713863818; _csrf_token=FqkwV65OFNmHginIOMMQGdf2GdHC5qjyTztdsl-FzB4; _gid=GA1.2.1487213100.1714027074; _web_session=dGl2ZVp1cm53d2Z2WGxVeXNMT3Vkdkx2WnZVbExkWmVtOXhrZDZ2RWhUU0c0bUZGbEkra0ZoR3cyYUNhNzE0SWRjUGhtR0ZjZGx4VlFlZDRyOEwvRkI3SkQzUURzUHhKaFpVTzRxSUtJWjNVVGZvS1NDRlBna0UySXBaU2JoQ1MtLXJ2RjlneGtOVXdMYjcxd2xkaEl4UWc9PQ%3D%3D--4c22a3fce928a5ebb6eef5f1ec8bf4773d322001; _ga_NNCFE4W9M4=GS1.2.1714032360.3.1.1714032785.0.0.0",
        "referer": "https://www.waze.com/en/live-map",
        "sec-ch-ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response


def get_data(url, max_retry, rootfolder, city_lower):
    logger = get_logging(datetime.today().strftime("%Y-%m-%d"))
    response = get_response(url)
    i = 0  # round of retry
    j = 0  # round of saving error
    if (response.status_code != 200) and (i <= max_retry):
        # retry after sleeping for 30 seconds
        time.sleep(30)
        get_data(url, max_retry=max_retry, rootfolder=rootfolder, city_lower=city_lower)
        i += 1
        print(response.status_code, "Failed to get data. Retrying", i, "times")
        logger.info(f"Failed to get data. Retrying {i} times")
    elif i > max_retry:
        print(response.status_code)
        raise ValueError("Failed to get data")
    else:
        data = response.json()

        today = datetime.today().strftime("%Y-%m-%d")
        month = today.split("-")[1]
        date = today.split("-")[-1]
        folder_save = (
            """{ROOTFOLDER}/city={city_lower}/month={month}/date={date}""".format(
                ROOTFOLDER=rootfolder,
                date=date,
                month=month,
                city_lower=city_lower,
            )
        )
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"""{folder_save}/{now}.json""", "w") as f:
            try:
                json.dump(data, f)
            except:
                # if the data cannot be saved, print the error and create a new folder to save it
                # show the error message
                print(f"Error in saving {city_lower} at {now}")
                logger.info(f"Error in saving {city_lower} at {now}")
                folder_save = """{ROOTFOLDER}_r{j}/city={city_lower}/month={month}/date={date}""".format(
                    ROOTFOLDER=rootfolder,
                    date=date,
                    month=month,
                    city_lower=city_lower,
                    j=j,
                )
                if not os.path.exists(folder_save):
                    os.makedirs(folder_save)
                with open(f"""{folder_save}/{now}.json""", "w") as f:
                    json.dump(data, f)

        logger.info(f"Scraped {city_lower} at {now}")
        print(f"Scraped {city_lower} at {now}")


def scrape_waze_city(city_lower, rootfolder, city_meta, max_retry=5):

    city_data = city_meta.loc[city_meta["city_lower"] == city_lower]
    left, bottom, right, top = (
        city_data["left"].values[0],
        city_data["bottom"].values[0],
        city_data["right"].values[0],
        city_data["top"].values[0],
    )
    # make sure the values are valid
    assert left < right, "left should be less than right"
    URL_SCRAPE = sel_url(city_lower, city_meta)
    url = URL_SCRAPE.format(left=left, bottom=bottom, right=right, top=top)
    get_data(url, max_retry=max_retry, rootfolder=rootfolder, city_lower=city_lower)
    time.sleep(random.randint(20, 40))


def scrape_waze_city_ls(cityls, rootfolder, city_meta, max_retry=5):

    # keep scraping for 1 year
    # loop through the cities
    for city_lower in cityls:
        city_data = city_meta.loc[city_meta["city_lower"] == city_lower]
        left, bottom, right, top = (
            city_data["left"].values[0],
            city_data["bottom"].values[0],
            city_data["right"].values[0],
            city_data["top"].values[0],
        )
        # make sure the values are valid
        assert left < right, "left should be less than right"
        URL_SCRAPE = sel_url(city_lower, city_meta)

        url = URL_SCRAPE.format(left=left, bottom=bottom, right=right, top=top)
        get_data(url, max_retry=max_retry, rootfolder=rootfolder, city_lower=city_lower)

        time.sleep(random.randint(20, 40))


def main():
    parser = argparse.ArgumentParser(description="waze scraper")
    parser.add_argument(
        "--rootfolder",
        type=str,
        default="/lustre1/g/geog_pyloo/05_timemachine/_raw/waze",
        help="Root folder for the GSV data",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="City Name to scrape. Default is all cities in the google sheet. If more than one city, uses the pattern of city1&&city2&&city3",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to scrape. Default is 365 days",
    )
    args = parser.parse_args()

    rootfolder = args.rootfolder
    sel_city = args.city
    print(f"Root folder is {rootfolder}")
    print(f"City to scrape is {sel_city}")
    max_days = args.days
    print(f"Scraping for {max_days} days")
    stop_time = datetime.now() + timedelta(days=max_days)
    city_meta, other_worksheet = get_city_meta(gc_url)

    if sel_city == "all":
        cityls = city_meta["city_lower"].values
        while datetime.now() < stop_time:
            scrape_waze_city_ls(cityls, rootfolder, city_meta)

    elif "&&" in sel_city:
        cityls = sel_city.split("&&")
        while datetime.now() < stop_time:
            scrape_waze_city_ls(cityls, rootfolder, city_meta)
    else:
        print(f"Scraping {sel_city}")
        city_lower = sel_city.replace(" ", "").lower()
        while datetime.now() < stop_time:
            scrape_waze_city(sel_city, rootfolder, city_meta)


if __name__ == "__main__":
    main()
