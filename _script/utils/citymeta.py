import gspread
import pandas as pd
import numpy as np
import os

SHEETNAME = "select_city_classifier"
FATALITY_SCHEMA = {
    "source": "Data source",
    "level": "city, state or country",
    "country": "Country",
    "state": "State",
    "city": "City",
    "date_created": "Date the data was created for",
    "num_person_killed": "number of person killed",
    "num_person_killed_share": "number of person killed share",
    "num_person_killed_per_lakh": "number of person killed per lakh(100_000)",
    "num_accidents": "number of accidents",
    "num_accidents_per_lakh": "number of accidents per lakh(100_000)",
}


def load_all(sheet_name=SHEETNAME):
    serviceaccount = (
        "/Users/yuan/Dropbox (Personal)/personal files/ssh/google_drive_personal.json"
    )

    # from oauth2client.service_account import ServiceAccountCredentials
    gc = gspread.service_account(filename=serviceaccount)

    def read_url(url, sheet_name):
        SHEET_ID = url.split("/")[5]
        spreadsheet = gc.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)
        rows = worksheet.get_all_records()
        df_spread = pd.DataFrame(rows)
        return df_spread, worksheet

    url = "https://docs.google.com/spreadsheets/d/1o5gFmZPUoDwrrbfE6M26uJF3HnEZll02ivnOxP6K6Xw/edit?usp=sharing"

    city_meta, other_worksheet = read_url(url, sheet_name)
    city_meta = city_meta[city_meta["City"] != ""].reset_index(drop=True)
    city_meta["city_lower"] = city_meta["City"].apply(
        lambda x: x.lower().replace(" ", "")
    )
    return city_meta
