import numpy as np
import os
import pandas as pd
import glob
import h3


ROOTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine"
PANO_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/gsv_pano.csv"
CURATED_FOLDER = f"{ROOTFOLDER}/_curated"
META_PATH = "{ROOTFOLDER}/GSV/gsv_rgb/{cityabbr}/gsvmeta/{cityabbr}_meta.csv"

EXFOLDER = os.path.join(CURATED_FOLDER, "c_pano_crossectional")
if not os.path.exists(EXFOLDER):
    os.makedirs(EXFOLDER)

EXFOLDER_LONG = os.path.join(CURATED_FOLDER, "c_pano_longitudinal")
if not os.path.exists(EXFOLDER_LONG):
    os.makedirs(EXFOLDER_LONG)

TOTAL_PX = 160000
H3_RES = [8, 9, 12]


##################################################################
############### CLEAN UP SEG TYPE
##################################################################
def get_seg_types():
    building = [0, 1, 25, 48]
    greenery = [4, 9, 17, 66, 72]
    street_furniture = [19, 15, 31, 69, 82, 136, 138]
    sidewalk = [11]
    car = [20, 80, 83, 102]
    person = [12]
    bike = [127, 116]
    sky = [2]
    sel = building + greenery + street_furniture + sidewalk + car + person + bike + sky
    other = [x for x in range(150) if not x in sel]

    obj_dicts = {
        # "building":building,
        # "greenery":greenery,
        "street_furniture": street_furniture,
        # "sidewalk": sidewalk,
        "car": car,
        "person": person,
        "bike": bike,
        # "sky":sky,
        "other": other,
    }

    def get_cat(label):
        for obj, v in obj_dicts.items():
            if label in v:
                return obj

    obj_dict_rev = {}
    for x in range(150):
        obj_dict_rev[x] = get_cat(x)
    ops = {}
    for label in list(obj_dicts.keys())[:-1]:
        ops[label] = "mean"
    return ops, obj_dict_rev


def get_result(cityabbr, curated_folder, f_suffixes="*panoptic.csv"):
    outfolder = f"{curated_folder}/{cityabbr}"
    seg_file = glob.glob(os.path.join(outfolder, f_suffixes))
    panoptic_df = []
    for p in seg_file:
        temp = pd.read_csv(p)
        panoptic_df.append(temp)
    panoptic_df = pd.concat(panoptic_df).reset_index(drop=True)
    return panoptic_df


def clean_pano(panoptic_df, ring):
    _, obj_dict_rev = get_seg_types()
    panoptic_df = panoptic_df[panoptic_df["img"].notnull()].reset_index(drop=True)
    panoptic_df["cat"] = panoptic_df["category_id"].apply(lambda x: obj_dict_rev[x])

    panoptic_df = panoptic_df[
        ~panoptic_df["cat"].isin(["other", "greenery"])
    ].reset_index(drop=True)
    panoptic_df["panoid"] = panoptic_df["img"].apply(lambda x: x[:22])
    panoptic_df_summary = (
        panoptic_df[panoptic_df["isthing"] == True]
        .groupby(["panoid", "cat"])["id"]
        .count()
        .reset_index()
    ).rename(
        columns={"id": "obs"}
    )  # count the number of objects at panoid level first

    panoptic_df_summary = panoptic_df_summary[
        panoptic_df_summary["cat"] != "other"
    ].reset_index(drop=True)

    panoptic_df_summary_pano = panoptic_df_summary.merge(
        ring, on=["panoid"], how="left"
    )
    if panoptic_df_summary_pano.shape[0] < panoptic_df_summary.shape[0]:
        print("data missing after data join.")
    else:
        print("data consistent")
    col_cols = ["cat"]
    index_cols = ["panoid", "year", "h3_8", "h3_9", "h3_12"]
    panoptic_df_summary_pano = panoptic_df_summary_pano.drop_duplicates(
        index_cols + col_cols
    )
    print("data shape: ", panoptic_df_summary_pano.shape[0])
    panoptic_df_pivot = (
        panoptic_df_summary_pano.pivot(columns=col_cols, index=index_cols, values="obs")
        .reset_index()
        .fillna(0)
    )
    return panoptic_df_pivot


def get_crossectional(panoptic_df_pivot, ops):

    h3_summary_no_year = []
    for res in H3_RES:
        # for each resolution of h3 id we get a average pixel of one category
        df_h3_summary = (
            panoptic_df_pivot.groupby([f"h3_{res}"])
            .agg(ops)
            .reset_index()
            .rename(columns={f"h3_{res}": "hex_id"})
        )
        df_h3_summary["res"] = res
        h3_summary_no_year.append(df_h3_summary)
        print("resolution: ", res)
    h3_summary_no_year = pd.concat(h3_summary_no_year).reset_index(drop=True)
    return h3_summary_no_year


# assume the data can be understand every year
def get_longitudinal(panoptic_df_pivot, ops):
    null_group = [2019]  # do not use this for now
    panoptic_df_summary_pano = panoptic_df_pivot[
        ~panoptic_df_pivot["year"].isin(null_group)
    ].reset_index(drop=True)
    panoptic_df_summary_pano["year_group"] = np.where(
        panoptic_df_summary_pano["year"] <= 2018, "2015-2018", "2020-2023"
    )
    h3_summary = []
    for res in H3_RES:
        # for each resolution of h3 id we get a average pixel of one category
        df_h3_summary = (
            panoptic_df_summary_pano.groupby([f"h3_{res}", "year_group"])
            .agg(ops)
            .reset_index()
            .rename(columns={f"h3_{res}": "hex_id"})
        )
        df_h3_summary["res"] = res
        h3_summary.append(df_h3_summary)
    h3_summary = pd.concat(h3_summary).reset_index(drop=True)
    return h3_summary


def load_data(city, ops):
    cityabbr = city.lower().replace(" ", "")
    panoptic_df = get_result(cityabbr, CURATED_FOLDER)
    pano_df = pd.read_csv(PANO_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr))[
        ["panoid", "lat", "lon", "year", "month"]
    ]

    for res in H3_RES:
        pano_df[f"h3_{res}"] = pano_df.apply(
            lambda x: h3.geo_to_h3(x.lat, x.lon, res), axis=1
        )

    meta_df = pd.read_csv(META_PATH.format(ROOTFOLDER=ROOTFOLDER, cityabbr=cityabbr))

    ring = (
        meta_df[["panoid", "ring"]]
        .drop_duplicates()
        .dropna()
        .merge(pano_df, on="panoid", how="right")
    )
    ring["ring"] = ring["ring"].fillna(-1)
    # here make sure

    panoptic_df_pivot = clean_pano(panoptic_df, ring)
    panoptic_crossectional = get_crossectional(panoptic_df_pivot, ops)
    panoptic_longitudinal = get_longitudinal(panoptic_df_pivot, ops)

    panoptic_crossectional.to_parquet(
        os.path.join(EXFOLDER, cityabbr + ".parquet"), index=False
    )
    panoptic_longitudinal.to_parquet(
        os.path.join(EXFOLDER_LONG, cityabbr + ".parquet"), index=False
    )
    print(f"city {cityabbr} saved")
    print("*" * 50)


def check_finished():
    finished = [x.split(".")[0] for x in os.listdir(EXFOLDER)]
    return finished


def load_all():
    serviceaccount = "../../google_drive_personal.json"
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
    return city_meta


# step 1


def main():
    city_meta = load_all()
    ops, obj_dict_rev = get_seg_types()
    city_to_process = city_meta["City"].tolist()

    for city in city_to_process:
        # if not cityabbr in finished:
        #     print(f"{city} has not been processed.")
        try:
            load_data(city, ops)
        except:
            print(f"check problem for this city {city}")
        else:
            print("This city is done")


if __name__ == "__main__":
    main()
