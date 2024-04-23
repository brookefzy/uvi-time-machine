import pandas as pd
import os
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import glob
from multiprocessing import Pool
import concurrent.futures
import datetime


def get_path(subfolder):
    files = glob.glob(os.path.join(subfolder, "*/*/*.jpg"))
    return files


def get_file_size(file):
    try:
        return file, os.path.getsize(file)
    except OSError as e:
        print(f"Error: {e}")
        return file, None


class GSVSummary:
    # year1 = 2018
    # year2 = 2020
    def __init__(self, city):
        self.city = city
        self.citylower = city.lower().replace(" ", "")
        self.generalgsv = (
            "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb/{citylower}".format(
                citylower=self.citylower
            )
        )
        self.imgfolder = self.generalgsv + "/img_rgb"
        self.metafolder = self.generalgsv + "/gsvmeta"
        self.roadfolder = self.generalgsv + "/road"
        self.selpanopath = os.path.join(self.metafolder, "gsv_pano_label.csv")
        # self.pairpath = os.path.join(self.metafolder,
        #                              'paired_{year1}_{year2}.csv'.format(year1 = self.year1, year2 = self.year2))
        self.today = datetime.datetime.today().strftime("%Y-%m-%d")

    def load_gsv_path(self):
        panodf = pd.read_csv(os.path.join(self.metafolder, "gsv_path.csv"))
        panodf["angle"] = panodf["path"].apply(lambda x: x.split("/")[-1][23:-4])
        panodf = panodf[panodf["angle"] != ""].reset_index(drop=True)
        panodf["angle"] = panodf["angle"].astype(int)
        panodf["panoid"] = panodf["path"].apply(lambda x: x.split("/")[-1][:22])
        return panodf

    def load_gsv_meta(self):
        metadf = pd.read_csv(os.path.join(self.metafolder, "gsv_pano.csv"))
        return metadf

    def check_gsv_download(self):

        panodf = self.load_finshed_gsv()
        # save panodf to folder
        return panodf.shape[0]

    def load_finshed_gsv(self):
        subfolders = os.listdir(self.imgfolder)
        subfolders = [os.path.join(self.imgfolder, sub) for sub in subfolders]
        pool = Pool(8)
        allfiles = []
        for ret in tqdm(pool.imap(get_path, subfolders), total=len(subfolders)):
            allfiles.append(ret)
        allfiles = [item for sublist in allfiles for item in sublist]
        panodf = pd.DataFrame(allfiles, columns=["path"])
        if panodf.shape[0] == 0:
            print("No images available.")
            return pd.DataFrame()
        else:
            panodf["panoid"] = panodf["path"].apply(
                lambda x: x.split("/")[-1].split(".")[0][:22]
            )
            # panodf.to_csv(
            #     os.path.join(self.metafolder, f"gsv_path_{self.today}.csv"), index=False
            # )
        return panodf

    def load_gsv_sel_meta(self):
        metadf_sel = pd.read_csv(self.selpanopath)
        return metadf_sel

    def merge_meta(self, sel=True):
        if sel:
            metadf = self.load_gsv_sel_meta()
        else:
            metadf = self.load_gsv_meta()
        panodf = self.load_gsv_path()
        metadf_update = panodf.merge(metadf, on="panoid", how="inner")
        return metadf_update

    def check_road(self):
        if os.path.exists(os.path.join(self.metafolder, "sentPt.p")):
            return 1
        else:
            return 0

    def check_pano(self):
        panopath = os.path.join(self.metafolder, "gsv_pano.csv")
        if os.path.exists(panopath):
            df = pd.read_csv(panopath)
            return 1, len(df)
        else:
            return 0, 0

    def construct_pair(self, year1, year2, maxdist=5):
        metadf = self.load_gsv_meta()
        panodf = self.load_gsv_path()
        metadf = metadf[metadf["panoid"].isin(panodf["panoid"])].reset_index(drop=True)
        print("Total number of GSV panoids: ", len(metadf))
        # Construct image pairs based on smallest distance
        metagdf = gpd.GeoDataFrame(
            metadf, geometry=gpd.points_from_xy(metadf["lon"], metadf["lat"]), crs=4326
        )
        metagdf = metagdf.to_crs(3857)
        metagdf_first = metagdf[
            (metagdf["year"] >= (year1 - 3)) & (metagdf["year"] <= year1)
        ].reset_index(drop=True)
        metagdf_second = metagdf[
            (metagdf["year"] >= year2) & (metagdf["year"] <= year2 + 3)
        ].reset_index(drop=True)
        print("Frist group size: ", len(metagdf_first))
        print("Second group size: ", len(metagdf_second))

        # Construct image pairs based on smallest distance
        tree = metagdf_second.sindex
        lst = tree.nearest(
            metagdf_first["geometry"], return_all=False, return_distance=True
        )
        lst_df = pd.DataFrame(columns=["index_left", "index_right"])
        lst_df["index_left"] = lst[0][0]
        lst_df["index_right"] = lst[0][1]
        lst_df["distance"] = lst[1]
        lst_df = lst_df.sort_values(["distance"]).groupby(["index_left"]).head(1)

        # merge the pairs
        locmerge = (
            metagdf_first[["panoid", "year", "lat", "lon", "month"]]
            .merge(lst_df, left_index=True, right_on="index_left", how="left")
            .merge(
                metagdf_second[["panoid", "year", "lat", "lon", "month"]],
                left_on="index_right",
                right_index=True,
                how="left",
                suffixes=("_first", "_second"),
            )
            .drop(["index_left", "index_right"], axis=1)
        )
        locmerge = locmerge[locmerge["distance"] <= maxdist].reset_index(drop=True)
        return locmerge

    def save_paired(self, locmerge):
        locmerge.to_csv(self.pairpath, index=False)
        return None

    def get_gsv_file_size(self):
        def get_file_size(file):
            try:
                return file, os.path.getsize(file)
            except OSError as e:
                print(f"Error: {e}")
                return file, None

        gsvpath = pd.read_csv(os.path.join(self.metafolder, "gsv_path.csv"))
        files = gsvpath["path"].values
        # Store file sizes in a dictionary
        file_sizes = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # The map method helps maintain the order of results
            results = executor.map(get_file_size, files)

            for file, size in results:
                if size is not None:
                    file_sizes[file] = size
                else:
                    file_sizes[file] = 0  # the file is removed

        gsvpath["size"] = gsvpath["path"].apply(lambda x: file_sizes[x])
        gsvpath.to_csv(os.path.join(self.metafolder, "gsv_path.csv"), index=False)


if __name__ == "__main__":
    main()
