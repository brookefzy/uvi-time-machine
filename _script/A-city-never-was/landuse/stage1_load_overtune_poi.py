# load polygon from file
import geopandas as gpd
from overturemaps import core
import os
import pandas as pd

EXPORT_FOLDER = (
    "/Users/yuan/Dropbox (Personal)/Personal Work/_commondata/POI/overtunemaps/raw"
)
SHAPE_FOLDER = "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2023/01_city-never-was/_data/_raw/r_boundary_osm"


def get_all_places(city_lower: str) -> gpd.GeoDataFrame:
    boundary_gdf = gpd.read_file(os.path.join(SHAPE_FOLDER, f"{city_lower}.geojson"))
    bbox = boundary_gdf.total_bounds  # (minx, miny, maxx, maxy)
    places_df = core.geodataframe("place", bbox=(bbox[0], bbox[1], bbox[2], bbox[3]))
    print(f"Total records for {city_lower}:", len(places_df))
    places_df.crs = "EPSG:4326"
    places_df.to_parquet(
        os.path.join(EXPORT_FOLDER, f"{city_lower}.parquet"),
        index=False,
    )
    return places_df


def check_finished(city_lower: str) -> bool:
    file_path = os.path.join(EXPORT_FOLDER, f"{city_lower}.parquet")
    return os.path.exists(file_path)


def main():
    cityls = os.listdir(SHAPE_FOLDER)
    valid_cityls = [x for x in cityls if x.endswith(".geojson")]
    cityls = [x.replace(".geojson", "") for x in valid_cityls]

    print("Cities to process:", cityls)
    for city_lower in cityls:
        print("=" * 50)
        print(f"Processing city: {city_lower}")
        print("=" * 50)
        if not check_finished(city_lower):
            get_all_places(city_lower)
        else:
            print(f"{city_lower} already finished.")


if __name__ == "__main__":
    main()
