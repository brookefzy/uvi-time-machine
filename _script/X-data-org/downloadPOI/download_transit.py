"""Get data from overtur API and save it to a file."""

import osmnx as ox
import geopandas as gpd
import os
import numpy as np

EXPORTFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_raw/r_pois/transit"
if not os.path.exists(EXPORTFOLDER):
    os.makedirs(EXPORTFOLDER)
city_bound_folder = "/lustre1/g/geog_pyloo/05_timemachine/_raw/r_boundary_osm"
bound_files = os.listdir(city_bound_folder)


def download_transit_stations_osmnx(polygon):

    # Step 2: Use osmnx to download points of interest (POIs) for railway, subway, and bus stations
    tags = {'railway': ["station", "subway_entrance", "tram_stop", "halt"], 
            'subway': ['yes'], 
            "tram": ["yes"],
            "bus": ["yes"],
            "public_transport": ["station", "platform", "stop_position"],
            'amenity': [
            'bus_station', 
            'bus_stop',
            'bus_terminal', 
            'ferry_terminal',
            'subway_entrance',
            'train_station',
            'light_rail_station',
            'tram_stop',
            'tram_station',
            'subway_station',
            'rail_station',
            'railway_station',
            'train_stop',
            ]}

    # osmnx will fetch the nodes from OSM within the given polygon
    transit_stations = ox.features_from_polygon(polygon, tags)

    return transit_stations


def download_city(bound_sample):
    geo_sample = gpd.read_file(os.path.join(city_bound_folder, bound_sample))
    polygon = geo_sample.to_crs(epsg=4326).geometry[0].convex_hull
    print(geo_sample.crs)
    col_keep = ["osmid","element_type","name", "type", "geometry","amenity",'railway','subway','tram','bus','public_transport']
    stations = download_transit_stations_osmnx(polygon)
    # if a field is a list, convert to a long string
    col_update = [x for x in col_keep if x in stations.columns]
    stations = stations[col_update]
    for col in stations.columns:
        if stations[col].dtype == "object":
            stations[col] = stations[col].apply(
                lambda x: ", ".join([str(t) for t in x]) if isinstance(x, list) else x
            )
    stations.to_file(os.path.join(EXPORTFOLDER, bound_sample), driver="GeoJSON")
    print(bound_sample)


from tqdm import tqdm

city_to_avoid = ["London", "Sao Paulo", "New York", "Hong Kong", "Copenhagen"]
bound_to_avoid = [x.lower().replace(" ","")+ ".geojson" for x in city_to_avoid]
# for bound_sample in tqdm(bound_files):
#     if bound_sample in bound_to_avoid:
#         continue
for bound_sample in tqdm(["copenhagen.geojson"]):
    # else:
        try:
            download_city(bound_sample)
            print(bound_sample, "done")
        except Exception as e:
            print(bound_sample, "failed", e)
            pass
