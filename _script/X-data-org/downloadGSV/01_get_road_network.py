# import networkx as nx
# import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from fiona.crs import from_epsg
import os
from scipy import spatial
import shutil
from math import sin, cos, sqrt, atan2, radians
import random
import shapely
import networkx as nx
import osmnx as ox

boundfolder = "/lustre1/g/geog_pyloo/05_timemachine/GSV/bound"
metafolder = "/lustre1/g/geog_pyloo/05_timemachine/GSV"
GSVROOT = "/lustre1/g/geog_pyloo/05_timemachine/GSV/gsv_rgb"