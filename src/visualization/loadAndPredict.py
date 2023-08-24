import sys
sys.path.append('../neural_network')
from predict import *
from fileCheck import validate_csv
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

file_path = "../../examples/trajectory_1.csv"
validate_csv(file_path, verbose=True)
df = pd.read_csv(file_path)
df = park_search_predict(df, model_path='../../model/ParkingSearchPrediction.h5')
gdf = gpd.GeoDataFrame(data=df,
                       geometry=[Point(p) for p in df[['lon', 'lat']].values],
                       crs='EPSG:4326')
gdf.to_file("output.geojson", driver="GeoJSON")
