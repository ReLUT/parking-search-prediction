import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import contextily as ctx

def format_duration(seconds, unit='sec'):
    """
    Display time duration in HH:MM:SS or MM:SS format.
    
    Parameters:
    - seconds: The time in seconds or minutes.
    - unit: The unit of the input time ('sec' for seconds, 'min' for minutes).

    Returns:
    - A string representation of the time in either HH:MM:SS or MM:SS format.
    """
    
    if unit == 'min':
        seconds *= 60
    
    if np.isnan(seconds):
        return np.nan
    
    hours = 0
    if seconds >= 60*60:
        hours = seconds // (60*60)
        seconds %= (60*60)
    
    minutes = seconds // 60
    seconds %= 60
    
    if hours:
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):2}"
    else:
        return f"{int(minutes):02}:{int(seconds):02}"

def plot_gps_route(ax,
                   GPS_df,
                   trip_ID=None,
                   col_ID='TripID',
                   label='y_hat_labels',
                   n_driving_points=30,
                   col_lat='lat',
                   col_lon='lon',
                   title='',
                   basemap=True,
                   ax_legend=True):
    
    if trip_ID is not None:
        trip_test = GPS_df[GPS_df[col_ID]==trip_ID].reset_index(drop=True).reset_index()
    else:
        trip_test = GPS_df


    gdf=gpd.GeoDataFrame(data=trip_test,
                         geometry=trip_test[[col_lat, col_lon]]\
                             .apply(lambda x: Point(x[col_lon], x[col_lat]), axis=1).values,
                         crs='EPSG:4326')

    ## If more than one search point predicted in the labels then we can visualize the search
    if (gdf[f'{label}']=='searching').sum()>1:

        idx_drive_max = gdf[gdf[label]=='driving'].iloc[-1].name

        ## route points
        gdf[gdf[f'{label}']=='driving'][idx_drive_max-n_driving_points:]\
                .to_crs('EPSG:3857').plot(ax=ax, color='blue', label='Driving',zorder=2)
        gdf[gdf[f'{label}']=='searching'].to_crs('EPSG:3857').plot(ax=ax, color='tab:red', label='Searching')
        gdf[gdf[f'{label}']=='walking'].to_crs('EPSG:3857').plot(ax=ax, color='tab:green', label='Walking')
        
        
        ## route lines
        begin_loc_search = gdf[gdf[f'{label}']=='searching'].iloc[0].name
        gpd.GeoDataFrame(geometry=[LineString(gdf[idx_drive_max-n_driving_points:begin_loc_search+1]['geometry'].values)],
                         crs='epsg:4326').to_crs('EPSG:3857').plot(ax=ax, color='tab:blue')
        gpd.GeoDataFrame(geometry=[LineString(gdf[gdf[f'{label}']=='searching']['geometry'].values)],
                         crs='epsg:4326').to_crs('EPSG:3857').plot(ax=ax, color='tab:red')
        gpd.GeoDataFrame(geometry=[LineString(gdf[gdf[f'{label}']=='walking']['geometry'].values)],
                         crs='epsg:4326').to_crs('EPSG:3857').plot(ax=ax, color='tab:green')
        ## main points
        # gdf.iloc[[0]].to_crs('EPSG:3857').plot(color='blue',edgecolor='black', markersize=200, ax=ax,lw=3, label='Origin')
        gdf[gdf[f'{label}']=='searching'].iloc[[0]].to_crs('EPSG:3857').plot(ax=ax, edgecolor='black',lw=3, zorder=2,
                                                       markersize=350,color='red', label='Search Starting Point')


        psd_tmp = gdf[gdf[label] == "searching"]['remainingTime'].iloc[0]
            
    else:
        
        gdf[-n_driving_points:]\
                .to_crs('EPSG:3857').plot(ax=ax, color='blue', label='Driving',zorder=2)
        gpd.GeoDataFrame(geometry=[LineString(gdf[-n_driving_points:]['geometry'].values)],
                         crs='epsg:4326').to_crs('EPSG:3857').plot(ax=ax, color='tab:blue')
        psd_tmp=0
    

    gdf[gdf[f'{label}']=='searching'].iloc[[-1]].to_crs('EPSG:3857').plot(ax=ax, edgecolor='black',lw=3, zorder=2,
                                                        label='Parking Spot',markersize=350,color='tab:orange')
        
    gdf.iloc[[-1]].to_crs('EPSG:3857').plot(ax=ax, edgecolor='black',lw=3, zorder=2,
                                                        label='Final Destination',markersize=350,color='tab:green')
        
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    ax.set_xlim([xlim[0]-10,xlim[1]+10])
    ax.set_ylim([ylim[0],ylim[1]+10])
    ax.set_axis_off()
    if ax_legend:
        ax.legend()

    ax.set_title(f'{title}\nParking Search Duration: {format_duration(psd_tmp)}')
    
    if basemap:
        ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)
        #ctx.add_basemap(ax,source=ctx.providers.CartoDB.Voyager)
        #ctx.add_basemap(ax,source=ctx.providers.OpenStreetMap.Mapnik)

import folium

def visualize_trip(df, tripID=None, label='label', col_ID='TripID'):
    
    if tripID!=None:     
        # Filter the dataframe for the specific tripID
        trip_data = df[df[col_ID] == tripID]
    else:
        trip_data=df

    # Colors based on label
    color_map = {
        'driving': 'blue',
        'searching': 'red',
        'walking': 'green'
    }

    # Use the last point as the center
    last_point = trip_data.iloc[-1]
    m = folium.Map(location=[last_point['lat'], last_point['lon']], zoom_start=15)

    # Iterate through the rows in trip_data and add markers to the map
    for _, row in trip_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,  # Adjust this as needed to change the size of the circle markers
            popup=f"Speed: {row['speed_kmh']}",
            color=color_map[row[label]],
            fill=True,
            fill_color=color_map[row[label]],
            fill_opacity=1.0
        ).add_to(m)

    return m