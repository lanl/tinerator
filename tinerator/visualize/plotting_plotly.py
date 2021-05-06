"""
A Plotly backend for visualization.
"""

import shapely
import pandas as pd
from colorcet import fire
import datashader.transfer_functions as tf
import xarray as xr
import numpy as np
import plotly.express as px
from ..constants import PLOTLY_PROJECTION
from ..gis import reproject_raster

def plotly_raster_layer(layer):
    raster_layer = reproject_raster(layer, PLOTLY_PROJECTION)
    xmin, ymin, xmax, ymax = raster_layer.extent

    coordinates = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ]

    center = np.mean(coordinates, axis=0)
    df = pd.DataFrame(data={'Lon': [center[0]], 'Lat': [center[1]]})

    xs = np.linspace(xmin, xmax, raster_layer.shape[1])
    ys = np.linspace(ymin, ymax, raster_layer.shape[0])

    da = xr.DataArray(raster_layer.masked_data(), coords=[('y', ys), ('x', xs)])
    img = tf.shade(da, cmap=fire)[::-1].to_pil()

    fig = px.scatter_mapbox(df, lat='Lat', lon='Lon', zoom=10)
    fig.update_layout(mapbox_style="carto-darkmatter",
                     mapbox_layers = [
                    {
                        "sourcetype": "image",
                        "source": img,
                        "coordinates": coordinates
                    }]
    )
    fig.show()

def plotly_polygon_layer(layer):
    return None

def plotly_scatter_layer(layer):
    return None
    #fig = px.scatter_mapbox(df_sample,
    #lat=“lat”,
    #lon=“lng”,
    #hover_name=“id”,
    #color=“company”,
    #hover_data=[“id”],
    #zoom=10.5,
    #height=500)

def plotly_linestring_layer(layer):
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    else:
        warn(f"Unknown shape {type(feature)}")
        
    for linestring in linestrings:
        x, y = linestring.xy
        lats = np.append(lats, y)
        lons = np.append(lons, x)
        names = np.append(names, [name]*len(y))
        lats = np.append(lats, None)
        lons = np.append(lons, None)
        names = np.append(names, None)

    return px.line_mapbox(lat=lats, lon=lons, hover_name=names, zoom=1)
    

def plotly_plot(layers, mapbox_style="stamen-terrain"):
    """
    
    * "white-bg" yields an empty white canvas which results in no external HTTP requests
    * "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", 
      "stamen-toner" or "stamen-watercolor" yield maps composed of raster tiles 
      from various public tile servers.
    * "usgs-imagery" yields maps composed of USGS satellite images.
    
    mapbox_style (:obj:`str`, optional): Any of Plotly supported Mapbox styles.
        ``["white-bg", "stamen-terrain", "carto-darkmatter"]``
    """
    # https://community.plotly.com/t/multiple-mapbox-figures-in-one-map/31735/2
    if not isinstance(layers, list):
        layers = [layers]
    
    figs = []
    for layer in layers:
        feature = reproject_geometry(layer, PLOTLY_PROJECTION)
        figs.append(plotly_linestring_layer(feature))
    
    fig = figs[0]
    
    if len(figs) > 1:
        for f in figs[1:]:
            fig.add_trace(f.data[0])
    
    if mapbox_style == "usgs-imagery":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                # USGS tile maps
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                    ]
                },
                {
                    "sourcetype": "raster",
                    "sourceattribution": "Government of Canada",
                    "source": ["https://geo.weather.gc.ca/geomet/?"
                               "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
                               "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"],
                }
              ]
        )
    else:
        fig.update_layout(mapbox_style=mapbox_style)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()