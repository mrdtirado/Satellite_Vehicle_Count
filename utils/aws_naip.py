import boto3
import numpy as np
from io import BytesIO
import geopandas as gpd
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save
from shapely.geometry import Point, Polygon
import folium
import pyproj
import rasterio.plot
import rasterio as rio
import matplotlib.image as mpimg
import pandas as pd
import cv2
import glob
import json
import os
import pickle
import tempfile
import io
import fiona


def index_file_collection(manifest='/home/ubuntu/sat_collection/visualize_manifest.txt'):
    '''
    Input: Manifest: directory link to downloaded manifest of naip data source
    Output: Series of shp and associated files with directory flattedn into
            the name
    '''
    with open(manifest) as f:
        targets = [line for line in f if '/index/' in line]
        targets = [line.rstrip() for line in targets]
        target_names = [line.replace('/','_') for line in targets]
        # Let's use Amazon S3
        s3 = boto3.client('s3')

        for i in range(len(targets)):
            print('Working with file {}'.format(targets[i]))
            obj = s3.get_object(Bucket='naip-visualization',
                               Key=targets[i],
                               RequestPayer = 'requester')
            response_content = obj['Body'].read()
            with open('/home/ubuntu/sat_collection/index_files/{}'.format(target_names[i]),'wb') as file:
                file.write(response_content)

def image_crop(crs_value = None, image = None, lat = None, lon = None, name=None, epsg_name=None, given_name=None):
    '''
    INPUT: locations = directory location of costco coordinates
           files = list of file names, and its epsg number
           output_loc = directory to save cropped tiffs
    OUTPUT: cropped tif files in a surrounding 500 x 500 pixel square around
            the coordinate location
    '''

    N=500

    image_crs = image.crs
    # transform coordinates
    lon_, lat_ = pyproj.transform(list(crs_value.values())[0], str(image_crs),lat, lon)

    # Get pixel coordinates from map coordinates
    y, x = image.index(lon_, lat_)
    print('Pixel Y, X coords: {}, {}'.format(y, x))

    # Build an NxN window
    window = rio.windows.Window(x - N//2, y - N//2, N, N)
    print(window)

    # Read the data in the window
    # clip is a nbands * N * N numpy array
    clip = image.read(window=window)

    # You can then write out a new file
    meta = image.meta
    meta['width'], meta['height'] = N, N
    meta['transform'] = rio.windows.transform(window, image.transform)

    with rio.open('/home/ubuntu/sat_collection/cropped_images/{}'.format(given_name), 'w', **meta) as dst:
        dst.write(clip)


def files_from_shp(shapes = None, locations = None):
    '''
    INPUT: shapes: Directory of shape files with their associated files
           locations: directory of costco locations .txt file
    OUTPUT: List of tifs and their associated crs value
    '''
    # read necessary files in

    shape_files = glob.glob('/home/ubuntu/sat_collection/index_files/*.shp')
    costco_raw = pd.read_csv('/home/ubuntu/sat_collection/costcolocations.txt', delimiter ='\t')

    lat = costco_raw['lat']
    lon = costco_raw['lon']

    # this order is correct for NAIP polygons should be in lonlat format
    lonlat = list(zip(lon,lat))
    lonlat_points = [Point(line) for line in lonlat]

    files_found = []
    error_files = []
    # loop through all the shape files
    for i in range(len(shape_files)):
        try:
            df = gpd.read_file(shape_files[i])

            # loop through all the images in each shape file
            for j in range(len(df)):

                # loop through all the coordinates for costco
                for k in range(len(lonlat_points)):

                    # check to see if the coordinates are within a specific image
                    if (df['geometry'][j]).contains(lonlat_points[k])==True:
                        lon_, lat_ = lonlat[k]

                        print(df['FileName'][j])
                        files_found.append(([df['FileName'][j], df.crs, lon_, lat_]))
                        print([df['FileName'][j], df.crs, lon_, lat_])
                        print('Found')


        except:
            error_files.append(shape_files[i])
            continue
    return files_found

def find_file(filename = None, manifest_directory = None):
    '''

    '''
    manifest_directory = '/home/ubuntu/sat_collection/visualize_manifest.txt'
    with open(manifest_directory) as f:

        # find the filename using the first three identifiers for region
        targets = [line for line in f if '_'.join(filename.split('_')[0:3]) in line]
        # strip the line space
        targets = [line.rstrip() for line in targets]

        # find only the .tif files
        targets = [line for line in targets if '.tif' in line]
    return targets
