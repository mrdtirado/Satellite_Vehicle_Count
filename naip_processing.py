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


def main(pickle_file ='/home/ubuntu/list.pkl'):
    '''
    USEAGE: Point it at the pickle file and direct an output directory in imagecrop.py
            helper function.
    INPUT: Pickle or values from files_from_shape.
    OUTPUT: Series of cropped images in 500 x 500 pixel radius of
    '''
    # load the pickle created by files from shp
    saved = pickle.load(open(pickle_file, 'rb'))
    saved = np.array(saved)

    # break up the individual values
    filename = np.array(saved)[:,0]
    epsg = np.array(saved)[:,1]
    lon_values = np.array(saved)[:,2]
    lat_values = np.array(saved)[:,3]


    # loop over all the names of files
    for i in range(len(filename) -1):

        # find the file across multiple years for that segment
        found_files = find_file(filename[i])

        # create new names for cropped filed
        flat_name = ['_'.join(line.split('/')) for line in found_files]

        # loop over the found files
        for j in range(len(found_files)):
            test_name = ['_'.join(line.split('/')) for line in found_files]

            #work with aws
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket='naip-visualization',
                               Key=found_files[j],
                               RequestPayer = 'requester')

            # write an image as a temporary file
            tmp = tempfile.NamedTemporaryFile()

            response_content = obj['Body'].read()
            with open(tmp.name, 'wb') as f:
                f.write(response_content)

            # open the temporary file with rasterio
            image = rio.open(tmp.name)

            # crop the image along with this meta data and save the image in the directory
            # specificed in the helper function
            image_crop(image=image,
                       lat = lat_values[i],
                       lon=lon_values[i],
                       given_name=flat_name[j],
                       crs_value=epsg[i])


if __name__ == '__main__':
    main()
