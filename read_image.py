## after https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
from collections import defaultdict
import csv
import sys

# cv2 might give an error in the editor, but should still compile correctly regardless
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

# Needed for proper Tifffile plotting using imshow, even if editor says it is not used
from matplotlib import pyplot

#this shows the image after plotting something
#pyplot.interactive(True)

import os


def read_image(IM_ID = '6120_2_2', POLY_TYPE = '1'):
    ##
    # Fix for overflowerror from sys.maxsize
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    #csv.field_size_limit(sys.maxsize);
    ##


    os.chdir('input')
    #print(os.listdir())

    # Load grid size
    x_max = y_min = None
    for _im_id, _x, _y in csv.reader(open('../input/grid_sizes.csv')):
        if _im_id == IM_ID:
            x_max, y_min = float(_x), float(_y)
            break

    # Load train poly with shapely
    train_polygons = None
    for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):
        if _im_id == IM_ID and _poly_type == POLY_TYPE:
            print('polygons selected successfully')
            train_polygons = shapely.wkt.loads(_poly)
            break

    # Read image with tiff

    ###im_rgb = tiff.imread('../input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
    ###im_rgb = tiff.imread('../input/three_band/6010_0_0.tif'.format(IM_ID)).transpose([1, 2, 0])
    #data on martins pc
    im_rgb = tiff.imread(('C:/Users/Martin/Documents/Southampton/Advanced Machine Learning/data/three_band/three_band/'+IM_ID+'.tif').format(IM_ID)).transpose([1, 2, 0])
    im_size = im_rgb.shape[:2]


    x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)

    train_polygons_scaled = None
    if(train_polygons!=None):
        train_polygons_scaled = shapely.affinity.scale( train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    else:
        print('no polygons here')



    train_mask = mask_for_polygons(train_polygons_scaled, im_size)

    tiff.imshow(255 * scale_percentile(im_rgb[2900:3200, 2000:2300]));

    show_mask(train_mask[2900:3200,2000:2300])

    #this actually shows the image if not interactive
    pyplot.show()

    return [im_rgb, train_mask, im_size, x_scaler, y_scaler, train_polygons]


def get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def mask_for_polygons(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def show_mask(m):
    # hack for nice display
    tiff.imshow(255 * np.stack([m, m, m]));
    pyplot.show(block=True)


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix
