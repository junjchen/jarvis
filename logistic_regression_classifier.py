## after https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

import pandas as pd


# imports also in read_image

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

# get the right values from openimage
import read_image

#number of input files
N_Cls = 10
#location of input files
inDir = 'input'

def setup(im_rgb, train_mask):
    #xs is the training input
    # xs is created by taking the x by y by 3 colours image and reshaping it to one big three colour array
    print()
    xs = im_rgb.reshape(-1, 3).astype(np.float32)

    # ys is the training output, reshaped in one big array
    ys = train_mask.reshape(-1)

    # a pipeline is created using sklearn
    pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))

    return [xs, ys, pipeline]

def train(xs, ys, pipeline, train_mask):
    print('training...')
    # do not care about overfitting here
    #Use the pipeline to fit the inputs to the outputs
    pipeline.fit(xs, ys)

    #predict outputs from the trained pipeline and inputs
    pred_ys = pipeline.predict_proba(xs)[:, 1]

    #see how close the outputs are to the actual values
    print('average precision', average_precision_score(ys, pred_ys))

    #Shape the outputs to be the training mask shape
    pred_mask = pred_ys.reshape(train_mask.shape)

    #show the mask that was trained on
    #read_image.show_mask(pred_mask[2900:3200,2000:2300])
    read_image.show_mask(pred_mask)

    #Cut off predicted values lower than 0.3
    threshold = 0.3
    pred_binary_mask = pred_mask >= threshold
    #show the binary predicted mask
    #read_image.show_mask(pred_binary_mask[2900:3200,2000:2300])
    return pred_binary_mask



def mask_to_polygons(mask, epsilon=10., min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

# def stick_all_train():
#     print("let's stick all imgs together")
#     s = 835
#
#     x = np.zeros((5 * s, 5 * s, 8))
#     y = np.zeros((5 * s, 5 * s, N_Cls))
#
#     df = pd.read_csv(inDir + '/train_wkt_v4.csv')
#     ids = sorted(df.ImageId.unique())
#     print(len(ids))
#     for i in range(5):
#         for j in range(5):
#             id = ids[5 * i + j]
#
#             [im_rgb, train_mask, im_size, x_scaler, y_scaler, train_polygons] = read_image.read_image(id, '5')
#
#             img = stretch_n(im_rgb)
#             print(img.shape, id, np.amax(img), np.amin(img))
#             x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
#             for z in range(N_Cls):
#                 y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
#                     (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]
#
#     print(np.amax(y), np.amin(y))
#
#     np.save('data/x_trn_%d' % N_Cls, x)
#     np.save('data/y_trn_%d' % N_Cls, y)
#
# def stretch_n(bands, lower_percent=0, higher_percent=100):
#     out = np.zeros_like(bands, dtype=np.float32)
#     n = bands.shape[2]
#     for i in range(n):
#         a = 0  # np.min(band)
#         b = 1  # np.max(band)
#         c = np.percentile(bands[:, :, i], lower_percent)
#         d = np.percentile(bands[:, :, i], higher_percent)
#         t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
#         t[t < a] = a
#         t[t > b] = b
#         out[:, :, i] = t
#
#     return out.astype(np.float32)