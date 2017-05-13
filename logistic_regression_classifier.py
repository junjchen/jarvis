## after https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

import pandas as pd

import random

# imports also in read_image

from collections import defaultdict
import csv
import sys

# cv2 might give an error in the editor, but should still compile correctly regardless
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt

from shapely.wkt import loads as wkt_loads
import shapely.affinity
import numpy as np
import tifffile as tiff

# get the right values from openimage
import read_image

#number of input files
N_Cls = 10

ISZ = 160

#location of input files
inDir = 'C:/Users/Martin/Documents/Southampton/Advanced Machine Learning/data/'
DF = pd.read_csv(inDir+'train_wkt_v4.csv')
GS = pd.read_csv(inDir+'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(inDir+'sample_submission.csv')

#set rgb to 1 for rgb, 0 for M
rgb = 1

#for a one image setup pipeline
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

#all images
def setup():
    CLASSES = {
        1: 'Bldg',
        2: 'Struct',
        3: 'Road',
        4: 'Track',
        5: 'Trees',
        6: 'Crops',
        7: 'Fast H20',
        8: 'Slow H20',
        9: 'Truck',
        10: 'Car',
    }
    #get inputs:

    #stick_all_train()
    print('loading inputs')
    #validation in and output
    #x_val, y_val = np.load('data/x_tmp_%d.npy' % N_Cls), np.load('data/y_tmp_%d.npy' % N_Cls)

    #train input
    img = np.load('data/x_trn_%d.npy' % N_Cls)
    #train output
    msk = np.load('data/y_trn_%d.npy' % N_Cls)

    print('just reshaping now')
    #reshape the loaded input for the learner
    if rgb:
        xs = img.reshape(-1, 3).astype(np.float32)
    else:
        xs = img.reshape(-1, 8).astype(np.float32)
    print('inputs loaded')
    mskdict = {}
    ysdict = {}
    pipelinedict = {}
    print(xs.size)
    print(round(xs.size*0.8/3))
    print(xs[round(xs.size*0.8/3):].shape)
    print(xs[round(xs.size*0.8):].shape)
    for classCounter in range(0,N_Cls):
        print('starting: ', classCounter)

        mskdict[classCounter] = msk[:,:,classCounter]
        ysdict[classCounter] = mskdict[classCounter].reshape(-1)
        #ys = msk.reshape(-1)


        #print("inputs size img: %2, msk: %t") % (str(np.shape(img)), str(np.shape(msk)))
        #print("inputs size xs: %2, ys: %t") % (str(np.shape(xs)), str(np.shape(ys)))
        #print(np.shape(img))
        #print(np.shape(msk))
        #print(np.shape(mskbuilding))
        #print(np.shape(xs))
        #print(np.shape(ysbuilding))

        print()
        # a pipeline is created using sklearn
        pipelinedict[classCounter] = make_pipeline(StandardScaler(), SGDClassifier(loss='log',  n_iter=5))

        #unse inputs to train lrc
        print('training...')
        # do not care about overfitting here
        # Use the pipeline to fit the inputs to the outputs
        round(xs.size*0.8)
        pipelinedict[classCounter].fit(xs[1:round(xs.size*0.7/3)], ysdict[classCounter][1:round(xs.size*0.7/3)])

        print('training finished')

        # predict outputs from the trained pipeline and inputs
        #pred_ys = pipelinedict[classCounter].predict_proba(xs)[:, 1]

        # see how close the outputs are to the actual values
        #print('average precision', average_precision_score(ysdict[classCounter], pred_ys))

    #quick test
    pred_ys = {}

    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for classCounter in range(0, N_Cls):
        pred_ys[classCounter] = pipelinedict[classCounter].predict_proba(xs)[:, 1]
        pred_ys[classCounter] = pred_ys[classCounter] > trs[classCounter]
        print(ysdict[classCounter].shape)
        print(ysdict[classCounter].shape)
        print(ysdict[classCounter])
        #17430625
        print(ysdict[classCounter][round(xs.size*0.8/3):xs.size-1])
        print(pred_ys[classCounter][round(xs.size * 0.8/3):xs.size - 1])
        print('average precision', classCounter, ' ' , average_precision_score(ysdict[classCounter][round(xs.size*0.7/3):], pred_ys[classCounter][round(xs.size*0.7/3):]))

    #real test
    #predict_test(pipelinedict)
    #print('creating submission file')
    #make_submit()
    # Shape the outputs to be the training mask shape
    #pred_mask = pred_ys.reshape(train_mask.shape)

    # show the mask that was trained on
    # read_image.show_mask(pred_mask[2900:3200,2000:2300])
    #read_image.show_mask(pred_mask)

    # Cut off predicted values lower than 0.3
    #threshold = 0.3
    #pred_binary_mask = pred_mask >= threshold
    # show the binary predicted mask
    # read_image.show_mask(pred_binary_mask[2900:3200,2000:2300])
    #return pred_binary_mask
    return

def make_train_and_val():
    #only do the two statements beneath once
    #make one big happy training set
    stick_all_train()
    #make a validation set
    make_val()

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


########based fullimp below this comment
def stick_all_train():
    print("let's stick all imgs together")
    s = 835

    if rgb:
        x = np.zeros((5 * s, 5 * s, 3))
    else:
        x = np.zeros((5 * s, 5 * s, 8))
    #y = np.zeros((5 * s, 5 * s, N_Cls))
    y = np.zeros((5 * s, 5 * s, N_Cls))


    ids = sorted(DF.ImageId.unique())
    print(len(ids))
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]

            if rgb:
                img = rgb(id)
            else:
                img = M(id)
            img = stretch_n(img)
            print(img.shape, id, np.amax(img), np.amin(img))
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(N_Cls):
                #y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                #    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]

    print(np.amax(y), np.amin(y))

    np.save('data/x_trn_%d' % N_Cls, x)
    np.save('data/y_trn_%d' % N_Cls, y)

def rgb(image_id):
    # __author__ = amaia
    # based on M
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    img = tiff.imread(inDir + 'three_band/three_band/{}.tif'.format(image_id))

    # img = tiff.imread(inDir + 'sixteen_band/sixteen_band/6010_0_0_M.tif'.format(image_id))

    img = np.rollaxis(img, 0, 3)
    return img

def M(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    img = tiff.imread(inDir+'sixteen_band/sixteen_band/{}_M.tif'.format(image_id))

    #img = tiff.imread(inDir + 'sixteen_band/sixteen_band/6010_0_0_M.tif'.format(image_id))

    img = np.rollaxis(img, 0, 3)
    return img

def stretch_n(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands, dtype=np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)

def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask

def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)

def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list

def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask

def make_val():
    print("let's pick some samples for validation")
    img = np.load('data/x_trn_%d.npy' % N_Cls)
    msk = np.load('data/y_trn_%d.npy' % N_Cls)
    x, y = get_patches(img, msk, amt=3000)

    np.save('data/x_tmp_%d' % N_Cls, x)
    np.save('data/y_tmp_%d' % N_Cls, y)

def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * ISZ)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(N_Cls):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    return x, y

def predict_test(pipeline, trs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
    print("predict test")
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, pipeline, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0:
            print(i, id)

        #just do the first one using break when testing for bugs
        #break

def predict_id(id, pipelinedict, trs):
    #rewritten for lrc
    if rgb:
        img = rgb(id)

        xs = img.reshape(-1, 3).astype(np.float32)
    else:
        img = M(id)

        xs = img.reshape(-1, 8).astype(np.float32)
    pred_ys = {}

    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for classCounter in range(0, N_Cls):
        pred_ys[classCounter] = pipelinedict[classCounter].predict_proba(xs)[:, 1]
        pred_ys[classCounter] = pred_ys[classCounter] > trs[classCounter]
        pred_ys[classCounter].reshape(img.shape[0:2])


    return pred_ys

def make_submit():
    print("make submission file")
    df = pd.read_csv(inDir+'sample_submission.csv')
    print(df.head())
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print(idx)
    print(df.head())
    df.to_csv('subm/lrc.csv', index=False)

def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min