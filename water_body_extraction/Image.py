from __future__ import division

import sys
import csv
csv.field_size_limit(sys.maxsize)

import tifffile as tiff
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import shapely.wkt
import shapely.affinity

import cv2

from sklearn import preprocessing

class Images:

    def __init__(self, id, dr):
        self.img_id = id
        self.dr = dr
        self.read_files()
    
    def read_files(self):
        img_rgb = tiff.imread('input/%s.tif'%self.img_id).transpose((1, 2, 0))

        (h, w, _) = img_rgb.shape
        ih = int(h / self.dr)
        iw = int(w / self.dr)

        self.imgs = {}

        self.imgs['rgb'] = cv2.resize(img_rgb, (iw, ih))
        #self.img_p = tiff.imread('input/%s_P.tif'%self.img_id)
        #self.img_a = tiff.imread('input/%s_A.tif'%self.img_id).transpose((1, 2, 0))
        self.imgs['m'] = cv2.resize(tiff.imread('input/%s_M.tif'%self.img_id).transpose((1, 2, 0)), (iw, ih))

        self.h = ih
        self.w = iw

    def preview(self):
        plt.imshow(cv2.convertScaleAbs(self.imgs['rgb'], alpha=(255.0/2048.0)))
    
    def mask_of(self, labels):
        label_masks = {}

        for _im_id, _x, _y in csv.reader(open('../input/grid_sizes.csv')):
            if _im_id == self.img_id:
                x_max, y_min = float(_x), float(_y)
                break      
        x_scaler = self.w * (self.w / (self.w+1)) / x_max
        y_scaler = self.h * (self.h / (self.h+1)) / y_min   
        
        for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):
            if _im_id == self.img_id and _poly_type in labels:
                p = shapely.affinity.scale(shapely.wkt.loads(_poly), xfact = x_scaler, yfact = y_scaler, origin = (0,0,0)) 
                mask = np.zeros((self.h, self.w), np.uint8)
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                exteriors = [int_coords(poly.exterior.coords) for poly in p]
                interiors = [int_coords(pi.coords) for poly in p for pi in poly.interiors]
                cv2.fillPoly(mask, exteriors, 1)
                cv2.fillPoly(mask, interiors, 0)
                label_masks[_poly_type] = mask
        
        combined_label_mask = np.zeros((self.h, self.w), np.uint8)
        for k in label_masks:
            combined_label_mask = np.bitwise_or(combined_label_mask, label_masks[k])
        
        plt.imshow(combined_label_mask)
        return combined_label_mask
    
    def apply_mask(self, img, band, mask, standardize = True):
        image = self.imgs[img][:, :, band-1]
        if standardize:
            image = preprocessing.scale(image)
        
        stacked = np.dstack((image, mask))
        pos = stacked[1 == stacked[:, :, 1]][:, 0]
        neg = stacked[0 == stacked[:, :, 1]][:, 0]

        plt.hist(pos, bins=200, label='Pos')
        plt.hist(neg, bins=200, histtype='stepfilled', color='r', alpha=0.5, label='Neg')
        plt.legend()
        plt.show()

        return (pos, neg)
    
    def density_slicing(self, img, band, mask, lower_limit, upper_limit, standardize = True):
        tp = tn = fp = fn = 0
        image = self.imgs[img][:, :, band-1]
        if standardize:
            image = preprocessing.scale(image)

        data = np.dstack((image, mask)).reshape(-1, 2)
        for d in data:
            dn = d[0]
            lb = d[1]
            if lower_limit >= dn or upper_limit <= dn:
                if lb == 0:
                    tn += 1
                else:
                    fn += 1
            else:
                if lb == 1:
                    tp += 1
                else:
                    fp += 1

        mask_preview = np.ones((self.h, self.w), np.uint8)
        upper_indecies = np.where(image >= upper_limit)
        lower_indecies = np.where(image <= lower_limit)
        mask_preview[upper_indecies] = 0
        mask_preview[lower_indecies] = 0

        plt.imshow(mask_preview)

        return (tp, tn, fp, fn)