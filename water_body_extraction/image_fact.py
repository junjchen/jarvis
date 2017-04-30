from __future__ import division
import sys
import csv
csv.field_size_limit(sys.maxsize)

import tifffile as tiff
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shapely.wkt
import shapely.affinity
import cv2

matplotlib.use('Agg')

from sklearn import preprocessing

import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
py.sign_in('junjchen', 'bdGkZpiRZ5nLdOCSD4o6') # Replace the username, and API key with your credentials.

class ImageFact:
    
    r = 4
    
    def __init__(self, img_id, labels):
        self.img_id = img_id
        self.labels = labels
        
        self._read_images()
        self._read_train_polygons()
        self._read_train_labels()
        self._read_training_data()
    
    def _read_images(self):
        img_rgb = tiff.imread('input/%s.tif'%self.img_id).transpose((1, 2, 0))                
        #img_p = tiff.imread('input/%s_P.tif'%self.img_id)        
        #img_a = tiff.imread('input/%s_A.tif'%self.img_id).transpose((1, 2, 0))
        img_m = tiff.imread('input/%s_M.tif'%self.img_id).transpose((1, 2, 0))
        
        (h, w, _) = img_rgb.shape
        self.h = int(h / ImageFact.r)
        self.w = int(w / ImageFact.r)
        
        #self.img_a = self._scale_image(img_a)        
        self.img_m = self._scale_image(img_m)
        self.img_rgb = self._scale_image(img_rgb)
        
        self.img_m_std = self._preprocess_image(self.img_m)

        # plot image
        plt.imshow(cv2.convertScaleAbs(self.img_rgb, alpha=(255.0/2048.0)))

    def _scale_image(self, img):
        (h, w, _) = img.shape
        scaled = cv2.resize(img, (self.w, self.h))
        return scaled
        
    def _preprocess_image(self, img):
        return preprocessing.scale(img)

    def _read_train_polygons(self):
        self.train_polygons = {}
        
        for _im_id, _x, _y in csv.reader(open('../input/grid_sizes.csv')):
            if _im_id == self.img_id:
                x_max, y_min = float(_x), float(_y)
                break      
        x_scaler = self.w * (self.w / (self.w+1)) / x_max
        y_scaler = self.h * (self.h / (self.h+1)) / y_min   
        
        for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):                
            if _im_id == self.img_id and _poly_type in self.labels:
                p = shapely.affinity.scale(shapely.wkt.loads(_poly), xfact = x_scaler, yfact = y_scaler, origin = (0,0,0)) 
                self.train_polygons[_poly_type] = p

    def _read_train_labels(self):
        self.img_labels = {}
        for k in self.train_polygons:
            train_polygon = self.train_polygons[k]
            img_label = np.zeros((self.h, self.w), np.uint8)
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exteriors = [int_coords(poly.exterior.coords) for poly in train_polygon]
            interiors = [int_coords(pi.coords) for poly in train_polygon
                         for pi in poly.interiors]
            cv2.fillPoly(img_label, exteriors, 1)
            cv2.fillPoly(img_label, interiors, 0)
            self.img_labels[k] = img_label
        
        self.combined_labels = np.zeros((self.h, self.w), np.uint8)
        for k in self.img_labels:
            self.combined_labels = np.bitwise_or(self.combined_labels, self.img_labels[k])
    
    def _gen_dn_stats_for_ds_table(self, dns):
        dn_mean = np.mean(dns, axis = 0)
        dn_std = np.std(dns, axis = 0)
        dn_max = np.max(dns, axis = 0)
        dn_min = np.min(dns, axis = 0)
        
        dn_mean_formatted = ['%1.2f'%dn for dn in dn_mean]
        dn_std_formatted = ['%1.2f'%dn for dn in dn_std]
        dn_max_formatted = ['%d'%dn for dn in dn_max]
        dn_min_formatted = ['%d'%dn for dn in dn_min]
        
        return (dn_mean_formatted, dn_std_formatted, dn_max_formatted, dn_min_formatted, dn_mean, dn_std, dn_max, dn_min)
    
    def _read_training_data(self):
        self.tr_img_m = np.dstack((self.img_m_std, self.combined_labels))
        self.tr_img_m_water_dn = self.tr_img_m[1 == self.tr_img_m[:,:,8]]
        self.tr_img_m_non_water_dn = self.tr_img_m[0 == self.tr_img_m[:,:,8]]
        
    def m_bands_stats(self):                
        water_dn_mean_formatted, water_dn_std_formatted, water_dn_max_formatted, water_dn_min_formatted, _, _, _, _ = self._gen_dn_stats_for_ds_table(self.tr_img_m_water_dn[:,0:8])
        water_dn_mean_formatted.insert(0, 'MEAN (W)')
        water_dn_std_formatted.insert(0, 'STD (W)')
        water_dn_max_formatted.insert(0, 'MAX (W)')
        water_dn_min_formatted.insert(0, 'MIN (W)')

        non_water_dn_mean_formatted, non_water_dn_std_formatted, non_water_dn_max_formatted, non_water_dn_min_formatted, _, _, _, _ = self._gen_dn_stats_for_ds_table(self.tr_img_m_non_water_dn[:,0:8])
        non_water_dn_mean_formatted.insert(0, 'MEAN (NW)')
        non_water_dn_std_formatted.insert(0, 'STD (NW)')
        non_water_dn_max_formatted.insert(0, 'MAX (NW)')
        non_water_dn_min_formatted.insert(0, 'MIN (NW)')
        
        colorscale = [[0, '#0D47A1'],[.5, '#64B5F6'],[1, '#ffffff']]
        table = ff.create_table([
            ['', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
            water_dn_mean_formatted,
            non_water_dn_mean_formatted,    
            water_dn_std_formatted,
            non_water_dn_std_formatted,    
            water_dn_max_formatted,
            non_water_dn_max_formatted,    
            water_dn_min_formatted,        
            non_water_dn_min_formatted
        ], colorscale=colorscale)

        py.image.ishow(table)
        
        
    def plot_histogram_of_band_in_img_m(self, b, i = False):
        start = np.min(self.tr_img_m_water_dn[:,b-1])
        end = np.mean(self.tr_img_m[:,:,b-1]) * 1.75
        data = [
            go.Histogram(
                x=self.tr_img_m_water_dn[:,b-1],
                opacity=0.75,
                name='water',
                autobinx=False,
                xbins=dict(
                    start=start,
                    end=end,
                    size=1
                )
            ),
            go.Histogram(
                x=self.tr_img_m_non_water_dn[:,b-1],
                opacity=0.5,
                name='not-water',
                autobinx=False,
                xbins=dict(
                    start=start,
                    end=end,
                    size=1
                )
            )
        ]
        layout = go.Layout(barmode='overlay', bargap=0, bargroupgap=0, title='Histogram for band ' + str(b))
        fig = go.Figure(data=data, layout=layout)

        if i:
            py.iplot(fig, filename=str(b))
        else:
            py.image.ishow(fig, width = 550, height = 400)
            
    def density_slicing(self, band, lower_limit, upper_limit):
        tp = tn = fp = fn = 0
        band_data = self.tr_img_m[:, :, [band-1, 8]].reshape(-1, 2)
        for d in band_data:
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
        self.analyze_result(tp, tn, fp, fn)
        return tp, tn, fp, fn
    
    def analyze_result(self, tp, tn, fp, fn):
        table = ff.create_table([
            ['', 'WATER (PREDICT)', 'NOT_WATER (PREDICT)', 'ACC'],
            ['WATER (TRUTH)', tp, fn, '%.2f%%'%(100 * tp / (tp + fn))],
            ['NOT_WATER (TRUTH)', fp, tn, '%.2f%%'%(100 * tn / (fp + tn))],
            ['REL', '%.2f%%'%(100 * tp / (tp + fp)), '%.2f%%'%(100 * tn / (fn + tn))]
        ], index_title='asdasd')
        py.image.ishow(table)
        print('Average accuracy: ' + '%.2f%%'%(100 * (tp / (tp + fn) + tn / (fp + tn)) / 2))
        print('Average reliability: ' + '%.2f%%'%(100 * (tp / (tp + fp) + tn / (fn + tn)) / 2))
        print('Overall accuracy: ' + '%.2f%%'%(100 * (tp + tn) / (tp + tn + fp + fn)))
        