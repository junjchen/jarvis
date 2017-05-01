from __future__ import division
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import numpy as np

def show_analyze_result(stats):
    tp, tn, fp, fn = stats
    df = pd.DataFrame()
   
    col1 = ['POSITIVE (TRUTH)', 'NEGATIVE (TRUTH)', 'REL']
    col2 = [tp, fp, '-' if tp == 0 and fp == 0 else'%.2f%%'%(100 * tp / (tp + fp))]
    col3 = [fn, tn, '-' if fn == 0 and tn == 0 else'%.2f%%'%(100 * tn / (fn + tn))]
    col4 = ['-' if tp == 0 and fn == 0 else '%.2f%%'%(100 * tp / (tp + fn)), '-' if fp == 0 and tn == 0 else '%.2f%%'%(100 * tn / (fp + tn)), '-']
    
    df['-'] = col1
    df['POSITIVE (PREDICT)'] = col2
    df['NEGATIVE (PREDICT)'] = col3
    df['ACC'] = col4
    
    print(df)

    print('========================================================')
    print('Average accuracy: ' + ('-' if tp == 0 and fn == 0 or fp == 0 and tn == 0 else '%.2f%%'%(100 * (tp / (tp + fn) + tn / (fp + tn)) / 2)))
    print('Average reliability: ' + ('-' if tp == 0 and fp == 0 or fn == 0 and tn == 0 else '%.2f%%'%(100 * (tp / (tp + fp) + tn / (fn + tn)) / 2)))
    print('Overall accuracy: ' + '%.2f%%'%(100 * (tp + tn) / (tp + tn + fp + fn)))

def decision_tree_result(image, predict, labels):
    h = image.h
    w = image.w

    plt.title('Truth')
    plt.imshow(labels.reshape(h, w))
    plt.figure()
    plt.title('Predict')
    plt.imshow(predict.reshape(h, w))

    predict = predict.flatten()
    labels = labels.flatten()

    compare = np.vstack((predict, labels)).transpose()
    tp = len(np.where(np.logical_and(compare[:, 0] == 1, compare[:, 1] == 1))[0])
    tn = len(np.where(np.logical_and(compare[:, 0] == 0, compare[:, 1] == 0))[0])
    fp = len(np.where(np.logical_and(compare[:, 0] == 1, compare[:, 1] == 0))[0])
    fn = len(np.where(np.logical_and(compare[:, 0] == 0, compare[:, 1] == 1))[0])

    show_analyze_result((tp, tn, fp, fn))

    return (tp, tn, fp, fn)

