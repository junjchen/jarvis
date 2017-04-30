from __future__ import division
import pandas as pd

def show_analyze_result(stats):
    tp, tn, fp, fn = stats
    df = pd.DataFrame()
   
    col1 = ['POSITIVE (TRUTH)', 'NEGATIVE (TRUTH)', 'REL']
    col2 = [tp, fp, '%.2f%%'%(100 * tp / (tp + fp))]
    col3 = [fn, tn, '%.2f%%'%(100 * tn / (fn + tn))]
    col4 = ['%.2f%%'%(100 * tp / (tp + fn)), '%.2f%%'%(100 * tn / (fp + tn)), '-']
    
    df['-'] = col1
    df['POSITIVE (PREDICT)'] = col2
    df['NEGATIVE (PREDICT)'] = col3
    df['ACC'] = col4
    
    print(df)

    print('========================================================')
    print('Average accuracy: ' + '%.2f%%'%(100 * (tp / (tp + fn) + tn / (fp + tn)) / 2))
    print('Average reliability: ' + '%.2f%%'%(100 * (tp / (tp + fp) + tn / (fn + tn)) / 2))
    print('Overall accuracy: ' + '%.2f%%'%(100 * (tp + tn) / (tp + tn + fp + fn)))