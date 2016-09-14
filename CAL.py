# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:18:48 2016

@author: d_floriello

CAL analysis
"""

import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt 

cal = pd.read_excel('CAL.xlsx', sheetname = 'valori CAL')
cal = cal.fillna(0)

monthwise = OrderedDict()

mesi = ['gen', 'feb', 'mar', 'apr', 'mag', 'giu', 'lug', 'ago', 'set', 'ott', 'nov', 'dic']

for y in range(10,18,1):
    varn = 'AS'+str(y)
    res = np.repeat(0, 12)
    for m in range(len(mesi)):
        res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].mean()
    monthwise[varn] = res
    
mw = pd.DataFrame.from_dict(monthwise).set_index([mesi])
mw['AS17'].ix['ott'] = (39.7+40.1)/2

mw.plot()

mw.mean(axis=1).plot()

##############################################

normalized_monthwise = OrderedDict()

for y in range(10,18,1):
    varn = 'AS'+str(y)    
    res = np.repeat(0, 12)    
    if y < 17:        
        for m in range(len(mesi)):
            res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].mean()
        res2 = (res - np.mean(res))/np.std(res)
        normalized_monthwise[varn] = res2
    else:
        for m in range(11):
            if m < 10:
                res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].mean()
            else:
                res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].sum()/2
        res2 = (res - np.mean(res))/np.std(res)
        normalized_monthwise[varn] = res2
        
normalized_monthwise

nmw = pd.DataFrame.from_dict(normalized_monthwise).set_index([mesi])
nmw['AS17'].ix['nov'] = nmw['AS17'].ix['dic'] = 0 

#############################################

diffs = OrderedDict()

for y in range(10,18,1):
    varn = 'AS'+str(y)
    diffs[varn] = np.diff(cal[varn])
    plt.figure() 
    plt.plot(np.diff(cal[varn]))
    plt.suptitle(varn)
    
df = pd.DataFrame.from_dict(diffs)

############################################
def find_peaks_return(var, df, d, bVerbose = True):
    mu = df[var].mean()
    sigma = df[var].std()
    freq = np.count_nonzero( (df[var]-mu)/sigma > d )/df.shape[0]
    print('frequency of points greater than {} = {}'.format(d, freq))
    diff_values = np.diff(df[var].ix[np.abs((df[var]-mu)/sigma) > d])
    if bVerbose:
            print('values differences between peaks: {}'.format(diff_values))        
    num_contr = np.diff(df[var].ix[np.abs((df[var]-mu)/sigma) > d].index)
    if bVerbose: 
        print('distance between peaks:'.format(num_contr))
    return diff_values, num_contr
###########################################

dv, nc = find_peaks_return('AS10', df, 1, False)

means = []
for y in range(10,18,1):
    varn = 'AS'+str(y)
    dv, nc2 = find_peaks_return(varn, df, 1, False)
    print('media num contrattazioni di picco:'.format(np.nanmean(nc2)))
    means.append(np.mean(nc2))
