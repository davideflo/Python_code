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
from scipy.interpolate import interp1d
from scipy.misc import derivative
import seaborn as sns


cal = pd.read_excel('CAL.xlsx', sheetname = 'valori CAL')
cal = cal.fillna(None)

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

###########################################
### how far are the max and min from the mean? and from the initial value? 

stats = OrderedDict()

stats['mins'] = cal.min()
stats['maxs'] = cal.max()
stats['means'] = cal.mean()
stats['stds'] = cal.std()
stats['starts'] = cal.ix[0]
stats['ends'] = cal.ix[cal.shape[0] - 1]
stats['skews'] = cal.skew()

stats = pd.DataFrame.from_dict(stats)

######################################################
def std_distance(x, y, s):
    print('{} is at {} sigmas from {}'.format(x, abs(x - y)/s, y))
    return abs(x - y)/s
#####################################################

std_distance(stats['mins'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])
std_distance(stats['means'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])
std_distance(stats['starts'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])
std_distance(stats['ends'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])

#####################################################
def rolling_dist(ts):
    dist = []
    for i,x in enumerate(ts[1:]):
        pmean = np.mean(ts[:i])
        psig = np.std(ts[:i])
        dist.append(std_distance(x, pmean, psig))
    return np.array(dist)
######################################################
        
rd = rolling_dist(cal['AS16'])

plt.figure()
plt.plot(rd, marker='o')

######################################################
def plot_mean_graphs(ts):
    #ts = cal['AS16']
    cum_mu = []
    for i,x in enumerate(ts):
        cum_mu.append(np.mean(ts[:i]))
    cum_mu = np.array(cum_mu)
    plt.figure()
    plt.plot(cum_mu, marker = 'o', color = 'magenta')
    plt.figure()
    plt.plot(np.diff(cum_mu), marker = 'o', color = 'lime')
    return cum_mu, np.diff(cum_mu)
######################################################
mu,diff = plot_mean_graphs(cal['AS14'])

#####################################################

f_mean = interp1d(np.linspace(-1,  cal.shape[0], cal.shape[0]), cal['AS16'])

der = []
for x in np.linspace(5, cal.shape[0], 2*cal.shape[0]):
    print(x)
    der.append(derivative(f_mean, x))

plt.figure()
plt.plot(np.array(der))

####################################################
def cumulative_diffs(ts, bVerbose = False):
    M = np.max(ts)
    v = [] ### dist from cumulative mean
    curr = [] ### dist from current value
    st = []
    for i,x in enumerate(ts[1:]):
        if bVerbose:
            M = x
        if np.isnan((M - x)/np.std(ts[:i])) or np.isinf((M - x)/np.std(ts[:i])):
            pass
        else:
            curr.append((M - x)/np.std(ts[:i]))
        if np.isnan((M - np.mean(ts[:i]))/np.std(ts[:i])) or np.isinf((M - np.mean(ts[:i]))/np.std(ts[:i])):
            pass
        else:
            v.append((M - np.mean(ts[:i]))/np.std(ts[:i])) 
        if np.isnan((M - ts[0])/np.std(ts[:i])) or np.isinf((M - ts[0])/np.std(ts[:i])):
            pass
        else:
            st.append((M - ts[0])/np.std(ts[:i]))
            
    mark = [ts.tolist().index(np.max(ts)),ts.tolist().index(np.min(ts))] 
        
    plt.figure()
    plt.plot(np.array(v), '-gD', markevery = mark)
    plt.title('max vs cumulative mean')        
    plt.figure()
    plt.plot(np.array(curr), '-gD', markevery = mark)
    plt.title('max vs current value')        
    plt.figure()
    plt.plot(np.array(st), '-gD', markevery = mark)
    plt.title('max vs first value')        
        
    return v, curr, st
###################################################
dist_mu, current, dist_st = cumulative_diffs(cal['AS16'], True) 
dist_mu, current, dist_st = cumulative_diffs(cal['AS15'], True) 
dist_mu, current, dist_st = cumulative_diffs(cal['AS14'], True) 


plt.figure()
plt.plot(dist_mu[2:])

plt.figure()
plt.plot(current[2:])
plt.figure()
plt.plot(dist_st[2:])

###################################################
def H1_distance(ts1, ts2):
    return np.mean((ts1 - ts2)**2) + np.mean((np.diff(ts1) - np.diff(ts2))**2)
###################################################
    
sim = OrderedDict()
    
for cn1 in cal.columns[:7]:
    dist = []
    for cn2 in cal.columns[:7]:
        print('{} with {} = {}'.format(cn1, cn2, H1_distance(cal[cn1], cal[cn2])))
        dist.append(H1_distance(cal[cn1], cal[cn2]))
    sim[cn1] = dist
    
sim = pd.DataFrame.from_dict(sim).set_index([['AS10','AS11','AS12','AS13','AS14','AS15','AS16']])

sns.heatmap(sim)

log_cal = OrderedDict()

for cn in cal.columns[:8]:
    log_cal[cn] = np.diff(np.log(cal[cn]))
    
log_cal = pd.DataFrame.from_dict(log_cal)

############################################################################
def cumulative_maxmin(ts):
    M = []
    m = []
    for i in range(ts.size):
        M.append(np.max(ts[:i]))
        m.append(np.min(ts[:i]))
    plt.figure()
    plt.plot(np.array(M))
    plt.title('cumulative max')
    plt.figure()
    plt.plot(np.array(m))
    plt.title('cumulative min')
    return M, m
############################################################################
def sell(x, y, hold, threshold = 0):
    if hold: ### hold means I have something bought at a given price
        if y - x > threshold:
            return y - x
        else:
            return 0
    else:
        return 0
############################################################################  
def buy(last, y, free, threshold = 0):
    if free:
        if y < abs(last - threshold):
            return -y
        else:
            return 0
    else:
        return 0
############################################################################            
def simulate_strategy(ts, threshold = 0):
    x = last = ts[0]
    hold = True
    free = False
    portfolio = left = 0
    for y in ts[1:]:
        if hold:
            portfolio += sell(x, y, hold, threshold)
            hold = False
            last = y
            free = True
        elif free:
            portfolio += buy(last, y, free, threshold)
            free = False
            x = last
            hold = True
        else:
            left += 1
    print('days with no operations: {}'.format(left))
    return portfolio
        
        









