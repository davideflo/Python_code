# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:36:48 2016

@author: d_floriello

balncing price analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import datetime


gpb = pd.read_excel('C:/Users/d_floriello/Documents/Prezzo di bilanciamento gas.xlsx', sheetname = 'val')
gpb = gpb.set_index(pd.to_datetime(gpb[gpb.columns[0]]))
bil = gpb['sbil']

plt.figure()
plt.plot(np.array(bil))

#### find anomalies ####

std_bil = (bil - np.mean(bil))/(np.std(bil))

scipy.stats.mstats.mquantiles(std_bil, prob = [0.90, 0.95])

q = sp.stats.mstats.mquantiles(std_bil, prob = 0.95)[0]

outers = []
out_ix = []
plt.figure()
plt.plot(np.array(std_bil))
plt.axhline(y = q)
plt.axhline(y = -q)
for i in range(1,std_bil.size,1):
    if np.abs(std_bil[i]) > q:
        outers.append(std_bil[i])
        out_ix.append(i)
plt.scatter(np.array(out_ix), np.array(outers), color = 'red', marker = '*')
print('outers percentage: {}'.format(len(outers)/std_bil.size))

outers
out_ix

yyhat = np.polyfit(np.arange(std_bil.size), std_bil, deg = 3, full = True)[0]


def rolling_weight(ts, i):
    ts2 = []
    for k,j in enumerate(np.arange(ts[:i].size-1, 0, -1)):
        ts2.append(ts[k] * np.exp((-1/30)*(j)))
    return np.array(ts2)
#############################################
def find_outliers_rolling(ts):
    index = []
    for i in range(1, ts.size, 1):
        ts2 = rolling_weight(ts, i)        
        st_ts_i = (ts2 - np.mean(ts2))/np.std(ts2)        
        qu = sp.stats.mstats.mquantiles(st_ts_i, prob = 0.95)[0]
        index.append(np.where(np.abs(st_ts_i) > qu)[0].tolist())
    return index
#############################################
def approximating_polynomial(x, yhat):
    xx = 0
    for y in yhat[:(yhat.size-1)]:
        deg = yhat.size - (yhat.tolist().index(y) + 1)
        xx += (x**deg) * y
    return xx + yhat[-1]
#############################################
gi = find_outliers_rolling(np.array(bil))

unp = set(gi[-1])

for i in range(len(gi)-1):
    unp = unp.union(gi[i])
 
unp = list(unp)

plt.figure()
plt.plot(np.array(bil))
plt.scatter(np.array(unp), np.array(bil.ix[unp]), color = 'black', marker = 'o')
    
phat = approximating_polynomial(np.arange(std_bil.size), yyhat) 

plt.figure()
plt.plot(np.array(std_bil))
plt.plot(phat)

#### how often is there a spike? ####

residuals = std_bil - phat

np.mean(residuals)
np.std(residuals)
qp = sp.stats.mstats.mquantiles(residuals, prob = 0.95)[0]

ixix = np.where(np.abs(residuals) > qp)[0]

plt.figure()
plt.plot(np.array(std_bil))
plt.plot(phat)
plt.scatter(ixix, std_bil.ix[ixix.tolist()], color = 'black', marker = 'o')

plt.figure()
plt.plot(residuals)
plt.axhline(y = qp)
plt.axhline(y = -qp)


out_res = []
for i in range(residuals.size):
    if np.abs(residuals[i]) > qp:
        out_res.append(i)

spikes = []
for i in range(std_bil.size):
    if i in out_res:
        spikes.append(std_bil.index[i])

time_deltas = []
for i in range(len(spikes) - 1):
    dt1 = datetime.datetime(int(str(spikes[i])[0:4]), int(str(spikes[i])[5:7]), int(str(spikes[i])[8:10]))
    dt2 = datetime.datetime(int(str(spikes[i+1])[0:4]), int(str(spikes[i+1])[5:7]), int(str(spikes[i+1])[8:10]))
    time_deltas.append((dt2 - dt1).days)
    
TD = np.array(time_deltas)
### mean(TD) = 7.849 -- std(TD) = 18.308
plt.figure()
plt.hist(TD)
np.median(TD)

import statsmodels
#from statsmodels import api

plt.figure()
plt.plot(statsmodels.api.tsa.stattools.periodogram(bil)[6:int(580/2)])

#### seasonality ####

from collections import OrderedDict

days = OrderedDict()
days_of_week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
for i in range(bil.size):
    for d in days_of_week:
        d_i = days_of_week.index(d)
        days[d] = bil.ix[datetime.datetime(bil.index[i].year, bil.index[i].month, bil.index[i].day).weekday() == d_i].mean()
        
    
seas = pd.DataFrame.from_dict(days)

bil = pd.DataFrame(bil).set_index(pd.to_datetime(bil.index))
bil.plot()

db = np.diff(np.array(bil.values).ravel())

plt.figure()
plt.hist(db)

plt.figure()
plt.plot(db)

qd = sp.stats.mstats.mquantiles(db, prob = 0.95)[0]

indeces = np.where(np.abs(db) > qd) #### more correct in my opinion

plt.figure()
plt.plot(np.array(bil))
plt.scatter(indeces[0],bil.ix[indeces[0]], color = 'black')