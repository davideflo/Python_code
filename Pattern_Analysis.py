# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:11:16 2016

@author: d_floriello

Script for pattern analysis in PUN time series
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
from sklearn import linear_model

data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")
data2 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2011.xlsx")
data3 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2012.xlsx")
data4 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2013.xlsx")
data5 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2014.xlsx")
data6 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2015.xlsx")

data7 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_06.xlsx")

pun = data["PUN"]
pun = data["CSUD"]
#dates = temp.dates(data[data.columns[0]])
#prova = pd.to_datetime(dates)
df = pd.DataFrame(pun)
rng = pd.date_range('01/01/2010', periods=8760, freq='H')
df = df.set_index(rng)

dec = sm.tsa.seasonal_decompose(df.values,freq=24)
dec.plot()

s = dec.seasonal

min_season = np.array(s[0:24])
plt.plot(min_season)

rmse = []
for i in range(24, 8760, 24):
    ns = s[i:i+24]
    rmse.append(np.sqrt(np.mean((min_season - ns)**2)))
### seasonal component is identical during every day of the year

random_hour = np.random.randint(0,8760,1)    
    
initial = np.array(pun[0:24])
random_pun = np.array(pun[random_hour[0]:random_hour[0]+24])

plt.plot(initial)
plt.plot(random_pun)

###############################################
def deseasonalise(x, min_s, freq):
    x_ds = []
    for i in range(0, x.size, freq):
        x_ds.append(x[i:i+24] - min_s)
    x_ds = np.array(x_ds).flatten()
    return x_ds
##############################################

des = deseasonalise(pun, min_season, 24)     
    
dec_des = sm.tsa.seasonal_decompose(des,freq=24)
dec_des.plot()
dec.plot()

sds = dec_des.seasonal

min_season_des = np.array(sds[0:24])
plt.plot(min_season_des)

x = np.array(range(24))

new_length = 100
new_x = np.linspace(0, 23, new_length)
fhat = sp.interpolate.interp1d(x, min_season, kind='cubic')

plt.plot(new_x,fhat(new_x))
plt.plot(min_season)

tr = dec.trend
plt.plot(tr)

plt.subplot(2,1,1)
plt.plot(tr)
plt.subplot(2,1,2)
plt.plot(pun)

####### empirical risk analysis ######
(np.max(pun) - np.mean(pun))/np.std(pun)
 
index_max = pun.tolist().index(np.max(pun)) 
rng[index_max]
 
## how many times the values are > mean+x*sigma?
#################################################
def freq_greater_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int((x - np.mean(ts))/np.std(ts) > sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
################################################# 
for x in range(1,7,1):
   print(x)
   print( '%.6f' % freq_greater_than(pun, x, False))

sigma_4 = freq_greater_than(pun, 4, True)
out_hours = np.where(sigma_4 > 0)[0] 

count = 0
for i in range(out_hours.size - 1):
    if out_hours[i+1] - out_hours[i] == 1:
        count += 1
    else:
        pass
################################################
def rolling_mean_at(ts, time_interval):
    tsm = []
    for i in range(0, ts.size, time_interval):
        tsm.append(np.mean(ts[i:i+time_interval]))
    return np.array(tsm)
################################################
m24 = rolling_mean_at(pun, 24)    
m2 = rolling_mean_at(pun, 2)

plt.subplot(2,1,1)
plt.plot(m24)
plt.subplot(2,1,2)
plt.plot(tr)
################################################
### NOT RUN
def adaptive_trend(ts, m, n, I, O, coefs,epsilon=1e-06):
    y0 = np.array(ts[m:n+1])
    x0 = np.linspace(m,n,y0.size)
    model0 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 20)
    model0.fit(x0.reshape([x0.size,1]), y0)
    coefs.append(model0.estimator_.coef_)
    for i in range(n+1, ts.size, 1):
        xnew = ts[i]
        ynew = model0.predict(i)
        error = abs(ynew - xnew)
        if error <= epsilon:
            I.append(i)
        else:
            O.append(i)
            adaptive_trend(ts, i, i+1, I, O, coefs, epsilon)
    return 0
###############################################    
I = []
O = []
coefs = []   
epsilon = 20
adaptive_trend(pun, 0, 100, I, O, coefs, epsilon)    
### END: NOT RUN


#print("Residual sum of squares: %.2f"
#      % np.mean((fit.predict(x.reshape([len(x),1])) - y) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % fit.score(x.reshape([len(x),1]), y))
# 