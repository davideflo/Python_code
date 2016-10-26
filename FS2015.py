# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:33:20 2016

@author: d_floriello

Analisi Francia - Svizzera - PUN 2015
"""

import pandas as pd
import numpy as np
#from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats

fs = pd.read_excel('C:/Users/d_floriello/Documents/Prezzi Francia e Svizzera (2015 -2016).xlsx', sheetname = '2015')
fs = fs[fs.columns[[2,3]]].set_index(fs['Data'])
fs.plot()

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
pun = data['PUN [€/MWH]'].resample('D').mean()

fran = fs[fs.columns[0]].resample('D').mean()

plt.figure()
plt.plot(pun)
plt.plot(fran)

######## correlation analysis 2015 #############♣
cors = []
for i in range(2,pun.shape[0],1):
    cors.append(np.corrcoef(np.array(pun)[:i],np.array(fran)[:i])[1,0])

compl_cors = []
for i in range(2,pun.shape[0],1):
    compl_cors.append(np.corrcoef(np.array(pun)[pun.shape[0] - i:],np.array(fran)[pun.shape[0] - i:])[1,0])
    
plt.figure()
plt.plot(np.array(cors))
plt.figure()
plt.plot(np.array(compl_cors))

ottp = pun.ix[pun.index.month == 10]
ottf = fran.ix[fran.index.month == 10]
nottp = pun.ix[pun.index.month < 10]
nottf = fran.ix[fran.index.month < 10]

corr_ottobre = ottp.corr(ottf)
corr_else = []
corr_upto = []
for i in range(1, 13, 1):
    corr_else.append(pun.ix[pun.index.month == i].corr(fran.ix[fran.index.month == i]))
    corr_upto.append(pun.ix[pun.index.month <= i].corr(fran.ix[fran.index.month <= i]))

plt.figure()
plt.plot(np.array(corr_else), marker = 'o')
plt.plot(np.array(corr_upto), marker = '*')
plt.scatter(np.array([9]), np.array([corr_ottobre]), color = 'black', marker = 'D')

dcors = []
for i in range(3,pun.shape[0]-1,1):
    dcors.append(np.corrcoef(np.diff(pun)[pun.shape[0] - i:],np.diff(np.array(fran))[pun.shape[0] - i:])[1,0])
    
plt.figure()
plt.plot(np.array(dcors))    

dpun = pd.Series(np.diff(pun).ravel(), index = pd.date_range('2015-01-02', '2015-12-31', freq = 'D'))
dfran = pd.Series(np.diff(fran).ravel(), index = pd.date_range('2015-01-02', '2015-12-31', freq = 'D'))

dcorr_else = []
dcorr_upto = []
for i in range(1, 13, 1):
    dcorr_else.append(dpun.ix[dpun.index.month == i].corr(dfran.ix[dfran.index.month == i]))
    dcorr_upto.append(dpun.ix[dpun.index.month <= i].corr(dfran.ix[dfran.index.month <= i]))

plt.figure()
plt.plot(np.array(dcorr_else), marker = 'o')
plt.plot(np.array(dcorr_upto), marker = '*')

###### volatility and percentage increments #####
volp = pun.resample('M').std()
volf = fran.resample('M').std()

plt.figure()
plt.plot(volp, marker = 'o')
plt.plot(volf, marker = '*')

meanp = pun.resample('M').mean()
meanf = fran.resample('M').mean()

percp = []
percf = []
for i in range(meanp.size - 1):
    percp.append((meanp[i+1] - meanp[i])/meanp[i])
    percf.append((meanf[i+1] - meanf[i])/meanf[i])
    
plt.figure()
plt.plot(np.array(percp), marker = 'o')
plt.plot(np.array(percf), marker = '*')
###############################################################################

fm = fs[fs.columns[0]].resample('M').mean()
m = [5,5,7,10,12,16,19,18,12,10,10,8]
M = [8,8,13,19,20,25,28,27,20,16,14,11]

med = []
for i in range(12):
    med.append((m[i] + M[i])/2)

lm = linear_model.LinearRegression(fit_intercept = True).fit(np.array(med).reshape(-1,1), np.array(fm).reshape(-1,1))
lmr = linear_model.RANSACRegressor(linear_model.LinearRegression()).fit(np.array(med).reshape(-1,1), np.array(fm).reshape(-1,1))

lm.coef_

yhat = lm.predict(np.linspace(start = 0, stop = 30, num = 60).reshape(-1,1))
yrhat = lmr.predict(np.linspace(start = 0, stop = 30, num = 60).reshape(-1,1))

label = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
plt.figure()
plt.plot(np.linspace(start = 0, stop = 30, num = 60), yhat.ravel())
plt.plot(np.linspace(start = 0, stop = 30, num = 60), yrhat.ravel(), color = 'black')
plt.scatter(np.array(med), np.array(fm), color = 'red', marker = 'o')
for i,l in enumerate(label):
    plt.annotate(l, xy = (np.array(med)[i],np.array(fm)[i]))
###############################################################################
print(np.mean(pun))    
print(np.std(pun))
print(np.max(pun))
print(np.min(pun))
print(np.mean(fran))    
print(np.std(fran))
print(np.max(fran))
print(np.min(fran))

###### fabbisogno 

fabb = pd.read_excel('C:/Users/d_floriello/Documents/fabb_2016_fran.xlsx')

fabb = fabb.set_index(fabb[fabb.columns[0]])
fabb.columns = [['data', 'ore', 'jp', 'j']]

fabb = (1/1000)*fabb[['jp', 'j']]
fabb.plot()

fp = pd.DataFrame(fabb.resample('D').mean(),fran)
fp.plot()

fabb.resample('D').mean().plot()
plt.figure()
plt.plot(fran, color = 'black', lw = 2)

Jfab = pd.Series(fabb['j'], dtype = 'float64')
Fpun = pd.Series(fran, dtype = 'float64')

Jfab.corr(Fpun)

plt.figure()
plt.scatter(np.array(Jfab.resample('D').mean()), np.array(Fpun))

log_model = linear_model.LinearRegression(fit_intercept = True).fit(np.log(np.array(Jfab.resample('D').mean())).reshape(-1,1),np.array(Fpun))
Lyhat = log_model.predict(np.log(np.array(Jfab.resample('D').mean())).reshape(-1,1))

R2_log = 1 - np.sum((np.array(Fpun) - Lyhat)**2)/(np.sum((np.array(Fpun) - np.mean(Fpun))**2))

plt.figure()
plt.plot(np.array(Jfab.resample('D').mean()), Lyhat, color = 'red')
plt.scatter(np.array(Jfab.resample('D').mean()), np.array(Fpun))

sqr_model = linear_model.LinearRegression(fit_intercept = True).fit(np.sqrt(np.array(Jfab.resample('D').mean())).reshape(-1,1),np.array(Fpun))
Syhat = sqr_model.predict(np.sqrt(np.array(Jfab.resample('D').mean())).reshape(-1,1))

R2_sqr = 1 - np.sum((np.array(Fpun) - Syhat)**2)/(np.sum((np.array(Fpun) - np.mean(Fpun))**2))

plt.figure()
plt.plot(np.array(Jfab.resample('D').mean()), Syhat, color = 'grey')
plt.scatter(np.array(Jfab.resample('D').mean()), np.array(Fpun))

cub_model = linear_model.LinearRegression(fit_intercept = True).fit(np.power(np.array(Jfab.resample('D').mean()),1/3).reshape(-1,1),np.array(Fpun))
Cyhat = cub_model.predict(np.power(np.array(Jfab.resample('D').mean()),1/3).reshape(-1,1))

R2_cub = 1 - np.sum((np.array(Fpun) - Cyhat)**2)/(np.sum((np.array(Fpun) - np.mean(Fpun))**2))

plt.figure()
plt.plot(np.array(Jfab.resample('D').mean()), Cyhat, color = 'green')
plt.scatter(np.array(Jfab.resample('D').mean()), np.array(Fpun))

dfab = np.diff(Jfab.resample('D').mean())
dfpun = np.diff(Fpun)

plt.figure()
plt.plot(dfab)
plt.figure()
plt.plot(dfpun, color = 'red')

coh = np.sign(dfab) * np.sign(dfpun)
np.where(coh < 0)[0].size/365

qfab = scipy.stats.mstats.mquantiles(Jfab.resample('D').mean(), prob = [0.025, 0.975])
qfpun = scipy.stats.mstats.mquantiles(Fpun, prob = [0.025, 0.975])

fab_down = np.where(Jfab.resample('D').mean() < qfab[0])
fab_up = np.where(Jfab.resample('D').mean() > qfab[1])
fpun_down = np.where(Fpun < qfpun[0])
fpun_up = np.where(Fpun > qfpun[1])

plt.figure()
plt.plot(np.array(Jfab.resample('D').mean()))
plt.scatter(fab_down[0], np.array(Jfab.resample('D').mean())[fab_down[0]], color = 'black')
plt.scatter(fab_up[0], np.array(Jfab.resample('D').mean())[fab_up[0]], color = 'black')

plt.figure()
plt.plot(np.array(Fpun), color = 'red')
plt.scatter(fpun_down[0], np.array(Fpun)[fpun_down[0]], color = 'black')
plt.scatter(fpun_up[0], np.array(Fpun)[fpun_up[0]], color = 'black')