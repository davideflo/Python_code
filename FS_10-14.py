# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:18:23 2016

@author: d_floriello

comparison pun - Francia 2010 - 2014
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### 2011 - 2012 a bit anomalous
year = 2014

fran = pd.read_excel('C:/Users/d_floriello/Documents/Storico prezzi Francia (2010-2014).xlsx', sheetname = str(year))
fran = fran.set_index(fran['Data'])

fran = fran['EPEX Francia'].resample('D').mean()

pun = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Anno '+str(year)+'.xlsx', sheetname = 'Prezzi-Prices')
pun = pun.set_index(pd.date_range(start = str(year)+'-01-01', end = str(year+1)+'-01-01', freq = 'H')[:pun.shape[0]])

pun = pun['PUN'].resample('D').mean()

plt.figure()
plt.plot(pun)
plt.plot(fran)
####### correlation analysis #######
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

corr_else = []
corr_upto = []
for i in range(1, 13, 1):
    corr_else.append(pun.ix[pun.index.month == i].corr(fran.ix[fran.index.month == i]))
    corr_upto.append(pun.ix[pun.index.month <= i].corr(fran.ix[fran.index.month <= i]))

plt.figure()
plt.plot(np.array(corr_else), marker = 'o')
plt.plot(np.array(corr_upto), marker = '*')

dpun = pd.Series(np.diff(pun).ravel(), index = pd.date_range(str(year)+'-01-02', str(year)+'-12-31', freq = 'D'))
dfran = pd.Series(np.diff(fran).ravel(), index = pd.date_range(str(year)+'-01-02', str(year)+'-12-31', freq = 'D'))

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

volpc = []
volfc = []
years = [2010,2011,2012,2013,2014]
for year in years:
    fran = pd.read_excel('C:/Users/d_floriello/Documents/Storico prezzi Francia (2010-2014).xlsx', sheetname = str(year))
    fran = fran.set_index(fran['Data'])
    fran = fran['EPEX Francia'].resample('D').mean()
    pun = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Anno '+str(year)+'.xlsx', sheetname = 'Prezzi-Prices')
    pun = pun.set_index(pd.date_range(start = str(year)+'-01-01', end = str(year+1)+'-01-01', freq = 'H')[:pun.shape[0]])
    pun = pun['PUN'].resample('D').mean()
    volpc.append(pun.resample('M').std())
    volfc.append(fran.resample('M').std())
    
plt.figure()
plt.hist(np.array(volpc).ravel())
plt.figure()
plt.hist(np.array(volfc).ravel(), color = 'green')