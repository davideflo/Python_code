# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:18:23 2016

@author: d_floriello

comparison pun - Francia 2010 - 2014
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import OrderedDict
import Fourier

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
yvp = []
yvf = []
dvp = []
dvf = []
wvp = []
wvf = []
years = [2010,2011,2012,2013,2014,2015,2016]
for year in years:
    if year < 2015:        
        fran = pd.read_excel('C:/Users/d_floriello/Documents/Storico prezzi Francia (2010-2014).xlsx', sheetname = str(year))
        fran = fran.set_index(fran['Data'])    
        dvf.append(np.array(fran['EPEX Francia'].resample('D').std()).tolist())
        fran = fran['EPEX Francia'].resample('D').mean()
        franw = fran.resample('W').mean()        
        pun = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Anno '+str(year)+'.xlsx', sheetname = 'Prezzi-Prices')
        pun = pun.set_index(pd.date_range(start = str(year)+'-01-01', end = str(year+1)+'-01-01', freq = 'H')[:pun.shape[0]])
        dvp.append(np.array(pun['PUN'].dropna().resample('D').std()).tolist())        
        pun = pun['PUN'].resample('D').mean()
        punw = pun.resample('W').mean()
    elif year == 2015:
        fran = pd.read_excel('C:/Users/d_floriello/Documents/Prezzi Francia e Svizzera (2015 -2016).xlsx', sheetname = '2015')
        fran = fran[fran.columns[[2,3]]].set_index(fran['Data'])
        dvf.append(np.array(fran[fran.columns[0]].resample('D').std()).tolist())
        fran = fran[fran.columns[0]].resample('D').mean()
        franw = fran.resample('W').mean()
        data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx", sheetname = 'DB_Dati')
        data = data.set_index(data['Date'])
        pun = data['PUN [€/MWH]'].resample('D').mean()
        dvp.append(np.array(data['PUN [€/MWH]'].dropna().resample('D').std()).tolist())      
        punw = pun.resample('W').mean()
    elif year == 2016:
        fran = pd.read_excel('C:/Users/d_floriello/Documents/Prezzi Francia e Svizzera (2015 -2016).xlsx', sheetname = '2016')
        fran = fran[fran.columns[[2,3]]].set_index(fran['Data'])
        dvf.append(np.array(fran[fran.columns[0]].resample('D').std()).tolist())
        fran = fran[fran.columns[0]].resample('D').mean()
        days = np.unique(fran.index) 
        fr = []
        for d in days:
            fr.append(fran.ix[fran.index == d].mean())
        nd = [41.78, 38.19, 32.48, 36.04, 42.02,42.68 ,48.28,
              57.29208333	, 44.35375,	36.705,56.70208333, 71.21208333, 62.81041667, 64.25, 64.10, 44.28, 40.02, 56.41,66.94,
              67.69,	76.30,	72.95,	55.72,	44.57,	72.63,
              79.92, 70.53, 61.49,58.38,52.12,40.95,48.95,47.63,64.01,68.82,74.08,50.90,43.47,125.67]    
        for n in nd:
            fr.append(n)            
        fran = pd.DataFrame.from_dict(fr).set_index(pd.date_range('2016-01-01', '2016-11-07', freq = 'D'))
        franw = fran.resample('W').mean()
        data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')
        data = data.set_index(data['Date'])
        pun = data['PUN [€/MWH]'].dropna().resample('D').mean()
        dvp.append(np.array(data['PUN [€/MWH]'].dropna().resample('D').std()).tolist())      
        punw = pun.resample('W').mean()
   
    volpc.append(np.array(pun.resample('M').std()).tolist())
    volfc.append(np.array(fran.resample('M').std()).ravel().tolist())
    yvp.append(pun.std())
    yvf.append(fran.std())
    wvp.append(pun.resample('W').std())
    wvf.append(fran.resample('W').std())
  
  
vol_p = [item for sublist in volpc for item in sublist]
vol_f = [item for sublist in volfc for item in sublist]
wv_p = [item for sublist in wvp for item in sublist]
wv_f = [item for sublist in wvf for item in sublist]  
dv_p = [item for sublist in dvp for item in sublist] ### daily volatility
dv_f = [item for sublist in dvf for item in sublist]  
  
plt.figure()
plt.hist(np.array(vol_p))
plt.figure()
plt.hist(np.array(vol_f), color = 'green')

plt.figure()
plt.hist(np.array(dv_p), bins = 40)
plt.figure()
plt.hist(np.array(dv_f), bins = 400, color = 'green')

plt.figure()
plt.plot(np.array(vol_p))
plt.figure()
plt.plot(np.array(vol_f), color = 'green')

dv_f2 = dv_f
dv_f2[769] = 100
dv_f2[770] = 100

plt.figure()
plt.plot(np.array(dv_p)) ##### daily volatility very regular!!!
plt.figure()
plt.plot(np.array(dv_f), color = 'red')
plt.figure()
plt.plot(np.array(dv_f2), color = 'green')

dv_f = np.array(dv_f)
np.min(dv_p)
np.min(dv_f)
np.mean(dv_p)
np.std(dv_p)
np.mean(dv_f)
np.std(dv_f)
np.max(dv_f)

plt.figure()
plt.plot(np.array(vol_f), color = 'green')
plt.plot(np.array(vol_p))


plt.figure()
plt.plot(np.array(wv_f), color = 'green')
plt.plot(np.array(wv_p))

plt.figure()
plt.plot((vol_p - np.mean(vol_p))/np.std(vol_p))
###############################################################################
def ECDF(x,nbins):
    h, bins = np.histogram(np.array(x), nbins)
    n = len(x)
    ecdf = np.cumsum(h)/n
    return ecdf      
###############################################################################
volpp = ECDF(vol_p, 20)
volff = ECDF(vol_f, 20)

plt.figure()
plt.plot(volff)
plt.plot(volpp)

scipy.stats.ks_2samp(np.array(volff),np.array(volpp))

import statsmodels.api

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(vol_p)))
plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(vol_f)))

########## do sigmas cluster? #############
mesi = ['gen', 'feb', 'mar', 'apr', 'mag', 'giu', 'lug', 'ago', 'set', 'ott', 'nov', 'dic']

sigp = OrderedDict()
sigf = OrderedDict()

am = mesi * 6

X = np.vstack((np.array(vol_p, dtype='float64'),np.concatenate((am, mesi[:10]))))
Xf = np.vstack((np.array(vol_f, dtype = 'float64'),np.concatenate((am, mesi[:10]))))

volPm = pd.DataFrame((np.array(vol_p, dtype='float64'))).set_index(np.concatenate((am, mesi[:10])))
volFm = pd.DataFrame((np.array(vol_f, dtype='float64'))).set_index(np.concatenate((am, mesi[:10])))


winter = ['ott', 'nov', 'dic', 'gen', 'feb', 'mar']

wp = []
sp = []
wf = []
sf = []
for i in range(volPm.shape[0]):
    if volPm.index[i] in winter:
        wp.append(volPm.ix[i])
    else:
        sp.append(volPm.ix[i])

for i in range(volFm.shape[0]):
    if volFm.index[i] in winter:
        wf.append(volFm.ix[i])
    else:
        sf.append(volFm.ix[i])
 
plt.figure()
plt.scatter(np.repeat(0,len(wp)),np.array(wp), color = 'blue') 
plt.scatter(np.repeat(1,len(sp)),np.array(sp), color = 'red') 
 
plt.figure()
plt.scatter(np.repeat(0,len(wf)),np.array(wf), color = 'blue') 
plt.scatter(np.repeat(1,len(sf)),np.array(sf), color = 'red') 

plt.figure()
plt.scatter(np.repeat(0,len(wp)),np.array(wp), color = 'blue') 
plt.scatter(np.repeat(1,len(sp)),np.array(sp), color = 'red') 
plt.scatter(np.repeat(0.2,len(wf)),np.array(wf), color = 'black') 
plt.scatter(np.repeat(1.2,len(sf)),np.array(sf), color = 'orange') 
plt.scatter(np.repeat(0.4,len(vol_p)),np.array(vol_p), color = 'green') 
plt.scatter(np.repeat(0.6,len(vol_f)),np.array(vol_f), color = 'purple') 


scipy.stats.mstats.mquantiles(dv_p, prob = [0.025, 0.975])
scipy.stats.mstats.mquantiles(dv_f, prob = [0.025, 0.975])


###############

divp = np.diff(vol_p)
divf = np.diff(vol_f)

plt.figure()
plt.plot(divp)
plt.plot(np.array(vol_p))
plt.figure()
plt.plot(divf, color = 'magenta')
plt.plot(np.array(vol_f), color = 'black')


ddvp = np.diff(dv_p)
ddvf = np.diff(dv_f)

plt.figure()
plt.plot(ddvp)
plt.plot(np.array(dv_p))
plt.figure()
plt.plot(ddvf, color = 'magenta')
plt.plot(np.array(dv_f), color = 'black')

fdvp = Fourier.fourierExtrapolation(dv_p,0, 25) ### best one to see the 'right' process for the volatility
plt.figure()
plt.plot(np.array(dv_p))
plt.plot(fdvp, color = 'black', lw = 2)
