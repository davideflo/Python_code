# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:42:29 2017

@author: utente

Sbilanciamento Terna
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
import calendar
import scipy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

####################################################################################################
def SimilarDaysError(df):
    errors = []
    for m in range(1,13,1):
        dim = calendar.monthrange(2015, m)[1]
        dfm = df.ix[df.index.month == m]
        dfm5 = dfm.ix[dfm.index.year == 2015]
        dfm6 = dfm.ix[dfm.index.year == 2016]
        for d in range(1, dim, 1):
            ddfm5 = dfm5.ix[dfm5.index.day == d]
            ddfm6 = dfm6.ix[dfm6.index.day == d]
            if ddfm5.shape[0] == ddfm6.shape[0]:
                errors.extend(ddfm6['FABBISOGNO REALE'].values.ravel() - ddfm5['FABBISOGNO REALE'].values.ravel().tolist())
    return errors
####################################################################################################


sbil = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento.xlsx')
nord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_NORD']
nord.index = pd.date_range('2015-01-01', '2017-01-02', freq = 'H')[:nord.shape[0]]


nord['FABBISOGNO REALE'].plot()

nord['FABBISOGNO REALE'].resample('D').max()
nord['FABBISOGNO REALE'].resample('D').min()
nord['FABBISOGNO REALE'].resample('D').std()

nrange = nord['FABBISOGNO REALE'].resample('D').max() - nord['FABBISOGNO REALE'].resample('D').min()

plt.figure()
plt.plot(nrange)

dec = statsmodels.api.tsa.seasonal_decompose(nord['FABBISOGNO REALE'].values.ravel(), freq = 24)
dec.plot()

errn = SimilarDaysError(nord)

plt.figure()
plt.plot(np.array(errn), color = 'red')
plt.axhline(y = np.mean(errn), color = 'navy')
plt.axhline(y = np.median(errn), color = 'gold')
plt.axhline(y = scipy.stats.mstats.mquantiles(errn, prob = 0.025), color = 'black')
plt.axhline(y = scipy.stats.mstats.mquantiles(errn, prob = 0.975), color = 'black')


np.mean(errn)
np.median(errn)
np.std(errn)


wderrn = np.array(errn)[np.array(errn) <= 20]
wderrn = wderrn[wderrn >= -20]
wderrn.size/len(errn)

np.median(wderrn)
np.mean(wderrn)

plt.figure()
plt.plot(wderrn)

x = np.linspace(0, 8760, num = 8760)[:, np.newaxis]
y = nord['FABBISOGNO REALE'].ix[nord.index.year == 2015].values.ravel()
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 24),n_estimators=3000)

regr.fit(x, y)
yhat = regr.predict(x)

plt.figure()
plt.plot(yhat, color = 'blue', marker = 'o')
plt.plot(y, color = 'red')

plt.figure()
plt.plot(y - yhat)