# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:42:21 2016

@author: d_floriello

PUN Pricing
"""

import pandas as pd
import numpy as np
from pandas.tools import plotting
import matplotlib.pyplot as plt

data1 = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Anno '+str(2014)+'.xlsx', sheetname = 'Prezzi-Prices')

data2 = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx", sheetname = 'DB_Dati')

data3 = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')

pun = []
pun.append(data1['PUN'].values.ravel())
pun.append(data2['PUN [€/MWH]'].values.ravel())
pun.append(data3['PUN [€/MWH]'].dropna().values.ravel())

unlisted =  [item for sublist in pun for item in sublist]

df = pd.DataFrame(unlisted) ######### to: 2 DAYS AHEAD OF TODAY
df = df.set_index(pd.date_range('2014-01-01', '2016-12-24', freq = 'H')[:df.shape[0]])

df.plot()
df.resample('D').mean().plot()
df.resample('M').mean().plot()

plt.figure()
plotting.lag_plot(df.resample('M').mean())

plt.figure()
plotting.autocorrelation_plot(df)
plt.figure()
plotting.autocorrelation_plot(df.resample('D').mean())

plt.figure()
plotting.autocorrelation_plot(df.ix[df.index.year == 2014].resample('D').mean())
plt.figure()
plotting.autocorrelation_plot(df.ix[df.index.year == 2015].resample('D').mean())
plt.figure()
plotting.autocorrelation_plot(df.ix[df.index.year == 2016].resample('D').mean())

plt.figure()
plotting.lag_plot(df.ix[df.index.year == 2014])
plt.figure()
plotting.lag_plot(df.ix[df.index.year == 2015], color = 'red')
plt.figure()
plotting.lag_plot(df.ix[df.index.year == 2016], color = 'green')

df.ix[df.index.year == 2014].hist(bins = 20)
df.ix[df.index.year == 2015].hist(bins = 20, color = 'red')
df.ix[df.index.year == 2016].hist(bins = 20, color = 'green')

df.ix[df.index.year == 2014].mean()
df.ix[df.index.year == 2014].std()
df.ix[df.index.year == 2015].mean()
df.ix[df.index.year == 2015].std()
df.ix[df.index.year == 2016].mean()
df.ix[df.index.year == 2016].std()

df.ix[df.index.year == 2014].plot()
df.ix[df.index.year == 2015].plot(color = 'red')
df.ix[df.index.year == 2016].plot(color = 'green')

df.ix[df.index.year == 2014].resample('D').mean().plot()
df.ix[df.index.year == 2015].resample('D').mean().plot(color = 'red')
df.ix[df.index.year == 2016].resample('D').mean().plot(color = 'green')

df.resample('D').std().plot()

df.to_excel('dati_2014-2016.xlsx')

##########

import statsmodels.api

fit = statsmodels.api.tsa.ARIMA(df.values.ravel(), (2,2,2)).fit()


