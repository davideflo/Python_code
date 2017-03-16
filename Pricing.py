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

data4 = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2017/06. MI/DB_Borse_Elettriche_PER MI_17_conMacro - Copy.xlsm", sheetname = 'DB_Dati')


pun = []
pun.append(data1['PUN'].values.ravel())
pun.append(data2['PUN [€/MWH]'].values.ravel())
pun.append(data3['PUN [€/MWH]'].dropna().values.ravel())
pun.append(data4['PUN [€/MWH]'].dropna().values.ravel())


unlisted =  [item for sublist in pun for item in sublist]

df = pd.DataFrame(unlisted) ######### to: 2 DAYS AHEAD OF LAST PUN
df = df.set_index(pd.date_range('2014-01-01', '2018-01-02', freq = 'H')[:df.shape[0]])

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

df.to_excel('dati_2014-2017.xlsx')

##########

import statsmodels.api

fit = statsmodels.api.tsa.ARIMA(df.values.ravel(), (2,2,2)).fit()

##########
### remodulation

years = [2014,2015,2016]
months = list(range(1,13))

mp = 49.04788936	
sigmap = 14.69504823

rem = []
for y in years:
    print(y)
    dfy = df.ix[df.index.year == y]
    for m in months:
        print(m)
        dfym = dfy.ix[dfy.index.month == m]
        print((dfym.mean().values[0] - mp)/sigmap)
        if (dfym.mean().values[0] - mp)/sigmap > 1:            
            print('>1')
            alpha = (sigmap + sigmap)/dfym.mean().values[0]
            ll = alpha * dfym[dfym.columns[0]].values.ravel().tolist()
            print(alpha)
            rem.extend(ll)
        else:
            rem.extend(dfym[dfym.columns[0]].values.ravel().tolist())

plt.figure()
plt.plot(np.array(rem), color = 'red')

dfr = pd.DataFrame(rem)
dfr = dfr.set_index(pd.date_range('2014-01-01', '2018-01-02', freq = 'H')[:dfr.shape[0]])

dfr.to_excel('redati_2014-2017.xlsx')

####################

df = df.set_index(pd.date_range('2015-01-01', '2018-01-02', freq = 'H')[:df.shape[0]])
df.plot()

df.resample('M').std()
df.resample('M').std().plot()

ore2015 = pd.read_excel('H:/Energy Management/SUDDIVISIONE_ORE_ANNO.xlsx', sheetname = '2015')
ore2016 = pd.read_excel('H:/Energy Management/SUDDIVISIONE_ORE_ANNO.xlsx', sheetname = '2016')

pkop = []
pkop.extend(ore2015['PK-OP'].values.ravel().tolist())
pkop.extend(ore2016['PK-OP'].values.ravel().tolist())
fasce  = []
fasce.extend(ore2015['AEEG 181/06'].values.ravel().tolist())
fasce.extend(ore2016['AEEG 181/06'].values.ravel().tolist())

df['PK-OP'] = pkop
df['fasce'] = fasce

df.groupby('PK-OP').resample('M').agg(np.mean)
df.groupby('PK-OP').resample('M').agg(np.std)

df.groupby('PK-OP').resample('M').agg(np.mean).ix['PK'] - df.resample('M').mean()
df.resample('M').mean() - df.groupby('PK-OP').resample('M').agg(np.mean).ix['OP']

dfpk = df.groupby('PK-OP').resample('M').agg(np.mean).ix['PK'] - df.resample('M').mean()
dfop = df.resample('M').mean() - df.groupby('PK-OP').resample('M').agg(np.mean).ix['OP']

spread_pk = []
for m in range(1,13):
    print('mean in month {} = {}'.format(m, dfpk.ix[dfpk.index.month == m].mean()))
    spread_pk.append(dfpk.ix[dfpk.index.month == m].mean().values.ravel()[0])

spread_op = []
for m in range(1,13):
    print('mean in month {} = {}'.format(m, dfop.ix[dfpk.index.month == m].mean()))
    spread_op.append(dfop.ix[dfop.index.month == m].mean().values.ravel()[0])

spread = pd.DataFrame({'spread_pk': spread_pk, 'spread_op': spread_op})
spread.to_excel('historical_spreads.xlsx')


dff = df.groupby('fasce').resample('M').agg(np.mean)
dff = df.groupby('fasce').resample('M').agg(np.std)

F1 = df.groupby('fasce').resample('M').agg(np.mean).ix['F1']
F2 = df.groupby('fasce').resample('M').agg(np.mean).ix['F2']
F3 = df.groupby('fasce').resample('M').agg(np.mean).ix['F3']

for m in range(1,13):
    print('mean in month {} = {}'.format(m, (F1.ix[F1.index.month == m] - F2.ix[F2.index.month == m]).mean()))

for m in range(1,13):
    print('mean in month {} = {}'.format(m, (F2.ix[F2.index.month == m] - F3.ix[F3.index.month == m]).mean()))

diffF1 = F1.values.ravel() - df[df.columns[0]].resample('M').mean()
diffF2 = F2.values.ravel() - df[df.columns[0]].resample('M').mean()
diffF3 = F3.values.ravel() - df[df.columns[0]].resample('M').mean()

for m in range(1,13):
    print('mean diff F1 from BSL in month {} = {}'.format(m, diffF1.ix[diffF1.index.month == m].mean() ))

for m in range(1,13):
    print('mean diff F2 from BSL in month {} = {}'.format(m, diffF2.ix[diffF2.index.month == m].mean() ))

for m in range(1,13):
    print('mean diff F3 from BSL in month {} = {}'.format(m, diffF3.ix[diffF3.index.month == m].mean() ))
