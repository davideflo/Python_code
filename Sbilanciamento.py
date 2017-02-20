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
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict
import datetime

today = datetime.datetime.now()
####################################################################################################
### @param: y1 and y2 are the years to be compared; y1 < y2 and y1 will bw taken as reference, unless it is a leap year
def SimilarDaysError(df, y1, y2):
    errors = []
    y = y1
    if y % 4 == 0:
        y = y2
    for m in range(1,13,1):
        dim = calendar.monthrange(y, m)[1]
        dfm = df.ix[df.index.month == m]
        dfm5 = dfm.ix[dfm.index.year == y]
        dfm6 = dfm.ix[dfm.index.year == y2]
        for d in range(1, dim, 1):
            ddfm5 = dfm5.ix[dfm5.index.day == d]
            ddfm6 = dfm6.ix[dfm6.index.day == d]
            if ddfm5.shape[0] == ddfm6.shape[0]:
                errors.extend(ddfm6['FABBISOGNO REALE'].values.ravel() - ddfm5['FABBISOGNO REALE'].values.ravel().tolist())
    return errors
####################################################################################################
def AddHolidaysDate(vd):
    
  ##### codifica numerica delle vacanze
  ## 1 Gennaio = 1, Epifania = 2
  ## Pasqua = 3, Pasquetta = 4
  ## 25 Aprile = 5, 1 Maggio = 6, 2 Giugno = 7,
  ## Ferragosto = 8, 1 Novembre = 9
  ## 8 Dicembre = 10, Natale = 11, S.Stefano = 12, S.Silvestro = 13
    holidays = 0
    pasquetta = [datetime.datetime(2015,4,6), datetime.datetime(2016,3,28)]
    pasqua = [datetime.datetime(2015,4,5), datetime.datetime(2016,3,27)]
  
    if vd.month == 1 and vd.day == 1:
        holidays = 1
    if vd.month  == 1 and vd.day == 6: 
        holidays = 1
    if vd.month  == 4 and vd.day == 25: 
        holidays = 1
    if vd.month  == 5 and vd.day == 1: 
        holidays = 1
    if vd.month  == 6 and vd.day == 2: 
        holidays = 1
    if vd.month  == 8 and vd.day == 15: 
        holidays = 1
    if vd.month  == 11 and vd.day == 1: 
        holidays = 1
    if vd.month  == 12 and vd.day == 8: 
        holidays = 1
    if vd.month  == 12 and vd.day == 25: 
        holidays = 1
    if vd.month  == 12 and vd.day == 26: 
        holidays = 1
    if vd.month  == 12 and vd.day == 31: 
        holidays = 1
    if vd in pasqua:
        holidays = 1
    if vd in pasquetta:
        holidays = 1
  
    return holidays
####################################################################################################
def GetMeanCurve(df, var):
    mc = OrderedDict()
    for y in [2015, 2016]:
        dfy = df[var].ix[df.index.year == y]
        for m in range(1,13,1):
            dfym = dfy.ix[dfy.index.month == m]
            Mean = []
            for h in range(24):
                dfymh = dfym.ix[dfym.index.hour == h].mean()
                Mean.append(dfymh)
            mc[str(m) + '_' + str(y)] = Mean
    mc = pd.DataFrame.from_dict(mc, orient = 'index')
    return mc
####################################################################################################
def MakeDatasetTS(df, meteo):
    dts = OrderedDict()
    #mc = GetMeanCurve(df, 'FABBISOGNO REALE')
    for i in df.index.tolist():
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        dts[i] = [wd, h, dy, Tmax, rain, wind, hol, df['FABBISOGNO REALE'].ix[i]]
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    return dts
####################################################################################################
def MakeDatasetTSCurve(df, meteo):
    dts = OrderedDict()
    for i in df.index.tolist():
        m = i.month
        y = i.year
        dfm = df.ix[df.index.month == m]
        dfmy = dfm.ix[dfm.index.year == y]
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        cm = GetMeanCurve(dfmy.ix[dfmy.index.date <= pd.to_datetime(i).date()],'FABBISOGNO REALE').dropna().values.ravel()
        ll = [wd, h, dy, Tmax, rain, wind, hol]
        ll.extend(cm.tolist())
        ll.extend([df['FABBISOGNO REALE'].ix[i]])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    return dts
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

#### fabbisogno 2009
sbil2009 = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento2009.xlsx')
nord2009 = sbil2009.ix[sbil2009['CODICE RUC'] == 'UC_DP1608_NORD']
nord2009.index = pd.date_range('2009-01-01', '2010-01-02', freq = 'H')[:nord2009.shape[0]]

#### difference between 2015 and 2009 since they were identical years (same days were on the same days)
diff = nord['FABBISOGNO REALE'].ix[nord.index.year == 2015].values.ravel() - nord2009['FABBISOGNO REALE'].values.ravel()

plt.figure()
plt.plot(diff)

##### experiment AdaBoost + Decision Trees 2015 to 2016
cnord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_CNOR']
cnord.index = pd.date_range('2015-01-01', '2017-01-02', freq = 'H')[:cnord.shape[0]]
fi5 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2015.xlsx')
fi5 = fi5.ix[:364].set_index(pd.date_range('2015-01-01', '2015-12-31', freq = 'D'))
fi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2016.xlsx')
fi6 = fi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))


DT5 = MakeDatasetTS(cnord.ix[cnord.index.year == 2015], fi5)
DT6 = MakeDatasetTS(cnord.ix[cnord.index.year == 2016], fi6)

regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 24),n_estimators=5000)

DT5s = DT5.sample(frac = 1).reset_index(drop = True)

x = DT5[DT5.columns[:7]]
y = DT5[DT5.columns[7]]
xs = DT5s[DT5s.columns[:7]]
ys = DT5s[DT5s.columns[7]]
x6 = DT6[DT6.columns[:7]]
y6 = DT6[DT6.columns[7]]

regr.fit(xs, ys)
yhat = regr.predict(x)

regrR2 = 1 - (np.sum((y - yhat)**2))/(np.sum((y - np.mean(y))**2))

plt.figure()
plt.plot(yhat, color = 'blue', marker = 'o')
plt.plot(y.values.ravel(), color = 'red')

plt.figure()
plt.plot(y - yhat)

yhat6 = regr.predict(x6)

regr6R2 = 1 - (np.sum((y6 - yhat6)**2))/(np.sum((y6 - np.mean(y6))**2))


plt.figure()
plt.plot(yhat6, color = 'navy', marker = 'o')
plt.plot(y6.values.ravel(), color = 'coral')

plt.figure()
plt.plot(y6 - yhat6)
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()


rfregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
rfregr.fit(xs, ys)
yhat = rfregr.predict(x)

regrR2 = 1 - (np.sum((y - yhat)**2))/(np.sum((y - np.mean(y))**2))

yhat6 = rfregr.predict(x6)
regr6R2 = 1 - (np.sum((y6 - yhat6)**2))/(np.sum((y6 - np.mean(y6))**2))

plt.figure()
plt.plot(yhat6, color = 'navy', marker = 'o')
plt.plot(y6.values.ravel(), color = 'coral')

plt.figure()
plt.plot(y6 - yhat6)
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].std()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].std()

mc = GetMeanCurve(cnord, 'FABBISOGNO REALE')

mc15 = mc.ix[mc.index[:12]]
mc16 = mc.ix[mc.index[12:]]

mc15.T.plot(legend = False)
mc16.T.plot(legend = False)
ydiff = mc16.reset_index(drop = True) - mc15.reset_index(drop = True)
ytdiff = fi6['Tmedia'].resample('M').mean().values.ravel() - fi5['Tmedia'].resample('M').mean().values.ravel()

plt.figure()
plt.plot(ytdiff)

plt.figure()
ydiff.T.plot(legend = False)
plt.axhline(y = 0)

fi = fi5.append(fi6)

DTC = MakeDatasetTSCurve(cnord, fi)