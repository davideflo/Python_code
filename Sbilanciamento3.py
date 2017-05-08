# -*- coding: utf-8 -*-
"""
Created on Thu May 04 09:24:18 2017

@author: utente

Sbilanciamento 3 -- BACKTESTING --
"""

from __future__ import division
import pandas as pd
from pandas.tools import plotting
import numpy as np
import matplotlib.pyplot as plt
import calendar
import scipy
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict
import datetime
#from statsmodels.tsa.stattools import adfuller

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
    pasquetta = [datetime.datetime(2015,4,6), datetime.datetime(2016,3,28), datetime.datetime(2017,4,17)]
    pasqua = [datetime.datetime(2015,4,5), datetime.datetime(2016,3,27), datetime.datetime(2017,4,16)]
  
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
####################################################################################################
def percentageConsumption(db, zona):
    dr = pd.date_range('2016-01-01', '2017-04-11', freq = 'D')
    All116 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1601.xlsm", sheetname = "CRPP")
    All216 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1602.xlsm", sheetname = "CRPP")
    All316 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1603.xlsm", sheetname = "CRPP")
    All416 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1604.xlsm", sheetname = "CRPP")
    All516 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1605.xlsm", sheetname = "CRPP")
    All616 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1606.xlsm", sheetname = "CRPP")
    All716 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1607.xlsm", sheetname = "CRPP")
    All816 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1608.xlsm", sheetname = "CRPP")
    All916 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1609.xlsm", sheetname = "CRPP")
    All1016 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1610.xlsm", sheetname = "CRPP")
    All1116 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1611.xlsm", sheetname = "CRPP")
    All1216 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1612.xlsm", sheetname = "CRPP")
#    All117 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_01_2017.xlsx")
#    All217 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_02_2017.xlsx")
    All317 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_03_2017.xlsx")
    All417 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_04_2017.xlsx")
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
        drm = d.month
        dry = d.year
#        print drm
#        print dry
#        print '###'
        strm = str(drm) if len(str(drm)) > 1 else "0" + str(drm)  
        if drm == 1 and dry == 2016:
            All = All116
        elif drm == 2 and dry == 2016:
            All = All216
        elif drm == 3 and dry == 2016:
            All = All316
        elif drm == 4 and dry == 2016:
            All = All416
        elif drm == 5 and dry == 2016:
            All = All516
        elif drm == 6 and dry == 2016:
            All = All616
        elif drm == 7 and dry == 2016:
            All = All716
        elif drm == 8 and dry == 2016:
            All = All816
        elif drm == 9 and dry == 2016:
            All = All916
        elif drm == 10 and dry == 2016:
            All = All1016
        elif drm == 11 and dry == 2016:
            All = All1116
        elif drm == 12 and dry == 2016:
            All = All1216
        elif drm == 1 and dry == 2017:
            All = All317
            strm = '03'
        elif drm == 2 and dry == 2017:
            All = All317
            strm = '03'
        elif drm == 3 and dry == 2017:
            All = All317
        elif drm == 4 and dry == 2017:
            All = All417
        else:
            pass
        pods = dbz["POD"].ix[dbz["Giorno"] == d].values.ravel().tolist()
        All2 = All.ix[All["Trattamento_"+ strm] == 'O']
        totd = np.sum(np.nan_to_num([All2["CONSUMO_TOT"].ix[y] for y in All2.index if All2["POD"].ix[y] in pods]))/1000
        #totd = All2["CONSUMO_TOT"].ix[All2["POD"].values.ravel() in pods].sum()
        tot = All2["CONSUMO_TOT"].sum()/1000
        p = totd/tot
        diz[d] = [p]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    return diz
####################################################################################################
def MakeExtendedDatasetWithSampleCurve(df, db, meteo, zona):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    psample = percentageConsumption(db, zona)
    psample = psample.set_index(pd.date_range('2016-01-01', '2017-04-11', freq = 'D'))
    dts = OrderedDict()
    df = df.ix[df.index.date >= datetime.date(2016,1,3)]
    for i in df.index.tolist():
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == i.date()]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0]])
        ll.extend(cmym.tolist())
        ll.extend([df['MO [MWh]'].ix[i]])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    return dts
####################################################################################################
k2e = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregato_copia.xlsx", sheetname = 'Delta ORARI', skiprows = [0,1])
k2e = k2e.set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:k2e.shape[0]])
#db = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_2016_noperd.xlsx", converters = {'1': str, '2': str, '3': str,
#                                                                                                 '4': str, '5': str, '6': str,
#                '7': str, '8': str, '9': str, '10': str, '11': str, '12': str,
#                '13': str, '14': str, '15': str, '16': str, '17': str, '18': str,
#                '19': str, '20': str, '21': str, '22': str, '23': str, '24': str, '25': str} )
            
db = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_2016_noperd.h5")

db.columns = [["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24", "25"]]

db = db[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]


dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari - 17_03.xlsm",
                   skiprows = [0], sheetname = "Consumi base 24")

dt.columns = [str(i) for i in dt.columns]

dt = dt[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]

DB = db.append(dt, ignore_index = True) 

sbil = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento2.xlsx')


nord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_NORD']
nord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:nord.shape[0]]
mi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2016.xlsx')
mi6 = mi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
mi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2017.xlsx')
mi7 = mi7.set_index(pd.date_range('2017-01-01', '2017-04-30', freq = 'D'))
mi = mi6.append(mi7)


DBB = MakeExtendedDatasetWithSampleCurve(nord, DB, mi, "NORD")



train2 = DBB.ix[DBB.index.date < datetime.date(2017, 1, 1)]
train = DBB.sample(frac = 1)
test = DBB.ix[DBB.index.date > datetime.date(2016, 12, 31)]
test = test.ix[test.index.date < datetime.date(2017, 3, 1)]


ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(train[train.columns[:61]], train[train.columns[61]])
yhat_train = brf.predict(train2[train2.columns[:61]])

rfR2 = 1 - (np.sum((train2[train2.columns[61]] - yhat_train)**2))/(np.sum((train2[train2.columns[61]] - np.mean(train2[train2.columns[61]]))**2))
print rfR2


yhat_test = brf.predict(test[test.columns[:61]])
rfR2_test = 1 - (np.sum((test[test.columns[61]] - yhat_test)**2))/(np.sum((test[test.columns[61]] - np.mean(test[test.columns[61]]))**2))
print rfR2_test

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test[test.columns[61]].values.ravel(), color = 'red', marker = '+')

#### graphical comparison with k2e
nk2e = k2e["NORD"]/1000
tnk2e = nk2e.ix[nk2e.index < datetime.datetime(2017,3,1)]


plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[61]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tnk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')

error = test[test.columns[61]].values.ravel() - yhat_test    


k2e_error = test[test.columns[61]].values.ravel() - tnk2e.values.ravel()     

print np.mean(k2e_error)
print np.median(k2e_error)
print np.std(k2e_error)

print np.mean(error)
print np.median(error)
print np.std(error)

plt.figure() 
plt.hist(k2e_error, bins = 20, color = 'green')   
plt.figure() 
plt.hist(error, bins = 20)   

maek2 = k2e_error/test[test.columns[61]].values.ravel()
mae = error/test[test.columns[61]].values.ravel()

print np.mean(maek2)
print np.median(maek2)
print np.std(maek2)
print np.max(maek2)
print np.min(maek2)

print np.mean(mae)
print np.median(mae)
print np.std(mae)
print np.max(mae)
print np.min(mae)

print scipy.stats.mstats.mquantiles(maek2, prob = [0.025, 0.975])
print scipy.stats.mstats.mquantiles(mae, prob = [0.025, 0.975])

plt.figure()
plt.plot(maek2, color = 'green')
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)
plt.figure()
plt.plot(mae, color = 'blue')
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)

np.corrcoef(maek2, mae)

plt.figure()
plotting.autocorrelation_plot(maek2, color = 'green')
plt.figure()
plotting.autocorrelation_plot(mae)

plt.figure() 
plt.hist(maek2, bins = 20, color = 'green')   
plt.figure() 
plt.hist(mae, bins = 20)   

print np.where(np.abs(maek2) >= 0.15)[0].size/maek2.size
print np.where(np.abs(mae) >= 0.15)[0].size/mae.size






