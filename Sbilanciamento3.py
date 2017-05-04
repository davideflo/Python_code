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
    All116 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1601.xlsx")
    All216 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1602.xlsx")
    All316 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1603.xlsx")
    All416 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1604.xlsx")
    All516 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1605.xlsx")
    All616 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1606.xlsx")
    All716 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1607.xlsx")
    All816 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1608.xlsx")
    All916 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1609.xlsx")
    All1016 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1610.xlsx")
    All1116 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_16011.xlsx")
    All1216 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP_1612.xlsx")
#    All117 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_01_2017.xlsx")
#    All217 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_02_2017.xlsx")
    All317 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_03_2017.xlsx")
    All417 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_04_2017.xlsx")
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
        drm = d.month
        dry = d.year
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
        elif drm == 2 and dry == 2017:
            All = All317
        elif drm == 3 and dry == 2017:
            All = All317
        elif drm == 4 and dry == 2017:
            All = All417
        else:
            pass
        pods = dbz["POD"].ix[dbz["Giorno"] == d].values.ravel().tolist()
        All2 = All.ix[All["Trattamento_01"] == 'O']
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
    df = df.ix[df.index.date >= datetime.date(2016,1,1)]
    for i in df.index.tolist():
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[10:34]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
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

