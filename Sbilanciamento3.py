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
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.tree import DecisionTreeRegressor
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
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16)]

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
def StartsDaylightSaving(vd):
    dls = 0
    DLS = [datetime.date(2016,10,30), datetime.date(2017,10,29)]
    if vd in DLS:
        dls = 1
    return dls
####################################################################################################
def EndsDaylightSaving(vd):
    dls = 0
    DLS = [datetime.date(2016,3,27), datetime.date(2017,3,26)]
    if vd in DLS:
        dls = 1
    return dls
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
def percentageConsumption(db, zona, di, df):
    dr = pd.date_range(di, df, freq = 'D')
    All1 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_2016.h5")
    All2 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_2017.h5")  
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
        drm = d.month
        strm = str(drm) if len(str(drm)) > 1 else "0" + str(drm)  
        dry = d.year
        if dry == 2017 and drm <= 3:
            strm = '03'
        
        if dry == 2016:
            All = All1
        else:
            All = All2
        pods = dbz["POD"].ix[dbz["Giorno"] == d].values.ravel().tolist()
        AllX = All.ix[All["Trattamento_"+ strm] == 1]
        totd = np.sum(np.nan_to_num([AllX["CONSUMO_TOT_" + strm].ix[y] for y in AllX.index if AllX["POD"].ix[y] in pods]))/1000
        #totd = All2["CONSUMO_TOT"].ix[All2["POD"].values.ravel() in pods].sum()
        tot = AllX["CONSUMO_TOT_" + strm].sum()/1000
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
    final = max(df.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    psample = percentageConsumption(db, zona, '2016-01-01', final_date)
    psample = psample.set_index(pd.date_range('2016-01-01', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    df = df.ix[df.index.date >= datetime.date(2016,1,3)]
    for i in df.index.tolist():
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
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
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], dls, edls])
        ll.extend(cmym.tolist())
        if df.ix[i].shape[0] > 1:
            y = df['MO [MWh]'].ix[i].sum()
        elif df.ix[i].shape[0] == 0:
            y = 0
        else:
            y = df['MO [MWh]'].ix[i]
        ll.extend([y])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    return dts
####################################################################################################
def MakeExtendedDatasetGivenPOD(db, meteo, pod):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    db = db.ix[db["POD"] == pod].reset_index(drop = True)
    dts = OrderedDict()
    start = str(db["Giorno"].ix[0])[:10]
    end =  str(db["Giorno"].ix[db.shape[0]-1])[:10]
#    if db["Giorno"].ix[db.shape[0]-1] > datetime.datetime(2017,3,31):
#        end = '2017-04-02'
    dr = pd.date_range(start, end, freq = 'H')
    for i in dr:
        #print i
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
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol])
        ll.extend(cmym.tolist())
        if db[str(h + 1)].ix[db["Giorno"] == i.date()].shape[0] > 0:
            ll.extend([db[str(h + 1)].ix[db["Giorno"] == i.date()].values[0]])
            dts[i] =  ll
        else:
            pass
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    return dts
####################################################################################################
def MakeForecastDataset(db, meteo, zona, time_delta = 1):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta + 1)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    psample = percentageConsumption(db, zona, '2017-05-19',final_date)
    psample = psample.set_index(pd.date_range('2017-05-19', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    dr = pd.date_range('2017-05-19', final_date, freq = 'H')
    for i in dr[2*24:dr.size-1]:
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], dls, edls])
        ll.extend(cmym.tolist())
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
    return dts
####################################################################################################
def MakeForecastPerPOD(db, meteo, pod, time_delta = 1):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    db = db.ix[db['POD'] == pod]
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    dts = OrderedDict()
    dr = pd.date_range('2017-04-01', final_date, freq = 'H')
    for i in dr:
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
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol])
        ll.extend(cmym.tolist())
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
    return dts
####################################################################################################
def MakeDatasetLongTerm(df, db, meteo, zona, td):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    final = max(df.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    psample = percentageConsumption(db, zona, '2016-01-01', final_date)
    psample = psample.set_index(pd.date_range('2016-01-01', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    df = df.ix[df.index.date >= (datetime.date(2016,1,1) + datetime.timedelta(days = td))]
    for i in df.index.tolist():
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], dls, edls])
        ll.extend(cmym.tolist())
        if df.ix[i].shape[0] > 1:
            y = df['MO [MWh]'].ix[i].sum()
        elif df.ix[i].shape[0] == 0:
            y = 0
        else:
            y = df['MO [MWh]'].ix[i]
        ll.extend([y])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    return dts
####################################################################################################
def MakeForecastLongTerm(db, meteo, zona, time_delta):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta + 1)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    psample = percentageConsumption(db, zona, '2017-05-20',final_date)
    psample = psample.set_index(pd.date_range('2017-05-20', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    dr = pd.date_range('2017-05-20', final_date, freq = 'H')
    for i in dr[time_delta*24:dr.size-1]:
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        td = time_delta
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], dls, edls])
        ll.extend(cmym.tolist())
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
    return dts
####################################################################################################
def getImbalance(error, prediction):
    ### error becomes a DataFrame with same index set as test
    tau = error.values.ravel()/prediction
    ERR = pd.DataFrame.from_dict({"Sbil": error.values.ravel().tolist(), "Sbil_perc": tau.tolist()})
    ERR = ERR.set_index(error.index)
    return ERR
####################################################################################################    
def EvaluateImbalance(val, imb, zona, pun):
    diz = OrderedDict()
    valzona = val.ix[val["CODICE RUC"] == "UC_DP1608_" + zona]
    valzona = valzona.set_index(pd.date_range("2015-01-01", "2018-01-02", freq = "H")[:valzona.shape[0]])
    for i in valzona.index:
        valsbil = 0
        valdiff = 0
        if i in pun.index:
            dp = pun[zona].ix[i]
            ### dentro la banda:
            if np.abs(imb["Sbil_perc"].ix[i]) <= 0.15: 
                if valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] > 0:
                    valsbil = np.abs(imb["Sbil"].ix[i]) * min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])
                    valdiff = -np.abs(imb["Sbil"].ix[i]) * (dp - min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i]))
                elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] < 0 and imb["Sbil"].ix[i] > 0:
                    valsbil = np.abs(imb["Sbil"].ix[i]) * max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i])
                    valdiff = -np.abs(imb["Sbil"].ix[i]) * (dp - max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] < 0:
                    valsbil = -np.abs(imb["Sbil"].ix[i]) * min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i])
                    valdiff = np.abs(imb["Sbil"].ix[i]) * (dp - min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i]))
                else:
                    valsbil = -np.abs(imb["Sbil"].ix[i]) * max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i])
                    valdiff = np.abs(imb["Sbil"].ix[i]) * (dp - max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                diz[i] = [valsbil, valdiff]
            ### fuori banda:
            else:
                PV = abs(-imb["Sbil"].ix[i]/(imb["Sbil_perc"].ix[i] - 1))
                inside = PV * 0.15
                outside = imb["Sbil"].ix[i] - inside
                if valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] > 0:
                    valsbil = np.abs(outside) * min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i]) +  np.abs(inside) * min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])
                    valdiff = -np.abs(outside) * (dp - min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])) + (-np.abs(inside) * (dp - min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])))
                elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] < 0 and imb["Sbil"].ix[i] > 0:
                    valsbil = np.abs(outside) * valzona["Pmgp_ven"].ix[i] + np.abs(inside) * max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i])
                    valdiff = -np.abs(outside) * (dp - valzona["Pmgp_ven"].ix[i]) + ( -np.abs(inside) * (dp - max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i])))
                elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] < 0:
                    valsbil = -np.abs(outside) * valzona["Pmgp_acq"].ix[i] + (-np.abs(inside) * min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i])) 
                    valdiff = np.abs(outside) * (dp - valzona["Pmgp_acq"].ix[i]) + np.abs(inside) * (dp - min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i]))
                else:
                    valsbil = -np.abs(outside) * max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]) + (-np.abs(inside) * max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                    valdiff = np.abs(outside) * (dp - max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i])) + np.abs(inside) * (dp - max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                diz[i] = [valsbil, valdiff]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [["Val_sbil", "gain"]]
    return diz
####################################################################################################
def podwiseForecast(db, meteo, zona):
    removed = []
    fdf = OrderedDict()
    podlist = list(set(db["POD"].ix[db["Area"] == zona].values.ravel().tolist()))
    for pod in podlist:
        if db.ix[db["POD"] == pod].shape[0] > 366:
            print 'Avanzamento: {} %'.format((podlist.index(pod) + 1)/len(podlist))
            DBP = MakeExtendedDatasetGivenPOD(DB, meteo, pod)
            
            train2 = DBP.ix[DBP.index.date < datetime.date(2017, 1, 1)]
            train = DBP.sample(frac = 1)
            test = DBP.ix[DBP.index.date > datetime.date(2016, 12, 31)]
            #test = test.ix[test.index.date < datetime.date(2017, 4, 12)]
            if test['y'].sum() < 10e-6:
                removed.append(pod)
                continue
            
            brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
            
            brf.fit(train[train.columns[:60]], train[train.columns[60]])
            yhat_train = brf.predict(train2[train2.columns[:60]])
            
            rfR2 = 1 - (np.sum((train2[train2.columns[60]] - yhat_train)**2))/(np.sum((train2[train2.columns[60]] - np.mean(train2[train2.columns[60]]))**2))
            print 'R-squared per pod {} = {}'.format(pod, rfR2)
            
            
            yhat_test = brf.predict(test[test.columns[:60]])
            rfR2_test = 1 - (np.sum((test[test.columns[60]] - yhat_test)**2))/(np.sum((test[test.columns[60]] - np.mean(test[test.columns[60]]))**2))
            print 'R-squared test per pod {} = {}'.format(pod, rfR2_test)
            
            if yhat_test.size < 2185:
                cc = np.concatenate([yhat_test, np.repeat(0, 2185 - yhat_test.size)]).tolist()
            else:
                cc = yhat_test.tolist()
            
            fdf[pod] = cc
        else:
            removed.append(pod)
    
    fdf = pd.DataFrame.from_dict(fdf, orient = "columns")
    fdf = fdf.set_index(pd.date_range('2017-01-01', '2017-04-02', freq = 'H'))
    return fdf, removed
#################################################################################################### 
def transform_df(df):
    diz = OrderedDict()
    dr = pd.date_range('2016-01-01', '2017-04-01', freq = 'D')
    for i in dr:
        dfi = df['MO [MWh]'].ix[df.index.date == i.date()]
        if int(dfi.shape[0]) == 24:
            diz[i.date()] = dfi.values.ravel().tolist()
        elif int(dfi.shape[0]) == 23:
            vec = np.repeat(0.0, 24)
            for h in range(24):
                if h != 2:
                    vec[h] = dfi.ix[dfi.index.hour == h]
            diz[i.date()] = vec.tolist()
        else:
            vec = np.repeat(0.0, 24)
            for h in range(24):
                if h == 2:
                    vec[h] = dfi.ix[dfi.index.hour == h].sum()
                else:
                    vec[h] = dfi.ix[dfi.index.hour == h].sum()
            diz[i.date()] = vec.tolist()
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']]
    return diz
#################################################################################################### 
def convertDates(vec):
    CD = vec.apply(lambda x: datetime.datetime(year = int(str(x)[6:10]), month = int(str(x)[3:5]), day = int(str(x)[:2]), hour = int(str(x)[11:13])))
    return CD
####################################################################################################
    
k2e = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregato_copia.xlsx", sheetname = 'Forecast k2E', skiprows = [0,1,2])
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


#dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari - 17_03.xlsm",
#                   skiprows = [0], sheetname = "Consumi base 24")

dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari-2017.xlsx")

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
###############################
cnord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_CNOR']
cnord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:cnord.shape[0]]
fi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2016.xlsx')
fi6 = fi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
fi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2017.xlsx')
fi7 = fi7.set_index(pd.date_range('2017-01-01', '2017-04-30', freq = 'D'))
fi = fi6.append(fi7)
###############################
csud = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_CSUD']
csud.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:csud.shape[0]]
ro6 = pd.read_excel('C:/Users/utente/Documents/PUN/Roma 2016.xlsx')
ro6 = ro6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
ro7 = pd.read_excel('C:/Users/utente/Documents/PUN/Fiumicino 2017.xlsx')
ro7 = ro7.set_index(pd.date_range('2017-01-01', '2017-04-30', freq = 'D'))
ro = ro6.append(ro7)
###############################
sud = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SUD']
sud.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:sud.shape[0]]
rc6 = pd.read_excel('C:/Users/utente/Documents/PUN/Bari 2016.xlsx')
rc6 = rc6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
rc7 = pd.read_excel('C:/Users/utente/Documents/PUN/Bari 2017.xlsx')
rc7 = rc7.set_index(pd.date_range('2017-01-01', '2017-04-30', freq = 'D'))
rc = rc6.append(rc7)
###############################
sici = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SICI']
sici.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:sici.shape[0]]
pa6 = pd.read_excel('C:/Users/utente/Documents/PUN/Palermo 2016.xlsx')
pa6 = pa6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
pa7 = pd.read_excel('C:/Users/utente/Documents/PUN/Palermo 2017.xlsx')
pa7 = pa7.set_index(pd.date_range('2017-01-01', '2017-04-30', freq = 'D'))
pa = pa6.append(pa7)
###############################
sard = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SARD']
sard.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:sard.shape[0]]
ca6 = pd.read_excel('C:/Users/utente/Documents/PUN/Cagliari 2016.xlsx')
ca6 = ca6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
ca7 = pd.read_excel('C:/Users/utente/Documents/PUN/Cagliari 2017.xlsx')
ca7 = ca7.set_index(pd.date_range('2017-01-01', '2017-04-30', freq = 'D'))
ca = ca6.append(ca7)

######### mean behaviour per month ############                 ####################################
cd = convertDates(nord['DATA RIFERIMENTO CORRISPETTIVO'])
nord = nord.set_index(cd.values)

cd = convertDates(cnord['DATA RIFERIMENTO CORRISPETTIVO'])
cnord = cnord.set_index(cd.values)

cd = convertDates(csud['DATA RIFERIMENTO CORRISPETTIVO'])
csud = csud.set_index(cd.values)

cd = convertDates(sud['DATA RIFERIMENTO CORRISPETTIVO'])
sud = sud.set_index(cd.values)

cd = convertDates(sici['DATA RIFERIMENTO CORRISPETTIVO'])
sici = sici.set_index(cd.values)

cd = convertDates(sard['DATA RIFERIMENTO CORRISPETTIVO'])
sard = sard.set_index(cd.values)

df_t = transform_df(nord)
df_t = df_t.set_index(pd.date_range('2016-01-01', '2017-04-01', freq = 'D'))
df_t.resample('M').mean().T.plot()

###############################################                 ####################################
###############################################
DBB1 = MakeExtendedDatasetWithSampleCurve(nord, DB, mi, "NORD")
DBB2 = MakeExtendedDatasetWithSampleCurve(cnord, DB, fi, "CNOR")
DBB3 = MakeExtendedDatasetWithSampleCurve(csud, DB, ro, "CSUD")
DBB4 = MakeExtendedDatasetWithSampleCurve(sud, DB, rc, "SUD")
DBB5 = MakeExtendedDatasetWithSampleCurve(sici, DB, pa, "SICI")
DBB6 = MakeExtendedDatasetWithSampleCurve(sard, DB, ca, "SARD")

DBB1.to_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5', 'nord')
DBB2.to_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5', 'cnord')
DBB3.to_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5', 'csud')
DBB4.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5', 'sud')
DBB5.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5', 'sici')
DBB6.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5', 'sard')
###############################################
###############################################
DBB = MakeDatasetLongTerm(nord, DB, mi, "NORD", 5)
DBB = MakeDatasetLongTerm(cnord, DB, fi, "CNOR",5)
DBB = MakeDatasetLongTerm(csud, DB, ro, "CSUD",5)
DBB = MakeDatasetLongTerm(sud, DB, rc, "SUD",5)
DBB = MakeDatasetLongTerm(sici, DB, pa, "SICI",5)
DBB = MakeDatasetLongTerm(sard, DB, ca, "SARD",5)

DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/nord5.h5', 'nord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord5.h5', 'cnord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/csud5.h5', 'csud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sud5.h5', 'sud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sici5.h5', 'sici')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sard5.h5', 'sard')
####################################################
####################################################

DBB = MakeExtendedDatasetWithSampleCurve(nord, DB, mi, "NORD")
DBB = MakeExtendedDatasetWithSampleCurve(cnord, DB, fi, "CNOR")
DBB = MakeExtendedDatasetWithSampleCurve(csud, DB, ro, "CSUD")
DBB = MakeExtendedDatasetWithSampleCurve(sud, DB, rc, "SUD")
DBB = MakeExtendedDatasetWithSampleCurve(sici, DB, pa, "SICI")
DBB = MakeExtendedDatasetWithSampleCurve(sard, DB, ca, "SARD")

DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5', 'nord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5', 'cnord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5', 'csud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5', 'sud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5', 'sici')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5', 'sard')
#### check on holidays ####
feste = DBB.ix[DBB['holiday'] == 1].index
feste = set(feste.tolist())


train2 = DBB.ix[DBB.index.date < datetime.date(2017, 1, 1)]
train = DBB.sample(frac = 1)
test = DBB.ix[DBB.index.date > datetime.date(2016, 12, 31)]
test = test.ix[test.index.date < datetime.date(2017, 4, 12)]


train = train.ix[train.index.date != datetime.date(2016,3,27)]
train = train.ix[train.index.date != datetime.date(2016,10,30)]
train = train.ix[train.index.date != datetime.date(2017,3,26)]

###### check to see if the initial error is due to the model or some other pecularity
###### happened in January 2017
train2 = DBB.ix[DBB.index.date < datetime.date(2017, 2, 1)]
train = DBB.sample(frac = 1)
test = DBB.ix[DBB.index.date > datetime.date(2017, 1, 31)]
test = test.ix[test.index.date < datetime.date(2017, 4, 12)]


### It seems the performances depend on the initial permutation of the trainingset
### Google: cross validation with random forest sklearn

#ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(train[train.columns[:63]], train[train.columns[63]])
yhat_train = brf.predict(train2[train2.columns[:63]])

rfR2 = 1 - (np.sum((train2[train2.columns[63]] - yhat_train)**2))/(np.sum((train2[train2.columns[63]] - np.mean(train2[train2.columns[63]]))**2))
print rfR2


yhat_test = brf.predict(test[test.columns[:63]])
rfR2_test = 1 - (np.sum((test[test.columns[63]] - yhat_test)**2))/(np.sum((test[test.columns[63]] - np.mean(test[test.columns[63]]))**2))
print rfR2_test

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = '+')

#### graphical comparison with k2e
nk2e = k2e["NORD"]/1000
tnk2e = nk2e.ix[nk2e.index < datetime.datetime(2017,4,1)]
cnk2e = k2e["CNOR"]/1000
tcnk2e = cnk2e.ix[cnk2e.index < datetime.datetime(2017,4,1)]
csk2e = k2e["CSUD"]/1000
tcsk2e = csk2e.ix[csk2e.index < datetime.datetime(2017,4,1)]
sk2e = k2e["SUD"]/1000
tsk2e = sk2e.ix[sk2e.index < datetime.datetime(2017,4,1)]
sik2e = k2e["SICI"]/1000
tsik2e = sik2e.ix[sik2e.index < datetime.datetime(2017,4,1)]
sak2e = k2e["SARD"]/1000
tsak2e = sak2e.ix[sak2e.index < datetime.datetime(2017,4,1)]


plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tnk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona NORD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tcnk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona CNORD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tcsk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona CSUD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tsk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona SUD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tsik2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona SICI")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tsak2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona SARD")
##############################


error = test[test.columns[63]].values.ravel() - yhat_test    


k2e_error = test[test.columns[63]].values.ravel() - tnk2e.values.ravel()[:tnk2e.values.ravel().size - 1] 
k2e_error = test[test.columns[63]].values.ravel() - tcnk2e.values.ravel()[:tcnk2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tcsk2e.values.ravel()[:tcsk2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tsk2e.values.ravel()[:tsk2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tsik2e.values.ravel()[:tsik2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tsak2e.values.ravel()[:tsak2e.values.ravel().size - 1]     

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

maek2 = k2e_error/test[test.columns[63]].values.ravel()
mae = error/test[test.columns[63]].values.ravel()

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

###### ABSOLUTE ERRORS ########
amaek2 = np.abs(k2e_error)/test[test.columns[63]].values.ravel()
amae = np.abs(error)/test[test.columns[63]].values.ravel()

print np.mean(amaek2)
print np.median(amaek2)
print np.std(amaek2)
print np.max(amaek2)
print np.min(amaek2)

print np.mean(amae)
print np.median(amae)
print np.std(amae)
print np.max(amae)
print np.min(amae)
##############################

print scipy.stats.mstats.mquantiles(maek2, prob = [0.025, 0.975])
print scipy.stats.mstats.mquantiles(mae, prob = [0.025, 0.975])

zona = "NORD"

plt.figure()
plt.plot(maek2, color = 'green')
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)
plt.title("Errore percentuale forecast K2E zona " + zona)
plt.figure()
plt.plot(mae, color = 'blue')
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)
plt.title("Errore percentuale forecast AXO zona " + zona)


np.corrcoef(maek2, mae)

plt.figure()
plotting.autocorrelation_plot(maek2, color = 'green')
plt.title("Autocorrelazione errore percentuale forecast K2E zona " + zona)
plt.figure()
plotting.autocorrelation_plot(mae)
plt.title("Autocorrelazione errore percentuale forecast AXO zona " + zona)


plt.figure() 
plt.hist(maek2, bins = 20, color = 'green')   
plt.figure() 
plt.hist(mae, bins = 20)   

print np.where(np.abs(maek2) >= 0.15)[0].size/maek2.size
print np.where(np.abs(mae) >= 0.15)[0].size/mae.size


k2e_error = pd.DataFrame.from_dict({"error":k2e_error}).set_index(test.index)
axo_error = pd.DataFrame.from_dict({"error":error}).set_index(test.index)

K2E = getImbalance(k2e_error, tnk2e.values.ravel()[:tnk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tcnk2e.values.ravel()[:tcnk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tcsk2e.values.ravel()[:tcsk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tsk2e.values.ravel()[:tsk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tsik2e.values.ravel()[:tsik2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tsak2e.values.ravel()[:tsak2e.values.ravel().size - 1])


AXO = getImbalance(axo_error, yhat_test)

val = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/valorizzazione_sbilanciamento.xlsx")
pun = pd.read_excel("C:/Users/utente/Documents/DB_Borse_Elettriche_PER MI_17_conMacro - Copy.xlsm", sheetname = "DB_Dati")
pun = pun.set_index(pd.date_range("2017-01-01", "2018-01-02", freq = "H")[:pun.shape[0]])
pun = pun.ix[pun.index.date <= today.date()]
pun = pun[pun.columns[[12,16,19,24,27,28,30]]]
pun.columns = [["PUN","CNOR","CSUD","NORD","SARD","SICI","SUD"]]

##### Test for valorization ####
spn = nord["SBILANCIAMENTO FISICO [MWh]"].values.ravel()/nord["PV [MWh]"].values.ravel()
prova = pd.DataFrame.from_dict({"Sbil": nord["SBILANCIAMENTO FISICO [MWh]"].values.ravel().tolist(),
                                "Sbil_perc": spn.tolist()})
prova = prova.set_index(nord.index)                                
VS = EvaluateImbalance(val, prova, "NORD", pun)                                
#####                                

imb_k2e = EvaluateImbalance(val, K2E, "NORD", pun)
imb_k2e = EvaluateImbalance(val, K2E, "CNOR", pun)
imb_k2e = EvaluateImbalance(val, K2E, "CSUD", pun)
imb_k2e = EvaluateImbalance(val, K2E, "SUD", pun)
imb_k2e = EvaluateImbalance(val, K2E, "SICI", pun)
imb_k2e = EvaluateImbalance(val, K2E, "SARD", pun)


imb_k2e.mean()
imb_k2e.sum()
imb_k2e.to_excel("C:/Users/utente/Documents/Sbilanciamento/Backtesting/sbilanciamento_k2e_" + zona + ".xlsx")

imb_axo = EvaluateImbalance(val, AXO, "NORD", pun)
imb_axo = EvaluateImbalance(val, AXO, "CNOR", pun)
imb_axo = EvaluateImbalance(val, AXO, "CSUD", pun)
imb_axo = EvaluateImbalance(val, AXO, "SUD", pun)
imb_axo = EvaluateImbalance(val, AXO, "SICI", pun)
imb_axo = EvaluateImbalance(val, AXO, "SARD", pun)


imb_axo.mean()
imb_axo.sum()
imb_axo.to_excel("C:/Users/utente/Documents/Sbilanciamento/Backtesting/sbilanciamento_axo_" + zona + ".xlsx")

#######


###### podwise forecast

DBP = MakeExtendedDatasetGivenPOD(DB, mi, "IT001E00123781")

train2 = DBP.ix[DBP.index.date < datetime.date(2017, 1, 1)]
train = DBP.sample(frac = 1)
test = DBP.ix[DBP.index.date > datetime.date(2016, 12, 31)]
test = test.ix[test.index.date < datetime.date(2017, 4, 12)]


### It seems the performances depend on the initial permutatio of the trainingset
### Google: cross validation with random forest sklearn

#ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(train[train.columns[:60]], train[train.columns[60]])
yhat_train = brf.predict(train2[train2.columns[:60]])

rfR2 = 1 - (np.sum((train2[train2.columns[60]] - yhat_train)**2))/(np.sum((train2[train2.columns[60]] - np.mean(train2[train2.columns[60]]))**2))
print rfR2


yhat_test = brf.predict(test[test.columns[:60]])
rfR2_test = 1 - (np.sum((test[test.columns[60]] - yhat_test)**2))/(np.sum((test[test.columns[60]] - np.mean(test[test.columns[60]]))**2))
print rfR2_test

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test[test.columns[60]].values.ravel(), color = 'red', marker = '+')

errpod = test[test.columns[60]].values.ravel() - yhat_test
MAE = errpod/test[test.columns[60]].values.ravel()

plt.figure()
plt.plot(errpod, color = 'orange')
plt.figure()
plt.plot(MAE, color = 'navy')
plt.axhline(y = 0.15, color = 'red')
plt.axhline(y = -0.15, color = 'red')


forecast_per_pod = podwiseForecast(DB, mi, "NORD")
forecast_per_pod = podwiseForecast(DB, fi, "CNOR")
forecast_per_pod, rem = podwiseForecast(DB, ro, "CSUD")
fdf = podwiseForecast(DB, rc, "SUD")



fdf = fdf[0]
rem = fdf[1]

csud_ = DB.ix[DB["Area"] == "CSUD"]
CSUD = pd.DataFrame()
for pod in forecast_per_pod.columns:
    CSUD = CSUD.append(DB.ix[DB["POD"] == pod])

csud_gg = []
for d in pd.date_range('2017-01-01', '2017-03-31', freq = 'D'):
    #print d.month
    #print CSUD.ix[CSUD["Giorno"] == d].shape
    csud_gg.extend(CSUD.ix[CSUD["Giorno"] == d].sum())


err = forecast_per_pod.sum(axis = 1).values[:2160] - np.array(csud_gg)

csud_gg[2018] = 1
err[2018] = 1

R2_glob = 1 - (np.sum((err)**2))/(np.sum((np.array(csud_gg) - np.mean(np.array(csud_gg)))**2))
print R2_glob

plt.figure()
plt.plot(np.array(csud_gg), marker = '8', color = 'green')
plt.plot(forecast_per_pod.sum(axis = 1).values, marker = 'o', color = 'magenta')

errmae = err/np.array(csud_gg)
errmae[2018] = 0

print np.mean(err)
print np.median(err)
print np.std(err)
print np.mean(errmae)
print np.median(errmae)
print np.std(errmae)
print np.max(errmae)
print np.min(errmae)

plt.figure()
plt.plot(errmae)
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)

################################
######## Test on May 2017 ######
################################

DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5')


mi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Milano.xlsx')
mi2017 = mi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
fi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Firenze.xlsx')
fi2017 = fi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ro2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Roma.xlsx')
ro2017 = ro2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ba2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Bari.xlsx')
ba2017 = ba2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
pa2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Palermo.xlsx')
pa2017 = pa2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ca2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Cagliari.xlsx')
ca2017 = ca2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]

############################
mi52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Milano.xlsx')
mi52017 = mi52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
fi52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Firenze.xlsx')
fi52017 = fi52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ro52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Roma.xlsx')
ro52017 = ro52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ba52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Bari.xlsx')
ba52017 = ba52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
pa52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Palermo.xlsx')
pa52017 = pa52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ca52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Cagliari.xlsx')
ca52017 = ca52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]

DBT = MakeForecastLongTerm(DB, mi52017, "NORD",5)
DBT = MakeForecastLongTerm(DB, fi52017, "CNOR",5)
DBT = MakeForecastLongTerm(DB, ro52017, "CSUD",5)
DBT = MakeForecastLongTerm(DB, ba52017, "SUD",5)
DBT = MakeForecastLongTerm(DB, pa52017, "SICI",5)
DBT = MakeForecastLongTerm(DB, ca52017, "SARD",5)
###########################


DBH = DBB.ix[DBB['holiday'] == 1]
DBN = DBB.ix[DBB['holiday'] == 0]

DBtrain = DBB.sample(frac = 1)

DBHtrain = DBH.sample(frac = 1)
DBNtrain = DBN.sample(frac = 1)


DBT = MakeForecastDataset(DB, mi2017, "NORD")
DBT = MakeForecastDataset(DB, fi2017, "CNOR")
DBT = MakeForecastDataset(DB, ro2017, "CSUD")
DBT = MakeForecastDataset(DB, ba2017, "SUD")
DBT = MakeForecastDataset(DB, pa2017, "SICI")
DBT = MakeForecastDataset(DB, ca2017, "SARD")



DBTH =  DBT.ix[DBT['holiday'] == 1]
DBTN =  DBT.ix[DBT['holiday'] == 0]

################################ Separation holiday/non-holidays ###################################
brfh = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brfh.fit(DBHtrain[DBHtrain.columns[:63]], DBHtrain[DBHtrain.columns[63]])
yhat_train_h = brfh.predict(DBHtrain[DBHtrain.columns[:63]])

rfR2H = 1 - (np.sum((DBHtrain[DBHtrain.columns[63]] - yhat_train_h)**2))/(np.sum((DBHtrain[DBHtrain.columns[63]] - np.mean(DBHtrain[DBHtrain.columns[63]]))**2))
print rfR2H

yhat_test_h = brfh.predict(DBTH)

yth = pd.DataFrame.from_dict({'pred': yhat_test_h.tolist()})
yth = yth.set_index(DBTH.index)
yth.plot(color = 'orchid')

yth.to_excel("SUD_previsione_2017-06-02.xlsx")
##########################
brfn = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

params = {'n_estimators': 500, 'max_depth': 48, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'criterion': 'friedman_mse'}
gbm = GradientBoostingRegressor(**params)

gbm.fit(DBNtrain[DBNtrain.columns[:63]], DBNtrain['y'])
mse = mean_squared_error(DBNtrain[DBNtrain.columns[63]], gbm.predict(DBNtrain[DBNtrain.columns[:63]]))
yg_train_n = gbm.predict(DBNtrain[DBNtrain.columns[:63]])


brfn.fit(DBNtrain[DBNtrain.columns[:63]], DBNtrain['y'])
yhat_train_n = brfn.predict(DBNtrain[DBNtrain.columns[:63]])

rfR2n = 1 - (np.sum((DBNtrain[DBNtrain.columns[63]] - yhat_train_n)**2))/(np.sum((DBNtrain[DBNtrain.columns[63]] - np.mean(DBNtrain[DBNtrain.columns[63]]))**2))
print rfR2n

yhat_test_n = brfn.predict(DBTN)

ytn = pd.DataFrame.from_dict({'pred': yhat_test_n.tolist()})
ytn = ytn.set_index(DBTN.index)
ytn.plot(color = 'dimgrey')


ytn.to_excel("SUD_previsione_2017-05-26.xlsx")
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

bfr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(DBtrain[DBtrain.columns[:61]], DBtrain[DBtrain.columns[61]])
yhat_train = brf.predict(DBtrain[DBtrain.columns[:61]])

rfR2 = 1 - (np.sum((DBtrain[DBtrain.columns[61]] - yhat_train)**2))/(np.sum((DBtrain[DBtrain.columns[61]] - np.mean(DBtrain[DBtrain.columns[61]]))**2))
print rfR2

yhat_test = brf.predict(DBT)

##### Feature importance ####
importance = brf.feature_importances_
importance = pd.DataFrame(importance, index=DBtrain.columns[:61], columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_ for tree in brf.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

fig, ax = plt.subplots()
Ax = ax.bar(x, y, yerr=yerr, align="center")
Ax.set_xticklabels(DBtrain.columns)
####################################


plt.figure()
plt.plot(yhat_test, marker = '8', color = 'red')
plt.figure()
plt.plot(yhat_test, marker = '8', color = 'magenta')
plt.title('AdaBoost + RF')
plt.figure()
plt.plot(DBB['y'].values.ravel())


DBT['holiday'].ix[datetime.date(2017,4,16)]

pd.DataFrame.from_dict({'previsione': yhat_test[-24:].tolist()}).to_excel('C:/Users/utente/Documents/previsione_2017-05-17.xlsx')

Y = DBB['y'].astype(float)#.values.ravel()
U = np.repeat(0.0, Y.size)
U = U + Y
Y = []
for i in range(DBB.shape[0]):
    if DBB['y'].ix[i].size == 1:
        Y.append(float(DBB['y'].ix[i]))
    else:
        Y.append(float(DBB['y'].ix[i].sum()))

plt.figure()
plt.plot(Y[2831:3576])
plt.axvline(x = 2831, color = 'black',  linewidth=2.0)
plt.axvline(x = 3575, color = 'black',  linewidth=2.0)
