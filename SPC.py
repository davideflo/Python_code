# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:58:01 2016

@author: d_floriello

Self-PUN-computizer
"""


import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime
import statsmodels.api

###############################################################################
def divide_in_days(CMP):
    
    day = OrderedDict()
    
    lun = [] ## 0
    mar = []
    mer = []
    gio = []
    ven = []
    sab = []
    dom = [] ## 6
    
    for i in range(CMP.shape[0]):    
        dt = datetime.date(int(str(CMP.index[i])[:4]),int(str(CMP.index[i])[5:7]),int(str(CMP.index[i])[8:10]))
        if dt.weekday() == 0:
            lun.append(CMP.ix[i].values[0])
        elif dt.weekday() == 1:
            mar.append(CMP.ix[i].values[0])
        elif dt.weekday() == 2:
            mer.append(CMP.ix[i].values[0])
        elif dt.weekday() == 3:
            gio.append(CMP.ix[i].values[0])
        elif dt.weekday() == 4:
            ven.append(CMP.ix[i].values[0])
        elif dt.weekday() == 5:
            sab.append(CMP.ix[i].values[0])
        else:
            dom.append(CMP.ix[i].values[0])
    
    szs = [len(lun), len(mar), len(mer), len(gio), len(ven), len(sab), len(dom)]
    M = max(szs)
    
    if len(lun) < M:
        lun.append(np.nan)
    if len(mar) < M:
        mar.append(np.nan)
    if len(mer) < M:
        mer.append(np.nan)
    if len(gio) < M:
        gio.append(np.nan)
    if len(ven) < M:
        ven.append(np.nan)
    if len(sab) < M:
        sab.append(np.nan)
    if len(dom) < M:
        dom.append(np.nan)
            
    day['lun'] = lun        
    day['mar'] = mar        
    day['mer'] = mer        
    day['gio'] = gio        
    day['ven'] = ven        
    day['sab'] = sab        
    day['dom'] = dom        
    
    DBD = pd.DataFrame.from_dict(day)

    return DBD
###############################################################################
def de_trend(df, trend):
    dt = []
    yh = trend(np.linspace(0,df.shape[0],df.shape[0]))
    for i in range(df.shape[0]):
        mon = yh[(df.index.month[i] - 1)]
        print('month {} and correction {}'.format(df.index.month[i],mon))
        dt.append(df.ix[i] - mon)
    return pd.DataFrame(dt)
###############################################################################
def remainderizer(df):
    ### comupe trend in months:
    diz = OrderedDict()
    dow = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom'] 
    mp = np.array(df.resample('M').mean())
    dt = []
    for i in range(df.shape[0]):
        mon = mp[(df.index.month[i] - 1)]
        print('month {} and correction {}'.format(df.index.month[i],mon))
        dt.append(df.ix[i] - mon)
    dt = pd.DataFrame(dt).set_index(df.index)
    ### remove monthly seasonality:
    MONTHS = np.unique(dt.index.month)
    des = []
    for m in MONTHS:
        lm = []
        loc_dt = dt.ix[dt.index.month == m]
        dd = divide_in_days(loc_dt)
        diz[str(m)+'_mean'] = dd.mean()
        diz[str(m)+'_std'] = dd.std()
        for j in range(loc_dt.shape[0]):
            die = datetime.date(int(str(loc_dt.index[j])[:4]),int(str(loc_dt.index[j])[5:7]),int(str(loc_dt.index[j])[8:10]))
            giorno = die.weekday()
            x = (loc_dt.ix[j] - dd[dow[giorno]].mean())/dd[dow[giorno]].std()
            lm.append(x)
        des.append(lm)
    flattened_des = [item for sublist in des for item in sublist]
    rem = pd.DataFrame(flattened_des)
    seas = pd.DataFrame.from_dict(diz).set_index([dow])
    return mp, seas, rem
###############################################################################
def Forecast_(pun, year, month, day):
    ts = np.array(pun)
    dow = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom'] 
    dt = datetime.datetime(year, month, day)
    mm, sea, remn = remainderizer(pun)
    arma = statsmodels.api.tsa.ARMA(remn.values.ravel(), (1,0)).fit()
    resid = remn.values.ravel() - arma.predict()
    pred = arma.predict(start = ts.size, end = ts.size)
    forecasted = mm[-1] + sea[str(month)+'_mean'].ix[dow[dt.weekday()]] + pred
    #### sampled sigma is a bit overestimated
    sigma_hat = np.std(pun.ix[pun.index.month == 10] - mm[-1]) + sea[str(month)+'_std'].ix[dow[dt.weekday()]] + np.std(resid)
    return (forecasted - 2*sigma_hat, forecasted - sigma_hat, forecasted, forecasted + sigma_hat, forecasted + 2*sigma_hat)
###############################################################################
