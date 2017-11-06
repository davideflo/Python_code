# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 11:13:57 2017

@author: utente

Quoting -- Ricalendarizer --
"""

import pandas as pd
import numpy as np
import datetime
import os
import calendar
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import scipy

####################################################################################################
def AddHolidaysDate(vd):
    
  ##### codifica numerica delle vacanze
  ## 1 Gennaio = 1, Epifania = 2
  ## Pasqua = 3, Pasquetta = 4
  ## 25 Aprile = 5, 1 Maggio = 6, 2 Giugno = 7,
  ## Ferragosto = 8, 1 Novembre = 9
  ## 8 Dicembre = 10, Natale = 11, S.Stefano = 12, S.Silvestro = 13
    holidays = 0
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17),datetime.date(2018,4,1)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16),datetime.date(2018,4,2)]

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
def GetRicDateGeneral(dtf):
### Get the "most similar" corresponding date, given history   
    strm = str(dtf.month) if len(str(dtf.month)) > 1 else "0" + str(dtf.month)

    monthrange = pd.date_range(str(dtf.year) + '-' + strm + '-01', str(dtf.year) + '-' + strm + '-' + str(dtf.day), freq = 'D')
    todow = map(lambda date: date.weekday(), monthrange)

    monthrangey1 = pd.date_range(str(dtf.year - 1) + '-' + strm + '-01', str(dtf.year - 1) + '-' + strm + '-' + str(calendar.monthrange(dtf.year - 1, dtf.month)[1]), freq = 'D')
    todowy1 = map(lambda date: date.weekday(), monthrangey1)
    
    dow = dtf.weekday()
    dow_counter = np.where(np.array(todow) == dow)[0].size - 1
    
    dow_countery11 = [i for i, x in enumerate(todowy1) if x == dow]

    
    if AddHolidaysDate(dtf) == 0:
        try:
            dow_countery1 = dow_countery11[dow_counter]
        except:
            dow_countery1 = dow_countery11[-1]
        if AddHolidaysDate(monthrangey1[dow_countery1]) == 0:
            return monthrangey1[dow_countery1]
        else:
            try: 
                cand = monthrangey1[dow_countery11.index(dow_countery1)+1]
            except:
                cand = monthrangey1[dow_countery11.index(dow_countery1)-1]
            return cand    
    else:
        return GetRicHoliday(dtf)
####################################################################################################
def GetRicHoliday(vd):
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17),datetime.date(2018,4,1)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16),datetime.date(2018,4,2)]

    if vd.month == 1 and vd.day == 1:
        return datetime.date(vd.year - 1, 1,1)
    if vd.month  == 1 and vd.day == 6: 
        return datetime.date(vd.year - 1, 1,6)
    if vd.month  == 4 and vd.day == 25: 
        return datetime.date(vd.year - 1, 4,25)
    if vd.month  == 5 and vd.day == 1: 
        return datetime.date(vd.year - 1, 5,1)
    if vd.month  == 6 and vd.day == 2: 
        return datetime.date(vd.year - 1, 6,2)
    if vd.month  == 8 and vd.day == 15: 
        return datetime.date(vd.year - 1, 8,15)
    if vd.month  == 11 and vd.day == 1: 
        return datetime.date(vd.year - 1, 11,1)
    if vd.month  == 12 and vd.day == 8: 
        return datetime.date(vd.year - 1, 12,8)
    if vd.month  == 12 and vd.day == 25: 
        return datetime.date(vd.year - 1, 12,25)
    if vd.month  == 12 and vd.day == 26: 
        return datetime.date(vd.year - 1, 12,26)
    if vd.month  == 12 and vd.day == 31: 
        return datetime.date(vd.year - 1, 12,31)
    if vd in pasqua:
        return pasqua[pasqua.index(vd)-1]
    if vd in pasquetta:
        return pasquetta[pasquetta.index(vd)-1]
####################################################################################################
def GetPUNRic(year):
    
    pun = pd.read_excel('H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2017/06. MI/DB_Borse_Elettriche_PER MI_17_conMacro_072017.xlsm', sheetname = 'DB_Dati')
    pun2 = pd.read_excel('H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx', sheetname = 'DB_Dati')
    pun3 = pd.read_excel('H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx', sheetname = 'DB_Dati')
    
    pun = pun[pun.columns[:13]]    
    pun2 = pun2[pun2.columns[:13]]    
    pun3 = pun3[pun3.columns[:13]]    
    
    pun.columns = [[u'Date', u'Year', u'Quarter', u'Month', u'Week', u'Week Day', u'Day',
       u'Hour', u'Lavorativo/Festivo', u'Weekend', u'PEAK-OFF PEAK',
       u'AEEG 181/06', u'PUN']]
       
    pun2.columns = pun.columns  
    pun3.columns = pun.columns
    
    rowmax = 8784 if year % 4 == 0 else 8760
    diz = OrderedDict()
    dr = pd.to_datetime(pd.date_range(datetime.date(year, 1,1), datetime.date(year, 12, 31), freq = 'D'))
    
    RP = []
    for d in dr:
        ricd = GetRicDateGeneral(d.date())
        ricd2 = GetRicDateGeneral(ricd)
        ricd3 = GetRicDateGeneral(ricd2)
        if ricd <= datetime.datetime.now().date():
            ricpun = 0.9 * pun.PUN.ix[pun.Date == ricd].values.ravel() + 0.1 * pun2.PUN.ix[pun2.Date == ricd2].values.ravel()
        else:
            ricpun = 0.9 * pun2.PUN.ix[pun2.Date == ricd2].values.ravel() + 0.1 * pun3.PUN.ix[pun3.Date == ricd3].values.ravel()
        RP.extend(ricpun.tolist())
    
    diz['pun'] = RP
    diz = pd.DataFrame.from_dict(diz, orient = 'columns').set_index(pd.date_range(datetime.date(year, 1,1), datetime.date(year + 1, 12, 31), freq = 'H')[:rowmax])
    
    return diz