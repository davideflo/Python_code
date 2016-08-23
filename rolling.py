# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:38:48 2016

@author: utente

creation of ROLLING datasets in python
"""

import numpy as np
import pandas as pd
import datetime
#########################################################
def convert_day(day):
    if day == "Sunday":
        return "dom"
    elif day == "Monday":
      return "lun"
    elif day == "Tuesday":
      return "mar"
    elif day == "Wednesday":
      return "mer"
    elif day == "Thursday":
      return "gio"
    elif day == "Friday":
      return "ven"
    else:
      return "sab"
#########################################################
def convert_day_to_angle(day):
    days = ["dom", "lun", "mar", "mer", "gio", "ven", "sab"]
    ang = days.index(day)
    return np.cos(ang*np.pi/7)
##############################################################
def convert_hour_to_angle(vo):
    return np.sin((vo-12)*np.pi/24)
#############################################################
def numeric_days(vd):
    nd = np.zeros(len(vd))
    for i in range(len(vd)):
        nd[i] = convert_day_to_angle(vd[i])
    return nd
################################################################
def subsequent_day(day):
    if day == "lun":
        return "mar"
    elif day == "mar":
        return "mer"
    elif day == "mer":
        return "gio"
    elif day == "gio":
        return "ven"
    elif day == "ven":
        return "sab"
    elif day == "sab":
        return "dom"
    else:
        return "lun"
################################################################
def add_days(first_day, year):
    dl = []
    if year % 4 == 0:
        day2 = subsequent_day(first_day)
        day3 = subsequent_day(day2)
        day4 = subsequent_day(day3)
        day5 = subsequent_day(day4)
        day6 = subsequent_day(day5)
        day7 = subsequent_day(day6)
        week = np.repeat(np.array([first_day, day2,day3,day4,day5,day6,day7]),[24,24,24,24,24,24,24], axis = 0)
        nw = np.round(366/7, decimals = 0)
        wr = 366 - nw*7
        rdl = []
        for i in range(int(wr)*24):
            rdl.append(week[i])
        dl.append(np.repeat(week, nw, axis=0))
        dl.append(np.array(rdl))
    else:
        day2 = subsequent_day(first_day)
        day3 = subsequent_day(day2)
        day4 = subsequent_day(day3)
        day5 = subsequent_day(day4)
        day6 = subsequent_day(day5)
        day7 = subsequent_day(day6)
        week = np.repeat(np.array([first_day, day2,day3,day4,day5,day6,day7]),[24,24,24,24,24,24,24], axis = 0)
        nw = np.round(365/7, decimals = 0)
        wr = 365 - nw*7
        rdl = []
        for i in range(int(wr)*24):
            rdl.append(week[i])
        dl.append(np.repeat(week, nw, axis=0))
        dl.append(np.array(rdl))
    C = [item for sublist in dl for item in sublist]    
    return np.array(C)
##############################################################
def dates(vd):
    dt = []
    vd = vd.reset_index(drop=True)
    for i in range(vd.size):
        d = str(int(vd[i]))
        dt.append(d[6]+d[7]+"/"+d[4]+d[5]+"/"+d[0]+d[1]+d[2]+d[3])
    return dt
######################################################################
def dates2(dd):
    d = str(dd)
    dt = d[6]+d[7]+"/"+d[4]+d[5]+"/"+d[0]+d[1]+d[2]+d[3]
    return dt
######################################################################
def add_holidays(vd):
    holiday = np.zeros(len(vd))
    pasqua = ["04/04/2010", "24/04/2011", "08/04/2012", "31/03/2013", "20/04/2014", "05/04/2015", "27/03/2016"]
    pasquetta = ["05/04/2010", "25/04/2011", "09/04/2012", "01/04/2013", "21/04/2014", "06/04/2015", "28/03/2016"]
    for i in range(len(vd)):
        d = vd[i]
        if d[0:5] == "01/01":
            holiday[i] = 1
        elif d[0:5] == "06/01":
            holiday[i] = 2
        elif d[0:5] == "25/04":
            holiday[i] = 5
        elif d[0:5] == "01/05":
            holiday[i] = 6
        elif d[0:5] == "02/06":
            holiday[i] = 7
        elif d[0:5] == "15/08":
            holiday[i] = 8
        elif d[0:5] == "01/11":
            holiday[i] = 9
        elif d[0:5] == "08/12":
            holiday[i] = 10
        elif d[0:5] == "25/12":
            holiday[i] = 11
        elif d[0:5] == "26/12":
            holiday[i] = 12
        elif d[0:5] == "31/12":
            holiday[i] = 13
        elif d in pasqua:
            holiday[i] = 3
        elif d in pasquetta:
            holiday[i] = 4
        else:
            pass
    return holiday 
######################################################################
def add_holidays2(d):
    holiday = 0
    pasqua = ["04/04/2010", "24/04/2011", "08/04/2012", "31/03/2013", "20/04/2014", "05/04/2015", "27/03/2016"]
    pasquetta = ["05/04/2010", "25/04/2011", "09/04/2012", "01/04/2013", "21/04/2014", "06/04/2015", "28/03/2016"]
    if d[0:5] == "01/01":
        holiday = 1
    elif d[0:5] == "06/01":
        holiday = 2
    elif d[0:5] == "25/04":
        holiday = 5
    elif d[0:5] == "01/05":
        holiday = 6
    elif d[0:5] == "02/06":
        holiday = 7
    elif d[0:5] == "15/08":
        holiday = 8
    elif d[0:5] == "01/11":
        holiday = 9
    elif d[0:5] == "08/12":
        holiday = 10
    elif d[0:5] == "25/12":
        holiday = 11
    elif d[0:5] == "26/12":
        holiday = 12
    elif d[0:5] == "31/12":
        holiday = 13
    elif d in pasqua:
        holiday = 3
    elif d in pasquetta:
        holiday = 4
    else:
        pass
    return np.array(holiday) 
##########################################################################
def associate_days(ora, day):
    vday = np.repeat(np.array(day), 24, axis=0)     
    ora = ora.reset_index(drop=True)
    l = len(ora)-1
    for i in range(l):
        if (ora[i] == 24 and ora[i+1] == 1) or (ora[i] == 23 and ora[i+1] == 1) or (ora[i] == 25 and ora[i+1] == 1):        
            index = i+1
            for j in range(index,24):
                vday[j] = subsequent_day(day)
    return vday
########################################################################
def associate_meteo_ora(data, meteo, meteovar):
    vm = []
    dm = meteo[meteo.columns[0]].tolist()
    for i in range(len(data)):
        ir = dm.index(data[i])
        vm.append(meteo[meteovar].ix[ir])
    return np.array(vm)
########################################################################
def associate_meteo_ora2(data, meteo, meteovar):
    dm = meteo[meteo.columns[0]].tolist()
    ir = dm.index(data)
    return meteo[meteovar].ix[ir]
########################################################################
def day_splitter(dt):
    day, month, year = (int(x) for x in dt.split('/')) 
    ans = datetime.date(year, month,day)
    return ans.strftime("%A")    
########################################################################    
def create_rolling_dataset(pun, first_day, varn, meteo, step, day_ahead, hb):
    DF = pd.DataFrame()
    Y = []
    names1 = [varn , "aust-", "cors-","fran-", "grec-", "slov-", "sviz-", "angleday-","holiday-","angleora-",
              "tmin-", "tmax-", "tmed-","rain-", "vento-", "day-"]
    names2 = ["target_ora", "target_day", "target_holiday","target_tmin","target_tmax","target_tmed",
              "target_pioggia","target_vento"]
              
    nam = []    
    
    for n in names1:    
        for i in range(hb, 0, -1):
            nam.append(n+str(i))
    
    names = [nam[:-hb],names2,nam[len(nam)-hb:len(nam)]]    
    names = [item for sublist in names for item in sublist]    
    
    hbb = hb - 1
    da = 24 * day_ahead
    num = pun.shape[0] - (hb + step + da)    
    
    for i in range(num):
        p = pun[varn].ix[i:(i+hbb)]
        aus = pun["AUST"].ix[i:(i+hbb)]
        cors = pun["CORS"].ix[i:(i+hbb)]
        fran = pun["FRAN"].ix[i:(i+hbb)]
        grec = pun["GREC"].ix[i:(i+hbb)]
        slov = pun["SLOV"].ix[i:(i+hbb)]
        sviz = pun["SVIZ"].ix[i:(i+hbb)]
        ora = pun[pun.columns[1]].ix[i:(i+hbb)]
        dat = pun[pun.columns[0]].ix[i:(i+hbb)]

        day = 0
        if DF.shape[0] > 0:
            day = DF["day-1"].ix[DF.shape[0]-1]
        else:
            day = first_day

        ds = dates(dat)
        
        ### TARGET VALUES ###
        y = pun["PUN"].ix[i+da+hb+step]
        new_hour = pun[pun.columns[1]].ix[da+i+hb+step]
        new_date = dates2(pun[pun.columns[0]].ix[da+i+hb+step])
        thol = add_holidays2(new_date)       
        tday = convert_day_to_angle(convert_day(day_splitter(new_date)))
        thour = convert_hour_to_angle(new_hour)
        #####
        
        ### TARGET METEO ###
        ttmin = associate_meteo_ora2(new_date, meteo, "Tmin")
        ttmax = associate_meteo_ora2(new_date, meteo, "Tmax")
        ttmed = associate_meteo_ora2(new_date, meteo, "Tmedia")
        train = associate_meteo_ora2(new_date, meteo, "Pioggia")
        tvm = associate_meteo_ora2(new_date, meteo, "Vento_media")
        ####
        
        tmin = associate_meteo_ora(ds, meteo, "Tmin")
        tmax = associate_meteo_ora(ds, meteo, "Tmax")
        tmed = associate_meteo_ora(ds, meteo, "Tmedia")
        rain = associate_meteo_ora(ds, meteo, "Pioggia")
        vm = associate_meteo_ora(ds, meteo, "Vento_media")        
        
        hol = add_holidays(ds)
        vdays = associate_days(ora, day)
        aday = [convert_day_to_angle(v) for v in vdays]
        ahour = convert_hour_to_angle(ora)
        
        row = [np.array(p).T.tolist(), np.array(aus).T.tolist(), np.array(cors).T.tolist(),
        np.array(fran).T.tolist(), np.array(grec).T.tolist(), np.array(slov).T.tolist(), np.array(sviz).T.tolist(),
        np.array(aday).T.tolist(), np.array(hol).T.tolist(), np.array(ahour).T.tolist(), np.array(tmin).T.tolist(), 
        np.array(tmax).T.tolist(),
        np.array(tmed).T.tolist(), np.array(rain).T.tolist(), np.array(vm).T.tolist(), np.array([thour]).T.tolist(), 
        np.array([tday]).T.tolist(), np.array([thol]).T.tolist(),
        np.array([ttmin]).T.tolist(), np.array([ttmax]).T.tolist(), np.array([ttmed]).T.tolist(), 
        np.array([train]).T.tolist(), np.array([tvm]).T.tolist(), np.array(vdays).T.tolist()]        
        
        
        row = [item for sublist in row for item in sublist]           
        
        df = pd.DataFrame(np.array(row).reshape(1,392), columns = names)
        DF = DF.append(df, ignore_index = True)
        Y.append(y)

    return DF[DF.columns[0:(len(names)-hb)]], np.array(Y)
########################################################################    
def generate_days(ora, first_day):
    days = [] 
    for i in range(24):
        days.append(first_day)
    for i in range(24, ora.size):
        if ora[i] != 1:
            days.append(days[i-1])
        else:
            days.append(subsequent_day(days[i-1]))
    return np.array(days)









    
        
    
    
    