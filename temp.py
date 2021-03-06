# -*- coding: utf-8 -*-
"""
Stesse funzioni di "functions_for_PUN.R" in Python


"""
#import h2o
import numpy as np
import pandas as pd
#########################################################
def convert_day_to_angle(day):
    days = ["dom", "lun", "mar", "mer", "gio", "ven", "sab"]
    ang = days.index(day)
    return np.cos(ang*np.pi/7)
##############################################################
def convert_hour_to_angle(vo):
    return np.sin(vo*np.pi/24)
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
    for i in range(len(vd)):
        d = str(int(vd[i]))
        dt.append(d[6]+d[7]+"/"+d[4]+d[5]+"/"+d[0]+d[1]+d[2]+d[3])
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
##########################################################################
def associate_days(ora, day):
    vday = np.repeat(np.array(day), 24, axis=0)     
    ora = ora.reset_index(drop=True)
    l = len(ora)-1
    for i in range(l):
        if ora[i] == 24 and ora[i+1] == 1:        
            index = i+1
            for j in range(index,24):
                vday[j] = subsequent_day(day)
    return vday
########################################################################
def associate_meteo_ora(data, meteo, meteovar):
    vm = []
    dm = meteo[meteo.columns[0]].tolist()
    for i in range(data.size):
        ir = dm.index(data[i])
        vm.append(meteo[meteovar].ix[ir])
    return np.array(vm)
########################################################################
def create_dataset(pun, first_day):
    # pun is a pd.DataFrame
    DF = pd.DataFrame()
    Y = []
    names1 = ["pun-", "aust-", "cors-","fran-", "grec-", "slov-", "sviz-", "angleday-","holiday-","day-"]
    nam = []
    for n in names1:
        for i in range(24, 0, -1):
            nam.append(n+str(i))
    for i in range(pun.shape[0] - 24):
      #  print(i)
        p = pun["PUN"].ix[i:(i+23)]
        aus = pun["AUST"].ix[i:(i+23)]
        cors = pun["CORS"].ix[i:(i+23)]
        fran = pun["FRAN"].ix[i:(i+23)]
        grec = pun["GREC"].ix[i:(i+23)]
        slov = pun["SLOV"].ix[i:(i+23)]
        sviz = pun["SVIZ"].ix[i:(i+23)]
        ora = pun["Ora"].ix[i:(i+23)]
        dat = pun["   Data/Date"].ix[i:(i+23)]
        y = pun["PUN"].ix[i+24]
        day = 0
        if DF.shape[0] > 0:
            day = DF["day-1"].ix[DF.shape[0]-1]
        else:
            day = first_day
        ds = dates(dat)
        hol = add_holidays(ds)
        vdays = associate_days(ora, day)
        aday = [convert_day_to_angle(v) for v in vdays]
        df = pd.DataFrame(np.concatenate([np.array(p).T, np.array(aus).T, np.array(cors).T,
        np.array(fran).T, np.array(grec).T, np.array(slov).T, np.array(sviz).T,
        np.array(aday).T, np.array(hol).T, np.array(vdays).T]).ravel().reshape((1,240)), columns = nam)
        Y.append(y)
        DF = DF.append(df, ignore_index=True)
    return DF, Y
#####################################################################
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
####################################################################
def create_dataset23(pun, first_day, varn, meteo):
    # pun is a pd.DataFrame
    DF = pd.DataFrame()
    Y = []
    names1 = ["pun-", "aust-", "cors-","fran-", "grec-", "slov-", "sviz-", "angleday-","holiday-",
              "angleora-", "tmin-","tmax-","tmed-","rain-","vento-","day-"]
    nam = []
    for n in names1:
        for i in range(23, 0, -1):
            nam.append(n+str(i))
    for i in range(pun.shape[0] - 22):
      #  print(i)
        p = pun[varn].ix[i:(i+22)]
        aus = pun["AUST"].ix[i:(i+22)]
        cors = pun["CORS"].ix[i:(i+22)]
        fran = pun["FRAN"].ix[i:(i+22)]
        grec = pun["GREC"].ix[i:(i+22)]
        slov = pun["SLOV"].ix[i:(i+22)]
        sviz = pun["SVIZ"].ix[i:(i+22)]
        ora = pun[pun.column[1]].ix[i:(i+22)]
        dat = pun[pun.column[0]].ix[i:(i+22)]
        y = pun[varn].ix[i+23]
        day = 0
        if DF.shape[0] > 0:
            day = DF["day-1"].ix[DF.shape[0]-1]
        else:
            day = first_day
        ds = dates(dat)
        hol = add_holidays(ds)
        vdays = associate_days(ora, day)
        aday = [convert_day_to_angle(v) for v in vdays]
        ahour = convert_hour_to_angle(ora)
        tmin = associate_meteo_ora(np.array(ds),meteo,"Tmin")
        tmax = associate_meteo_ora(np.array(ds),meteo,"Tmax")
        tmed = associate_meteo_ora(np.array(ds),meteo,"Tmedia")
        rain = associate_meteo_ora(np.array(ds),meteo,"Pioggia")
        vm = associate_meteo_ora(np.array(ds),meteo,"Vento_media") 
        df = pd.DataFrame(np.concatenate([np.array(p).T, np.array(aus).T, np.array(cors).T,
        np.array(fran).T, np.array(grec).T, np.array(slov).T, np.array(sviz).T,
        np.array(aday).T, np.array(hol).T, np.array(ahour).T, np.array(tmin).T, np.array(tmax).T,
        np.array(tmed).T, np.array(rain).T, np.array(vm).T,
        np.array(vdays).T]).ravel().reshape((1,240)), columns = nam)
        Y.append(y)
        DF = DF.append(df, ignore_index=True)
    return DF, Y
#####################################################################
def sequence_days(vd, first_day):
    res = [first_day]
    for i in range(1,len(vd),1):
        res.append(subsequent_day(res[i-1]))
    return res
#####################################################################
def create_dataset_arima(pun,first_day,varn,meteo):
    Y = []
    dats = pd.to_datetime(np.unique(np.array(dates(pun[pun.columns[0]]))),dayfirst=True).sort_values()
    dats2 = dates(pd.Series(list(set(pun[pun.columns[0]]))))
    for dts in np.unique(np.array(pun[pun.columns[0]])):
        Y.append(max(pun[varn].ix[pun[pun.columns[0]] == dts]))
    hol = add_holidays(dats2)
    vdays = sequence_days(dats2,first_day)
    aday = [convert_day_to_angle(v) for v in vdays]
    pdf = {'Data': dats, 'days': aday, 'holiday': hol, 
           'Tmin':meteo['Tmin'].ix[[i for i,d in enumerate(meteo[meteo.columns[0]]) if d in dats]],
           'Tmedia': meteo['Tmedia'].ix[[i for i,d in enumerate(meteo[meteo.columns[0]]) if d in dats]], 
           'Tmax': meteo['Tmax'].ix[[i for i,d in enumerate(meteo[meteo.columns[0]]) if d in dats]], 
           'Rain': meteo['Pioggia'].ix[[i for i,d in enumerate(meteo[meteo.columns[0]]) if d in dats]],
           'Wind': meteo['Vento_media'].ix[[i for i,d in enumerate(meteo[meteo.columns[0]]) if d in dats]]}
    return pd.DataFrame(pdf).ix[:, ['Data', 'days','holiday', 'Tmin', 'Tmedia', 'Tmax', 'Rain', 'Wind']], np.array(Y)
        











    
        
    
    
    