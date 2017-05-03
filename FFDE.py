# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:08:15 2017

@author: d_floriello

Functions for Daily Measures Extractor
"""


import zipfile
import os
from os.path import isfile, join
import datetime
import time
from collections import OrderedDict
import pandas as pd
import numpy as np
import shutil
import re
#import unidecode


####################################################################################################
def Aggregator2(df):
    v = np.repeat(0.0, 24)    
    df2 = df[df.columns[2:98]]
    df2 = df2.values.ravel().astype(float)
    for k in range(1,25):
        v[k-1] += np.sum(np.array([x for x in df2[4*(k-1):4*k]], dtype = np.float64))
    return v
####################################################################################################
def Converter(s):
    points = [m.start() for m in re.finditer('\.', s)]
    if len(points) <= 1:
        return float(np.where(np.isnan(float(s)), 0, float(s)))
    else:
        s2 = s[:points[len(points)-1]].replace('.','') + s[points[len(points)-1]:]
        return float(np.where(np.isnan(float(s2)), 0, float(s2)))
####################################################################################################
def MeasureExtractor(s):
    mis = []
    E = [m.start() for m in re.finditer('=', s)]
    for e in E:
        se = ''
        for i in range(2, 50):
            if s[e+i] != '"':
                se += s[e+i]
            else:
                break
        mis.append(float(se.replace(',','.')))
    return mis
####################################################################################################
def FileFilter(ld, directory):
    mesi = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    to_be_extracted = []
    list_files = [of for of in os.listdir(directory) if isfile(join(directory, of))]
    M = datetime.datetime(2017,1,1,0,0,0)
    last_file = 0
    for f in list_files:
        #filedate = datetime.datetime(2017, int(f[2:4]), int(f[5:7]))
        fdt = time.ctime(os.path.getmtime(directory + "/" + f))
        filedate = datetime.datetime(int(fdt[20:]), mesi.index(fdt[4:7])+1, int(fdt[8:10]), hour = int(fdt[11:13]), minute = int(fdt[14:16]), second = int(fdt[17:19]))        
        if filedate > ld:
            to_be_extracted.append(f)
            if filedate > M:
                M = filedate
                last_file = f
    return to_be_extracted, time.ctime(os.path.getmtime(directory + "/" + last_file))
####################################################################################################    
def ZIPExtractor():
    mesi = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    last_date = open("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/last_date.txt")
    ld = last_date.read()
    LD = datetime.datetime(int(ld[20:]), mesi.index(ld[4:7])+1, int(ld[8:10]), hour = int(ld[11:13]), minute = int(ld[14:16]), second = int(ld[17:19]))
    strm = str(LD.month) if len(str(LD.month)) > 1 else "0" + str(LD.month)
    directory = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/" + str(LD.year) + "-" + strm + "/Giornalieri"
    tbe, M = FileFilter(LD, directory)
    for t in tbe:
        #print(t)
        path = directory + "/" + t
        zf = zipfile.ZipFile(path)
        lzf = [x for x in zf.namelist() if ".zip" in x]
        zf.extractall(path = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TEMP")
        for z in lzf:
                #print(z)
            path2 = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TEMP/" + str(z)
            with zipfile.ZipFile(path2) as zf2:
                right = [str(rf) for rf in zf2.namelist() if 'T1' in str(rf)]
                if len(right) > 0:
                    zf2.extract(right[0], path = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TBP")
    shutil.rmtree("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TEMP")
    os.makedirs("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TEMP")
    new_last_date = open("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/last_date.txt", 'r+')
    new_last_date.write(M)
    return 1    
####################################################################################################
def D_Monthly_Aggregator(month):
    cl = ['E', 'F']
    for h in range(24):
        cl.append(str(h) + '.A')
        cl.append(str(h) + '.B')
        cl.append(str(h) + '.C')
        cl.append(str(h) + '.D')
    cl.append('G')
    Aggdf = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
    Aggdf.columns = [['POD', 'Area', 'Giorno', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24']]
    m = month
    diz = OrderedDict()
    count = 0
    strm = str(m) if len(str(m)) > 1 else "0" + str(m)        
    if month >= 3:
        crppm = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/" + strm + "-2017/_All_CRPP_" + strm + "_2017.xlsx")
    else:
        crppm = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/03-2017/_All_CRPP_03_2017.xlsx")
    pathm = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm + "/Giornalieri/CSV"
    csvfiles = os.listdir(pathm)
    csvfiles = [cf for cf in csvfiles if 'T1' in cf and '.txt' not in cf]
    for cf in csvfiles:
        zona = ""        
        pod = cf[10:24]
        date = datetime.datetime(2017, int(cf[2:4]), int(cf[5:7]))
        if crppm["ZONA"].ix[crppm["POD"] == pod].values.size > 0:
            zona = crppm["ZONA"].ix[crppm["POD"] == pod].values[0]
        df = pd.read_csv(pathm + "/" + cf, sep = ";", dtype = object)
        df.columns = cl
        vec = np.repeat(0.0, 24)
        td = []
        for h in range(24):
            ha = str(h) + '.A'
            hb = str(h) + '.B'
            hc = str(h) + '.C'
            hd = str(h) + '.D'
            va = Converter(str(df[ha].ix[0]))
            vb = Converter(str(df[hb].ix[0]))
            vc = Converter(str(df[hc].ix[0]))
            vd = Converter(str(df[hd].ix[0]))
            vec[h] = np.sum([va, vb, vc, vd], dtype = np.float64)
            
        td.append(pod)
        td.append(zona)
        td.append(date)
        td.extend(vec.tolist())
        diz[count] = td
        count += 1
            
    diz = pd.DataFrame.from_dict(diz, orient = 'index')    
    diz.columns = [['POD', 'Area', 'Giorno', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24']]
    Aggdf = Aggdf.append(diz, ignore_index = True)
    return Aggdf
####################################################################################################
def Aggregator(today):
    cl = ['E', 'F']
    for h in range(24):
        cl.append(str(h) + '.A')
        cl.append(str(h) + '.B')
        cl.append(str(h) + '.C')
        cl.append(str(h) + '.D')
    cl.append('G')
    Aggdf = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
    Aggdf.columns = [['POD', 'Area', 'Giorno', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24']]
    m = today.month
    diz = OrderedDict()
    count = 0
    strm1 = str(m) if len(str(m)) > 1 else "0" + str(m)        
    strm2 = str(m-1) if len(str(m-1)) > 1 else "0" + str(m-1)        
    crppm1 = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/" + strm1 + "-2017/_All_CRPP_" + strm1 + "_2017.xlsx")
    crppm2 = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/" + strm2 + "-2017/_All_CRPP_" + strm2 + "_2017.xlsx")
    pathm = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TBP"
    csvfiles = os.listdir(pathm)
    csvfiles = [cf for cf in csvfiles if 'T1' in cf and '.txt' not in cf]
    for cf in csvfiles:
        pod = cf[10:24]
        date = datetime.datetime(2017, int(cf[2:4]), int(cf[5:7]))
        zona = 0
        if date.month == m:
            zona = crppm1["ZONA"].ix[crppm1["POD"] == pod].values[0] if crppm1["ZONA"].ix[crppm1["POD"] == pod].values.size > 0 else 0
            shutil.copy2(pathm + "/" + cf,"H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm1 + "/Giornalieri/CSV")
        else:
            zona = crppm2["ZONA"].ix[crppm2["POD"] == pod].values[0] if crppm2["ZONA"].ix[crppm2["POD"] == pod].values.size > 0 else 0
            shutil.copy2(pathm + "/" + cf,"H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm2 + "/Giornalieri/CSV")
        df = pd.read_csv(pathm + "/" + cf, sep = ";", dtype = object)
        df.columns = cl
        vec = np.repeat(0.0, 24)
        td = []
        for h in range(24):
            ha = str(h) + '.A'
            hb = str(h) + '.B'
            hc = str(h) + '.C'
            hd = str(h) + '.D'
            va = Converter(str(df[ha].ix[0]))
            vb = Converter(str(df[hb].ix[0]))
            vc = Converter(str(df[hc].ix[0]))
            vd = Converter(str(df[hd].ix[0]))
            vec[h] = np.sum([va, vb, vc, vd], dtype = np.float64)
            
        td.append(pod)
        td.append(zona)
        td.append(date)
        td.extend(vec.tolist())
        diz[count] = td
        count += 1
            
    diz = pd.DataFrame.from_dict(diz, orient = 'index')    
    diz.columns = [['POD', 'Area', 'Giorno', '1', '2', '3', '4', '5', '6','7', '8', '9', '10', '11', '12','13', '14', '15', '16', '17', '18','19', '20', '21', '22', '23', '24']]
    Aggdf = Aggdf.append(diz, ignore_index = True)
    shutil.rmtree("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TBP")
    os.makedirs("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TBP")
    Aggdf = Aggdf.drop_duplicates(subset = ["POD", "Giorno"], keep = "first")
    return Aggdf    
##################################################################################################    
def T1_Mover(m):
    strm = str(m) if len(str(m)) > 1 else "0" + str(m)   
    os.makedirs("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm + "/T1")
    pathm = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm + "/Giornalieri/CSV/"
    csvfiles = os.listdir(pathm)
    csvfiles = [cf for cf in csvfiles if 'T1' in cf and '.txt' not in cf]
    for cf in csvfiles:
        shutil.copy2(pathm + cf, "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm + "/T1")
    return 1
##################################################################################################    
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
def Bridge(vd):
    
    bridge = 0
    if vd.weekday() == 0:
        Tues = vd + datetime.timedelta(days = 1)
        if AddHolidaysDate(Tues) == 1:
            bridge = 1
    elif vd.weekday() == 4:
        Thur = vd - datetime.timedelta(days = 1)
        if AddHolidaysDate(Thur) == 1:
            bridge = 1    
    else:
        pass
    
    return bridge
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
def percentageConsumption(db, All, zona, today):
    str_month = str(today.month) if len(str(today.month)) > 1 else "0" + str(today.month)
    str_day = str(today.day) if len(str(today.day)) > 1 else "0" + str(today.day)
    dr = pd.date_range('2017-01-01', str(today.year) + '-' + str_month + '-' + str_day, freq = 'D')
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
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
def MakeExtendedDatasetWithSampleCurve(df, db, meteo, All, zona, today):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    psample = percentageConsumption(db, All, zona, today)
    str_month = str(today.month) if len(str(today.month)) > 1 else "0" + str(today.month)
    str_day = str(today.day) if len(str(today.day)) > 1 else "0" + str(today.day)
    psample = psample.set_index(pd.date_range('2017-01-01', str(today.year) + '-' + str_month + '-' + str_day, freq = 'D'))
    dts = OrderedDict()
    df = df.ix[df.index.date >= datetime.date(2017,1,3)]
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
    
    
    