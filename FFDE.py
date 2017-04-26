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
def Aggregator(df):
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
    M = 0
    for f in list_files:
        #filedate = datetime.datetime(2017, int(f[2:4]), int(f[5:7]))
        fdt = time.ctime(os.path.getmtime(directory + "/" + f))
        filedate = datetime.datetime(int(fdt[20:]), mesi.index(fdt[4:7]) +1, int(fdt[8:10]))        
        if filedate > ld:
            to_be_extracted.append(f)
            if filedate.day > M:
                M = filedate.day
    return to_be_extracted, M
####################################################################################################    
def ZIPExtractor():
    last_date = open("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/last_date.txt")
    ld = last_date.read()
    LD = datetime.datetime(int(ld[6:10]), int(ld[3:5]), int(ld[0:2]))
    #shutil.rmtree("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/last_date.txt")
    directory = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/" + ld[6:10] + "-" + ld[3:5]
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
                zf2.extract(right[0], path = directory + "/CSV")
        shutil.rmtree("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/TEMP")
    new_last_date = open("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/last_date.txt", 'r+')
    if len(str(M)) == 1:
        M = '0'+ str(M)
    new_last_date.write(str(M) + "/" + ld[3:5] + "/" + "2017")
    return 1    
####################################################################################################
def DAggregator(month):
    Aggdf = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
    for m in range(1, month + 1):
        strm = str(m) if len(str(m)) > 1 else "0" + str(m)
        pathm = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/2017-" + strm + "/CSV"
        csvfiles = os.listdir(pathm)
        for cf in csvfiles:
            df = pd.read_csv(pathm + "/" + cf, sep = ";", dtype = object)
            
    
