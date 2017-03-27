# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:27:26 2017

@author: d_floriello

PDO Unzipper
"""


#import zipfile
import os
from os import listdir
from os.path import isfile, join
import datetime
from collections import OrderedDict
import pandas as pd
import numpy as np
import shutil
import re
from bs4 import BeautifulSoup
import time

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
    

directory = 'C:/Users/d_floriello/Desktop/PDO2015'
files = os.listdir(directory)

destinationfile = 'C:/Users/d_floriello/Desktop/FTP_PDO_2015'
destinationDF = 'C:/Users/d_floriello/Desktop/DF_PDO_2015'

#files2 = files[:5]

filecounter = 0
count = 0
while filecounter < 500:
    files = os.listdir(directory)
    if len(files) > 0:
        files2 = files[:10]
        dix = OrderedDict()
        print 'done {} files'.format(filecounter)
        start_time = time.time()
        for f in files2:        
            pdo = BeautifulSoup(open(directory + '/' + f).read(), "xml")
            bs = pdo.find_all('DatiPod')
            for b in bs:
                pod = b.find_all('Pod')
                M = b.find_all('MeseAnno')[:2]
                Er = b.find_all('Er')
                for er in Er:
                    tbi = []
                    day = Er.index(er)
                    mis = MeasureExtractor(str(er))
                    tbi.append(pod[0])
                    tbi.append(day)
                    tbi.append(M)
                    tbi.append(2015)
                    tbi.extend(mis)
                    dix[count] = tbi
                    count += 1
            filecounter += 1
            shutil.move(directory + '/' + f, destinationfile + '/' + f)            
        dix = pd.DataFrame.from_dict(dix, orient = 'index')
        dix.to_excel(destinationDF + '/df_' + str(filecounter) + '.xlsx')    
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        break
            

filesdf = os.listdir(destinationDF)
DF = pd.DataFrame()
for fdf in filesdf:
    DF = DF.append(pd.read_excel(destinationDF + '/' + fdf), ignore_index = True)

DF.to_excel('C:/Users/d_floriello/PDO2015_estratti.xlsx')
DF.to_csv('C:/Users/d_floriello/Desktop/PDO2015_estratti.csv', sep = ';')
DF.to_pickle('C:/Users/d_floriello/Desktop/PDO2015_estratti.pkl')
DF.to_hdf('C:/Users/d_floriello/Desktop/PDO2015_estratti.h5', 'DF')