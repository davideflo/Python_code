# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:39:17 2017

@author: d_floriello

Extracting daily measures
"""

import zipfile
import os
from os import listdir
from os.path import isfile, join
import datetime

mypath = 'H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A'

years = [2015, 2016, 2017]

for y in years:
    path2 = mypath + '/' + str(y)
    #dirs = [os.path.join(path2,o) for o in os.listdir(path2) if os.path.isdir(os.path.join(path2,o))]
    dirs = os.listdir(path2)
    for d in dirs:
        if 'rettifiche tardive' not in d or 'storici aeeg' not in d:
            ### go into 'giornalieri'
            path3 = path2 + '/' + d + '/giornalieri'
            onlyfiles = [f for f in listdir(path3) if isfile(join(path3, f))]
            for f in onlyfiles:
                zip_ref = zipfile.ZipFile(path3 + '/' + f, 'r')
                zip_ref.extractall(path3 + '/extracted')
                inside = zip_ref.namelist() #### files inside the zipped directory
                for i in inside:
                    ### da qui entra in extracted, estrai i file, elabora e chiudi
                    archive = zip_ref.read(path3 + '/' + f)# + '/' + i)
                    zip_i = zipfile.ZipFile(path3 + '/' + f + '/' + i)
                    deepinside = zip_i.namelist()
                    T1 = [di for di in deepinside if 'T1' in di][0]
                    ### get date and pod
                    dt = datetime.datetime(y, int(T1[2:4]), int(T1[5:7]))
                    pod = T1[T1.find('_')+1:] 
            

zip_ref.close()