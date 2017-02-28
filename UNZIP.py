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
from collections import OrderedDict
import pandas as pd
import numpy as np
import shutil

####################################################################################################
def Aggregator(df):
    v = np.repeat(0.0, 24)    
    df2 = df[df.columns[2:98]]
    df2 = df2.values.ravel().astype(float)
    for k in range(1,25):
        v[k-1] += np.sum(np.array([x for x in df2[4*(k-1):4*k]], dtype = np.float64))
    return v
####################################################################################################

mypath = 'H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A'

years = [2015, 2016, 2017]

df = OrderedDict()
count = 0 
for y in years:
    path2 = mypath + '/' + str(y)
    #dirs = [os.path.join(path2,o) for o in os.listdir(path2) if os.path.isdir(os.path.join(path2,o))]
    dirs = os.listdir(path2)
    for d in dirs:
        if 'rettifiche tardive' not in d or 'storici aeeg' not in d:
            print 'Working in {}'.format(d)
            ### go into 'giornalieri'
            path3 = path2 + '/' + d + '/giornalieri'
            onlyfiles = [f for f in listdir(path3) if isfile(join(path3, f))]
            for f in onlyfiles:
                zip_ref = zipfile.ZipFile(path3 + '/' + f, 'r')
                inside = zip_ref.namelist() #### files inside the zipped directory
                for i in inside:
                    zn = inside.index(i)
                    path4 = path3 + '/extracted/' + str(zn)
                    zip_ref.extractall(path4)
                    ### da qui entra in extracted_zn, elenca i file, estrai i file, elabora e chiudi
                    onlyfileszn = [of for of in listdir(path4) if isfile(join(path4, of))]
                    for o in onlyfileszn:
                        i2 = onlyfileszn.index(o)
                        path5 = path4 + '/biextracted/' + str(i2)                        
                        zip2 = zipfile.ZipFile(path4 + '/' + o) 
                        zip2.extractall(path5)
                        inside_path5 = [iif for iif in listdir(path5) if isfile(join(path5, iif))]
                        T1 = [di for di in inside_path5 if 'T1' in di][0]
                        ### get date and pod
                        dt = datetime.datetime(y, int(T1[2:4]), int(T1[5:7]))
                        pod = T1[T1.find('_')+1:T1.find('.csv')] 
                        t1df = pd.read_csv(path5 + '/' + T1, sep = ';', dtype = object)
                        todiz = [pod, dt]
                        todiz.extend(Aggregator(t1df).tolist())
                        df[count] = todiz
                        count += 1
#### http://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python
                        zip2.close()
                        shutil.rmtree(path5)
                    shutil.rmtree(path4)
                zip_ref.close()
#### copy all files into a new directory and then operate in the new directory                
#### https://docs.python.org/2/library/shutil.html