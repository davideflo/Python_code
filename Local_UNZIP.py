# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:23:28 2017

@author: d_floriello


UNZIP in local
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
import re

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

mypath = 'H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A'
extracter = 'C:/Users/d_floriello/Desktop/extracted'

years = [2015, 2016, 2017]

for y in years:
    df = OrderedDict()
    count = 0 
    path2 = mypath + '/' + str(y)
    #dirs = [os.path.join(path2,o) for o in os.listdir(path2) if os.path.isdir(os.path.join(path2,o))]
    dirs = os.listdir(path2)
    for d in dirs:
        if 'rettifiche tardive' not in d or 'storici aeeg' not in d:
            print('Working in {}'.format(d))
            ### go into 'giornalieri'
            path3 = path2 + '/' + d + '/giornalieri'
            onlyfiles = [f for f in listdir(path3) if isfile(join(path3, f))]
            for f in onlyfiles:
                zip_ref = zipfile.ZipFile(path3 + '/' + f, 'r')
                inside = zip_ref.namelist() #### files inside the zipped directory
                for i in inside:
                    zn = inside.index(i)
                    path4 = extracter + '/' + str(zn)
                    #path4 = path3 + '/extracted/' + str(zn)
                    zip_ref.extractall(path4)
                    ### da qui entra in extracted_zn, elenca i file, estrai i file, elabora e chiudi
                    onlyfileszn = [of for of in listdir(path4) if isfile(join(path4, of))]
                    for o in onlyfileszn:
                        if '.csv' not in o:
                            i2 = onlyfileszn.index(o)
                            path5 = path4 + '/biextracted/' + str(i2)                        
                            zip2 = zipfile.ZipFile(path4 + '/' + o) 
                            zip2.extractall(path5)
                            #inside_path5 = [iif for iif in listdir(path5) if isfile(join(path5, iif))]
                            inside_path5 = zip2.namelist()
                            T1 = [di for di in inside_path5 if 'T1' in di]
                            ### get date and pod
                            for T in T1:
                                dt = datetime.datetime(y, int(T[2:4]), int(T[5:7]))
                                pod = T[T.find('_')+1:T.find('.csv')] 
                                s = zip2.read(T)
    #                            t1df = pd.read_csv(path5 + '/' + T1, sep = ';', dtype = object)
                                todiz = [pod, dt, s[548:741]]
    #                            todiz.extend(Aggregator(t1df).tolist())
                                df[count] = todiz
                                count += 1
    #### http://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python
                            zip2.close()
                    shutil.rmtree(path5)
                zip_ref.close()    
                shutil.rmtree(path4)
    df = pd.DataFrame.from_dict(df, orient = 'index')
    df.to_csv('Hdatabase_' + str(y), sep = ';')
    del df
#### copy all files into a new directory and then operate in the new directory                
#### https://docs.python.org/2/library/shutil.html
    
###############################################################################

extracter = 'C:/Users/d_floriello/Desktop/tbe2'
onlyfiles = [f for f in listdir(extracter) if isfile(join(extracter, f))]

for of in onlyfiles:
    if '.zip' in of:
        zip_ref = zipfile.ZipFile(extracter + '/' + of)
        zip_ref.extractall(extracter)
        zip_ref.close()
        
        
cl = ['E', 'F']
for h in range(24):
    cl.append(str(h) + '.A')
    cl.append(str(h) + '.B')
    cl.append(str(h) + '.C')
    cl.append(str(h) + '.D')
        
        
y = 2016
count = 0
df = OrderedDict()
for of in onlyfiles:
    if 'T1' in of:
        dt = datetime.datetime(y, int(of[2:4]), int(of[5:7]))
        pod = of[of.find('_')+1:of.find('.csv')] 
        t1df = pd.read_csv(extracter + '/' + of, sep = ';', dtype = object)
        todiz = [pod, dt]
        todiz.extend(t1df.ix[0].values.ravel().tolist()[:-1])
        df[count] = todiz
        count += 1
    
df = pd.DataFrame.from_dict(df, orient = 'index')
names = ['POD', 'date']    
names.extend(cl)
df.columns = names

df.to_csv('orari_2016.csv', sep = ';')


##################### Elaborazione curve giornaliere ##########################

df = pd.read_csv('orari_2016.csv', sep = ';', dtype = object)

diz = OrderedDict()

for i in range(df.shape[0]):
    vec = np.repeat(0.0, 24)
    td = []
    for h in range(24):
        ha = str(h) + '.A'
        hb = str(h) + '.B'
        hc = str(h) + '.C'
        hd = str(h) + '.D'
        va = Converter(str(df[ha].ix[i]))
        vb = Converter(str(df[hb].ix[i]))
        vc = Converter(str(df[hc].ix[i]))
        vd = Converter(str(df[hd].ix[i]))
        vec[h] = np.sum([va, vb, vc, vd], dtype = np.float64)
    td.append(df['POD'].ix[i])
    td.append(datetime.datetime(2016, int(df['date'].ix[i][5:7]), int(df['date'].ix[i][8:])))
    td.extend(vec.tolist())
    diz[i] = td
    
diz = pd.DataFrame.from_dict(diz, orient = 'index')    
diz.columns = [['POD', 'Date', '1', '2', '3', '4', '5', '6',
                '7', '8', '9', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24']]
 
diz.to_csv('G_orari_2016.csv', sep = ';')   
diz.to_excel('G_orari_2016.xlsx')   
    
    


