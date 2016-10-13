# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:20:25 2016

@author: d_floriello

Flow - MC analysis
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime
from lxml import objectify
from os import listdir
from os.path import join, isdir
from sklearn import linear_model
import scipy.stats


###############################################################################
def modify_XML(path):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    file = open(path, 'w')
    todelete = ['<xs', '</xs']
    for line in lines:
        if todelete[0] in line:
            print('skip')
        elif todelete[1] in line:
            print('skip this too')
        else:
            print('write')
            file.write(line.replace(',','.'))
    file.close()
###############################################################################    
def read_XML(path):
#    var = ['NORD', 'FRAN', 'SVIZ']
    xml = objectify.parse(open(path))
    root = xml.getroot()
    nord_fran = []
    fran_sviz = []
    sviz_nord = []
    nord_sviz = []
    sviz_fran = []
    fran_nord= []
    diz = OrderedDict()
    for i in range(len(root.getchildren())):
        child = root.getchildren()[i].getchildren()
        if child[3] == 'NORD' and child[4] == 'FRAN':
            nord_fran.append(float(child[5]))
        elif child[3] == 'NORD' and child[4] == 'SVIZ':
            nord_sviz.append(float(child[5]))
        elif child[3] == 'FRAN' and child[4] == 'SVIZ':
            fran_sviz.append(float(child[5]))
        elif child[3] == 'FRAN' and child[4] == 'NORD':
            fran_nord.append(float(child[5]))
        elif child[3] == 'SVIZ' and child[4] == 'NORD':
            sviz_nord.append(float(child[5]))
        elif child[3] == 'SVIZ' and child[4] == 'FRAN':
            sviz_fran.append(float(child[5]))
        else:
            pass
    diz['nord-fran'] = np.nanmean(nord_fran)
    diz['nord-sviz'] = np.nanmean(nord_sviz)
    diz['fran-nord'] = np.nanmean(fran_nord)
    diz['fran-sviz'] = np.nanmean(fran_sviz)
    diz['sviz-fran'] = np.nanmean(sviz_fran)
    diz['sviz-nord'] = np.nanmean(sviz_nord)
    return diz
###############################################################################
def get_flowsXML(prepath, lof, start, end):
    diz = OrderedDict()
    nf = []
    ns = []
    for p in lof:
        path = prepath+p
        modify_XML(path)
        res = read_XML(path)
        nf.append(res['nord-fran'])        
        ns.append(res['nord-sviz'])
    diz['nord-fran'] = nf
    diz['nord-sviz'] = ns
    df = pd.DataFrame.from_dict(diz).set_index(pd.date_range(start, end,freq='D'))        
    return df
###############################################################################
    
list_of_dirs = [f for f in listdir('C:/Users/d_floriello/Documents/MGP_Transiti_Gen') if isdir(join('C:/Users/d_floriello/Documents/Transiti', f))]

DF = pd.DataFrame()
list_of_files = listdir('C:/Users/d_floriello/Documents/MGP_Transiti_Gen')
df = get_flowsXML('C:/Users/d_floriello/Documents/MGP_Transiti_Gen/',list_of_files, '2016-01-01', '2016-10-14')
DF = DF.append(df)

plt.figure()
DF.plot()
plt.axhline(y = 0)

DF.corr()

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
pun = data['PUN [€/MWH]'].dropna().resample('D').mean()

plt.figure()
plt.scatter(np.array(DF[DF.columns[0]]), np.array(pun))

np.corrcoef(np.array(DF[DF.columns[0]]), np.array(pun))

from pandas.tools import plotting

plt.figure()
plotting.lag_plot(DF['nord-fran'])


perc = []
signed_perc = []
for i in range(DF['nord-fran'].size - 1):
    perc.append(np.abs((DF['nord-fran'].ix[i+1] - DF['nord-fran'].ix[i])/DF['nord-fran'].ix[i]))
    signed_perc.append((DF['nord-fran'].ix[i+1] - DF['nord-fran'].ix[i])/DF['nord-fran'].ix[i])
    
plt.figure()
plt.plot(np.array(signed_perc))    
    