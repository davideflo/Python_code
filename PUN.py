# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:46:50 2016

@author: d_floriello
"""

## activate snakes
## to deactivate: deactivate


import pandas as pd
import numpy as np
from collections import OrderedDict
import h2o
import matplotlib.pyplot as plt
from lxml import objectify
#from bs4 import BeautifulSoup
path = 'C:/Users/d_floriello/Documents/MGP_Transiti_Gen/20160107MGPTransiti.xml'


data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')

data = data.set_index(data['Date'])
data = data[data.columns[0:32]]
data = data.dropna()


rng = pd.date_range(start = '2016-01-01', end = '2016-09-26', freq = 'D')
#####
zones = OrderedDict()
sard = []
sici = []
sud = []
csud = []
cnor = []
nord = []
pun = []
for d in rng:
    pun.append(data['PUN [€/MWH]'].ix[data.index.date == d.date].mean())
    sard.append(data['MGP SARD [€/MWh]'].ix[data.index.date == d].mean())
    sici.append(data['MGP SICI [€/MWh]'].ix[data.index.date == d].mean())
    sud.append(data['MGP SUD [€/MWh]'].ix[data.index.date == d].mean())
    csud.append(data['MGP CSUD [€/MWh]'].ix[data.index.date == d].mean())
    cnor.append(data['MGP CNOR [€/MWh]'].ix[data.index.date == d].mean())
    nord.append(data['MGP NORD [€/MWh]'].ix[data.index.date == d].dropna().mean())

zones['PUN'] = np.array(pun)    
zones['SARD'] = np.array(sard)
zones['SICI'] = np.array(sici)
zones['SUD'] = np.array(sud)
zones['CSUD'] = np.array(csud)
zones['CNOR'] = np.array(cnor)
zones['NORD'] = np.array(nord)

Z = pd.DataFrame.from_dict(zones).set_index(rng)
    
Z.plot()
######
Z = data.resample('D').mean()

zones = pd.DataFrame(Z[Z.columns[[6,7,10,13,15,16,18,21,22,23,24,25]]])

zones2 = zones.ix[zones.index.month <= 9]

zones.plot()
zones[zones.columns[[0,4,6]]].plot()
zones[zones.columns[[0,4]]].plot(title='PUN vs FRAN')
zones[zones.columns[[0,6]]].plot(title='PUN vs NORD')
zones[zones.columns[[6,4]]].plot(title='NORD vs FRAN')

zones[zones.columns[0]].corr(zones[zones.columns[4]])
zones[zones.columns[0]].corr(zones[zones.columns[6]])
zones[zones.columns[4]].corr(zones[zones.columns[6]])

#### FRAN normalized:
nor_fran = (zones[zones.columns[4]] - zones[zones.columns[4]].mean())/zones[zones.columns[4]].std() 

plt.figure()
plt.plot(nor_fran)

zones[zones.columns[0]].corr(zones[zones.columns[4]])
zones[zones.columns[0]].corr(zones[zones.columns[6]])
zones[zones.columns[4]].corr(zones[zones.columns[6]])
zones[zones.columns[0]].corr(zones[zones.columns[11]])
zones[zones.columns[11]].corr(zones[zones.columns[6]])
zones[zones.columns[4]].corr(zones[zones.columns[11]])

nor_sviz = (zones[zones.columns[11]] - zones[zones.columns[11]].mean())/zones[zones.columns[11]].std() 
nor_nord = (zones[zones.columns[6]] - zones[zones.columns[6]].mean())/zones[zones.columns[6]].std() 
nor_pun = (zones[zones.columns[0]] - zones[zones.columns[0]].mean())/zones[zones.columns[0]].std() 

plt.figure()
nor_fran.ix[nor_fran.index.month == 9].plot()
plt.figure()
nor_sviz.ix[nor_sviz.index.month == 9].plot()
plt.figure()
nor_pun.ix[nor_pun.index.month == 9].plot()
plt.figure()
nor_nord.ix[nor_nord.index.month == 9].plot()

sep = OrderedDict()
sep['fran'] = nor_fran.ix[nor_fran.index.month == 9]
sep['sviz'] = nor_sviz.ix[nor_sviz.index.month == 9]
sep['pun'] = nor_pun.ix[nor_pun.index.month == 9]
sep['nord'] = nor_nord.ix[nor_nord.index.month == 9]

Sep = pd.DataFrame.from_dict(sep)

nnsep = OrderedDict()
nnsep['fran'] = zones[zones.columns[4]].ix[zones.index.month == 9]
nnsep['sviz'] = zones[zones.columns[11]].ix[zones.index.month == 9]
nnsep['pun'] = zones[zones.columns[0]].ix[zones.index.month == 9]
nnsep['nord'] = zones[zones.columns[6]].ix[zones.index.month == 9]

NSep = pd.DataFrame.from_dict(nnsep)

#######################################################################
days = OrderedDict()
days_of_week = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
cols = [12,21,24,31]
nms = ['pun','fran','nord','sviz']
for i in cols:
    dm = []
    for d in days_of_week:
        dm.append(data[data.columns[i]].ix[data['Week Day'] == d].mean())
    days[nms[cols.index(i)]] = dm

days = pd.DataFrame.from_dict(days).set_index([days_of_week])
days.plot()

### trend in the last 2 months:
last_trend = OrderedDict()
last = data.ix[data.index.month >= 9]
days_of_week = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
cols = [12,21,24,31]
nms = ['pun','fran','nord','sviz']
for i in cols:
    dm = []
    for d in days_of_week:
        dm.append(last[last.columns[i]].ix[last['Week Day'] == d].mean())
    last_trend[nms[cols.index(i)]] = dm

LT = pd.DataFrame.from_dict(last_trend).set_index([days_of_week])
LT.plot()

(LT['pun'].ix['Mer'] - LT['pun'].ix['Gio'])/LT['pun'].ix['Gio']

#### pun quando ha sparato ####
sdates = ['2016-09-01','2016-09-02','2016-09-07','2016-09-20','2016-09-23']

shot = pd.DataFrame()
for sd in sdates:
    shot = shot.append(zones.ix[zones.index == sd])
    
before = ['2016-08-30','2016-08-31','2016-09-01','2016-09-05','2016-09-06','2016-09-18','2016-09-19',
          '2016-09-21', '2016-09-22']
for b in before:
    print('on {} pun was lower than sviz: {}'
    .format(b,zones[zones.columns[0]].ix[zones.index == b] < zones[zones.columns[11]].ix[zones.index == b]))
################################################
def freq_greater_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int((x - np.mean(ts))/np.std(ts) > sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
################################################
def glob_perc(ts):
    res = []
    for x in range(1, 10, 1):
        sigma = freq_greater_than(ts, x, True)
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            count = 0
            for i in range(out.size - 1):
                if out[i+1] - out[i] == 0:
                    count += 1
                else:
                    pass
            res.append(float(count/np.sum(sigma)))
            print('at distance {}'.format(x))    
            print('%.6f' % float(count/np.sum(sigma)))
    return np.array(res)
################################################
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
################################################    
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
#################################################
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
################################################

from os import listdir
from os.path import isfile, join
list_of_files = [f for f in listdir('C:/Users/d_floriello/Documents/MGP_Transiti_Gen') if isfile(join('C:/Users/d_floriello/Documents/MGP_Transiti_Gen', f))]

df = get_flowsXML('C:/Users/d_floriello/Documents/MGP_Transiti_Gen/',list_of_files, '2016-01-01', '2016-01-31')

plt.figure()
plt.plot(zones[zones.columns[0]] - zones[zones.columns[11]])

diz= OrderedDict()
diz['peaks_pun'] = glob_perc(data[data.columns[12]]) ## pun
diz['peaks_fran'] = glob_perc(data[data.columns[21]]) ## fran
diz['peaks_nord'] = glob_perc(data[data.columns[24]]) ## nord
diz['peaks_sviz'] = glob_perc(data[data.columns[31]]) ## sviz

diz= OrderedDict()
var = [12, 21, 24, 31]
for v in var:
    vals = []
    for i in range(10):
        print('peaks at distance {} for {}:'.format(i, data.columns[v]))
        print(freq_greater_than(data[data.columns[v]],i,False))
        vals.append(freq_greater_than(data[data.columns[v]],i,False))
    diz[data.columns[v]] = vals    

peaks = pd.DataFrame.from_dict(diz)

h2o.init()

path = 'C:/Users/d_floriello/Documents/MGP_Transiti2016091920160925/20160919MGPTransiti.xml'
xml = objectify.parse(open(path))

root = xml.getroot()
root.getchildren()[1].getchildren() 
 
for child in root:
    print(child.attrib)
 
for neighbor in root.iter('NewDataSet'):
    print(neighbor.attrib) 
 
for atype in xml.findall('MgpTransiti'):
    print(atype.get('Mercato')) 
 
for i in range(0,4):
    obj = root.getchildren()[i].getchildren()
    row = dict(zip(['id', 'name'], [obj[0].text, obj[1].text]))
    row_s = pd.Series(row)
    row_s.name = i
    df = df.append(row_s) 

######################################################################
dpun = np.diff(np.array(data[data.columns[12]].dropna().resample('D').mean()))

import statsmodels.api

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(dpun))

per = statsmodels.api.tsa.periodogram(dpun)
np.where(per > 50)[0]
per[per > 50]

import Fourier

reconstructed = Fourier.fourierExtrapolation(dpun, 0, 16)

plt.figure()
plt.plot(dpun)
plt.plot(reconstructed, color = 'red')

np.mean(dpun - reconstructed)
np.std(dpun - reconstructed)

from pandas.tools import plotting

plt.figure()
plotting.lag_plot(pd.DataFrame(dpun))

plt.figure()
plt.plot(statsmodels.api.tsa.acf(dpun))

lags = []
for i in range(dpun.size - 1):
    lags.append(np.array([dpun[i], dpun[i+1]]))
    
lags = pd.DataFrame(lags)
lags.corr()

plt.figure()
plotting.lag_plot(pd.DataFrame(dpun), lag = 7)
plt.figure()
plotting.autocorrelation_plot(pd.DataFrame(dpun))