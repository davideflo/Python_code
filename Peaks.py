# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:46:30 2017

@author: utente

Sbilanciamento 14 -- Peaks analysis
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime
from pandas.tools import plotting
from collections import OrderedDict

####################################################################################################
def MahalanobisDistance(x):
    mu = np.mean(x)
    sigma = np.std(x)
    ds = []
    for i in range(x.size):
        ds.append((x[i] - mu)/sigma)
    return np.array(ds)
####################################################################################################

db = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
db.columns = [str(i) for i in db.columns]
db = db[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]
db = db.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')

dbz = db.ix[db['Area'] == zona]
dbz = dbz.set_index(dbz.Giorno)

dbzre = dbz.resample('D').sum()/1000

maxS = dbz.resample('D').sum().max(axis = 1)/1000
plt.figure()
plotting.autocorrelation_plot(maxS, color = 'red')

diz = OrderedDict()
maxS = []
dow = []
hours = []
for i in range(dbzre.shape[0]):
    maxS.append(dbzre.ix[i].max())
    dow.append(dbzre.index[i].weekday())
    hours.append(dbzre.ix[i].values.ravel().tolist().index(max(dbzre.ix[i])))

diz['M'] = maxS
diz['dow'] = dow
diz['H'] = hours

diz = pd.DataFrame.from_dict(diz, orient = 'columns')

plt.figure()
diz.H.hist(bins = 20)

c = ['red', 'navy', 'green', 'magenta', 'grey', 'black', 'violet']
for i in range(7):
    plt.figure()
    plotting.autocorrelation_plot(diz.M.ix[diz.dow == i], color = c[i])
    
plt.figure()
plt.scatter(diz.dow.values.ravel(), diz.H.values.ravel())
plt.figure()
plt.scatter(diz.dow.values.ravel(), diz.M.values.ravel(), color = 'black')
plt.figure()
plt.scatter(diz.M.values.ravel(), diz.H.values.ravel(), color = 'red')

RIC = rical

dates = []
for i in range(RIC.shape[0]):
    dates.append(RIC.Giorno.ix[i].replace(hour = RIC.Ora.ix[i]))
    
dates = np.array(dates)
sric = RIC[RIC.columns[2:]].sum(axis = 1)/1000
sric = pd.DataFrame({'S': sric.values.ravel()}).set_index(pd.to_datetime(dates))

plt.figure()
sric.plot(color = 'coral')

diz = OrderedDict()
maxS = []
dow = []
hours = []
month = []
for i in pd.date_range('2017-01-01', '2017-12-31', freq = 'D'):
    maxS.append(sric.ix[sric.index.date == i.date()].max().values[0])
    dow.append(i.weekday())
    hours.append(sric.ix[sric.index.date == i.date()].values.ravel().tolist().index(max(sric.ix[sric.index.date == i.date()].values.ravel())))
    month.append(i.month)
    
diz['M'] = maxS
diz['dow'] = dow
diz['H'] = hours
diz['month'] = month

diz = pd.DataFrame.from_dict(diz, orient = 'columns')

plt.figure()
diz.H.hist(bins = 20)

c = ['red', 'navy', 'green', 'magenta', 'grey', 'black', 'violet']
for i in range(7):
    plt.figure()
    plotting.autocorrelation_plot(diz.M.ix[diz.dow == i], color = c[i])
    
plt.figure()
plt.scatter(diz.dow.values.ravel(), diz.H.values.ravel())
plt.figure()
plt.scatter(diz.dow.values.ravel(), diz.M.values.ravel(), color = 'black')
plt.figure()
plt.scatter(diz.H.values.ravel(), diz.M.values.ravel(), color = 'red')
plt.figure()
plt.scatter(diz.month.values.ravel(), diz.M.values.ravel(), color = 'salmon')


for i in range(7):
    print i, diz.M.ix[diz.dow == i].std(), diz.M.ix[diz.dow == i].mean()
    
for i in range(7):
    plt.figure()
    plt.hist(MahalanobisDistance(diz.M.ix[diz.dow == i].values.ravel()), bins = 20)    
    #diz.M.ix[diz.dow == i].hist(bins = 20)

for m in range(1,13):
    for i in range(7):
        dizm = diz.ix[diz.month == m]
        print 'month, day, std, mean, skew: {}, {}, {}, {}, {}'.format(m, i, dizm.M.ix[dizm.dow == i].std(), dizm.M.ix[dizm.dow == i].mean(), scipy.stats.skew(dizm.M.ix[dizm.dow == i].values.ravel()))
#        print 'skewtest: {}'.format(scipy.stats.skewtest(dizm.M.ix[dizm.dow == i].values.ravel()))
    
for m in range(1,13):
    for i in range(7):
        dizm = diz.ix[diz.month == m]
        sigma = dizm.M.ix[dizm.dow == i].std() 
        mu = dizm.M.ix[dizm.dow == i].mean()
   
PK = range(8,20)
             
ott = dbz.resample('D').sum()/1000
ott = ott.ix[ott.index.month == 9]
ott['DOW'] = np.array(map(lambda date: date.weekday(), ott.index))

fer_ott = ott.ix[ott.index.date != datetime.date(2017,10,1)]
fer_ott = fer_ott.ix[fer_ott.index.date != datetime.date(2017,10,7)]
fer_ott = fer_ott.ix[fer_ott.index.date != datetime.date(2017,10,8)]

fer_ott = ott.ix[ott.DOW == 1]
fer_ott = fer_ott[fer_ott.columns[:-1]]

ottm = fer_ott.mean()
ottsig = fer_ott.std()

##### abs error
PKer = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0])
for i in fer_ott.index:
    print 'absolute error: {}'.format(PKer*(fer_ott.ix[i].values.ravel() - (ottm - ottsig).values.ravel()))
    print 'MAE: {}'.format(PKer*(fer_ott.ix[i].values.ravel() - (ottm - ottsig).values.ravel())/(ottm - ottsig).values.ravel())
    
    
pk_counter = 0
for i in range(diz.shape[0]):
    if diz.H.ix[i] in PK:
        pk_counter += 1
        
float(pk_counter)/float(diz.shape[0])