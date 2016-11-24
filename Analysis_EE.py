# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:27:06 2016

@author: utente

Analysis EE daily curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import datetime

data = pd.read_excel('C:/Users/utente/Documents/misure/Cartel2.xlsx')

val_col = map(str, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])


dt = []
for i in range(data.shape[0]):
    dt.append(pd.to_datetime(data[['Data ']].ix[i]))

data.set_index(data[['Data ']])

sbs = (data[['Data ']].values.ravel() == np.datetime64('2016-11-01')).tolist()

idd = dt.index(pd.to_datetime('2016-11-01'))

data2 = data.ix[sbs]

data2[val_col].ix[data['Area'] == 'SICI'].T.plot()

data3 = data2[val_col].ix[data['Area'] == 'SICI'].reset_index(drop=True)

plt.figure()
for i in range(data3.shape[0]):
    P = np.poly1d(np.polyfit(np.linspace(1,24,24), data3.ix[i], 5))
    plt.plot(np.linspace(1,25,1001),P(np.linspace(1,25,1001)))
    plt.title('5th degree')


### same pod ###
pod = 'IT001E90532162'
dtl = ['2016-11-01','2016-11-02','2016-11-03','2016-11-04','2016-11-05',
       '2016-11-06','2016-11-07','2016-11-08','2016-11-09','2016-11-10',
       '2016-11-11','2016-11-12','2016-11-13','2016-11-14','2016-11-15',
       '2016-11-16','2016-11-17','2016-11-18','2016-11-19','2016-11-20']

f, axarr = plt.subplots(2, sharex=True)
smooth = OrderedDict()
real = OrderedDict()
for i in range(len(dtl)):
    sbs = (data[['Data ']].values.ravel() == np.datetime64(dtl[i])).tolist()
    td = data.ix[sbs]
    td = td[val_col].ix[td['POD'] == pod]
    Pp = np.poly1d(np.polyfit(np.linspace(1,24,24), np.array(td).ravel(), 7))
    axarr[0].plot(np.array(td).ravel())
    axarr[1].plot(np.linspace(1,24,1001),Pp(np.linspace(1,24,1001)))
    pol = Pp(np.linspace(1,24,1001))  
    rr = np.array(td).ravel().tolist()
    ll = Pp(np.linspace(1,24,1001)).tolist()
    if pol[np.linspace(1,24,1001).tolist().index(15.007)]  > 80:
        ll.insert(0, 'L')
        rr.insert(0, 'L')
    else:
        ll.insert(0, 'F')
        rr.insert(0, 'F')
#    plt.plot(np.linspace(1,24,1001),Pp(np.linspace(1,24,1001)))
    smooth[i] = ll
    real[i] = rr

smooth = pd.DataFrame.from_dict(smooth, orient = 'index')
real =  pd.DataFrame.from_dict(real, orient = 'index')

L = np.where(np.array(smooth[[0]] == 'L'))[0]
F = np.where(np.array(smooth[[0]] == 'F'))[0]

plt.figure()
smooth.ix[L.tolist()].mean().plot()
smooth.ix[L.tolist()].std().plot()
smooth.ix[F.tolist()].mean().plot()
smooth.ix[F.tolist()].std().plot()

plt.figure()
real.ix[L.tolist()].boxplot()
real.ix[F.tolist()].boxplot(col = 'black')


#### only SICI ####
sumreal = OrderedDict()
f, axarr = plt.subplots(2, sharex=True)
for i in range(len(dtl)):
    sbs = (data[['Data ']].values.ravel() == np.datetime64(dtl[i])).tolist()
    td = data.ix[sbs]
    td = td[val_col].ix[td['Area'] == 'SICI'].sum()
    Pp = np.poly1d(np.polyfit(np.linspace(1,24,24), np.array(td).ravel(), 7))
    axarr[0].plot(np.array(td).ravel())
    axarr[1].plot(np.linspace(1,24,1001),Pp(np.linspace(1,24,1001)))
    pol = Pp(np.linspace(1,24,1001))  
    rr = np.array(td).ravel().tolist()
    if datetime.date(2016, 11, int(dtl[i][8:10])).weekday() < 5:
        rr.insert(0, 'L')
    else:
        rr.insert(0, 'F')
#    plt.plot(np.linspace(1,24,1001),Pp(np.linspace(1,24,1001)))
    sumreal[i] = rr

sumreal = pd.DataFrame.from_dict(sumreal, orient = 'index')

boxpropsL = dict(linestyle='-', linewidth=2, color='black')
medianpropsL = dict(linestyle='-', linewidth=2, color='black')
boxpropsF = dict(linestyle='-', linewidth=2, color='red')
medianpropsF = dict(linestyle='-', linewidth=2, color='red')
plt.figure()
sumreal.ix[L.tolist()].boxplot(boxprops = boxpropsL, medianprops = medianpropsL, showmeans = True)
plt.title('SICI-L')
sumreal.ix[F.tolist()].boxplot(boxprops = boxpropsF, medianprops = medianpropsF, showmeans = True)
plt.title('SICI-F')

plt.figure()
sumreal.ix[L.tolist()].mean().plot(color = 'blue')
sumreal.ix[F.tolist()].mean().plot(color = 'red')
sumreal.ix[L.tolist()].std().plot(color = 'turquoise')
sumreal.ix[F.tolist()].std().plot(color = 'coral')

plt.figure()
(sumreal.ix[L.tolist()].std()/sumreal.ix[L.tolist()].sum()).plot()

###################################################################################################
def PlotBoxPlot(zona):
    sumreal = OrderedDict()
    f, axarr = plt.subplots(2, sharex=True)
    for i in range(len(dtl)):
        sbs = (data[['Data ']].values.ravel() == np.datetime64(dtl[i])).tolist()
        td = data.ix[sbs]
        td = td[val_col].ix[td['Area'] == zona].sum()
        Pp = np.poly1d(np.polyfit(np.linspace(1,24,24), np.array(td).ravel(), 7))
        axarr[0].plot(np.array(td).ravel())
        axarr[1].plot(np.linspace(1,24,1001),Pp(np.linspace(1,24,1001)))
        rr = np.array(td).ravel().tolist()
        if datetime.date(2016, 11, int(dtl[i][8:10])).weekday() < 5:
            rr.insert(0, 'L')
        else:
            rr.insert(0, 'F')
    
        sumreal[i] = rr
    
    sumreal = pd.DataFrame.from_dict(sumreal, orient = 'index')
    
    boxpropsL = dict(linestyle='-', linewidth=2, color='black')
    medianpropsL = dict(linestyle='-', linewidth=2, color='black')
    boxpropsF = dict(linestyle='-', linewidth=2, color='red')
    medianpropsF = dict(linestyle='-', linewidth=2, color='red')
    plt.figure()
    sumreal.ix[L.tolist()].boxplot(boxprops = boxpropsL, medianprops = medianpropsL, showmeans = True)
    plt.title(zona+'-L')
    plt.figure()
    sumreal.ix[F.tolist()].boxplot(boxprops = boxpropsF, medianprops = medianpropsF, showmeans = True)
    plt.title(zona+'-F')
    return sumreal
####################################################################################################
    
PlotBoxPlot('NORD')
PlotBoxPlot('CNOR')
PlotBoxPlot('CSUD')
PlotBoxPlot('SUD')
PlotBoxPlot('SICI')
PlotBoxPlot('SARD')




