# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:33:20 2016

@author: d_floriello

Analisi Francia - Svizzera - PUN 2015
"""

import pandas as pd
import numpy as np
#from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import linear_model

fs = pd.read_excel('C:/Users/d_floriello/Documents/Prezzi Francia e Svizzera (2015 -2016).xlsx', sheetname = '2015')
fs = fs[fs.columns[[2,3]]].set_index(fs['Data'])
fs.plot()

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
pun = data['PUN [€/MWH]'].resample('D').mean()

fran = fs[fs.columns[0]].resample('D').mean()

plt.figure()
plt.plot(pun)
plt.plot(fran)


###############################################################################

fm = fs[fs.columns[0]].resample('M').mean()
m = [5,5,7,10,12,16,19,18,12,10,10,8]
M = [8,8,13,19,20,25,28,27,20,16,14,11]

med = []
for i in range(12):
    med.append((m[i] + M[i])/2)

lm = linear_model.LinearRegression(fit_intercept = True).fit(np.array(med).reshape(-1,1), np.array(fm).reshape(-1,1))
lmr = linear_model.RANSACRegressor(linear_model.LinearRegression()).fit(np.array(med).reshape(-1,1), np.array(fm).reshape(-1,1))

lm.coef_

yhat = lm.predict(np.linspace(start = 0, stop = 30, num = 60).reshape(-1,1))
yrhat = lmr.predict(np.linspace(start = 0, stop = 30, num = 60).reshape(-1,1))

label = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
plt.figure()
plt.plot(np.linspace(start = 0, stop = 30, num = 60), yhat.ravel())
plt.plot(np.linspace(start = 0, stop = 30, num = 60), yrhat.ravel(), color = 'black')
plt.scatter(np.array(med), np.array(fm), color = 'red', marker = 'o')
for i,l in enumerate(label):
    plt.annotate(l, xy = (np.array(med)[i],np.array(fm)[i]))
###############################################################################
print(np.mean(pun))    
print(np.std(pun))
print(np.max(pun))
print(np.min(pun))
print(np.mean(fran))    
print(np.std(fran))
print(np.max(fran))
print(np.min(fran))