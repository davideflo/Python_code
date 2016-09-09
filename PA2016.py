# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 11:28:29 2016

@author: utente

Pattern Analysis 2016
"""

import pandas as pd
import numpy as np
from functions_for_PA2016 import *
import matplotlib.pyplot as plt 
import statsmodels.api

data = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2016_08.xlsx", sheetname = 1)

data = np.array(data["PUN"])

for sig in range(10):
    print freq_greater_than(data, sig, False) 

stdata = (data - np.mean(data))/np.std(data)
i = np.where(stdata <  0)
stdata[i] = 0

plt.figure()
plt.plot(stdata)

stdiff = np.diff(stdata,n=1)
plt.figure()
plt.plot(stdiff)

rng = pd.date_range(start = '2016-01-01', periods = data.size, freq ='H')
D = pd.DataFrame(data).set_index(rng)

dm = pd.DataFrame(D.resample('D').mean())
dm = dm.set_index(pd.date_range(start = '2016-01-01', periods = dm.shape[0], fraq = 'D'))
dm.index = pd.DatetimeIndex(pd.date_range(start = '2016-01-01', periods = dm.shape[0], fraq = 'D'))
pd.infer_freq(dm)

dmdiff = np.diff(dm[dm.columns[0]])

dm.plot()
plt.figure()
plt.plot(dmdiff)

np.mean(dmdiff)

#dm.to_csv('daily_means_2016.csv', sep=',')
#pd.DataFrame(dmdiff).to_csv('diff_daily_means_2016.csv', sep=',')


#### decompose dm and dmdiff ####

for i in range(245):
    dec_dm = statsmodels.api.tsa.seasonal_decompose(dm, freq = 7)