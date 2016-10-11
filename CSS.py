# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:36:01 2016

@author: d_floriello

Analysis of Clean Spark Spread
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools import plotting
import statsmodels.api

spark = pd.read_excel('C:/Users/d_floriello/Documents/CSS_2015.xlsx')

spark = spark.set_index(spark['Giorno'])
spark.columns = ['Giorno','Ora','spread']

plt.figure()
plt.plot(spark['spread'].resample('D').mean())

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
pun = data['PUN [€/MWH]'].resample('D').mean()

plt.figure()
plt.plot(pun, color = 'grey')
plt.plot(spark['spread'].resample('D').mean())

plt.figure()
plt.scatter(spark['spread'].resample('D').mean(), pun)

plt.figure()
plt.plot(data['PUN [€/MWH]'], color = 'brown')
plt.plot(spark['spread'])

Diff = pun - spark['spread'].resample('D').mean()

plt.figure()
plt.plot(np.array(Diff))

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram((Diff - Diff.mean())/Diff.std()))

plt.figure()
plt.plot(np.array(Diff/pun))

spark_ts = pd.Series(spark['spread'].resample('D').mean(), dtype = 'float64')
pun_ts = pd.Series(pun, dtype = 'float64')

spark_ts.corr(pun_ts)

DS = spark['spread'].resample('D').mean()/pun

plt.figure()
plt.plot(DS)

DS.corr(pun_ts)

plt.figure()
plotting.lag_plot(DS)


plt.figure()
plt.plot(statsmodels.api.tsa.acf(np.array(DS)))
plt.plot(DS.resample('M').mean())
plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(DS)))

###############################################################################
def fourierExtrapolation(x, n_predict, n_harmonics = 0):
    x = np.array(x)
    n = x.size
    if n_harmonics == 0:
        n_harm = 100                     # number of harmonics in model
    else:
        n_harm = n_harmonics
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t        
###############################################################################
plt.figure()
plt.plot(np.array(Diff))
plt.plot(fourierExtrapolation(np.array(Diff), 0, 7), color = 'red')    
plt.figure()
plt.plot(np.array(Diff), color = 'grey')
plt.plot(fourierExtrapolation(np.array(Diff), 0, 52), color = 'purple')    


plt.figure()
plt.plot(np.array(DS))
plt.plot(fourierExtrapolation(np.array(DS), 0, 7), color = 'red')    
plt.figure()
plt.plot(np.array(DS), color = 'grey')
plt.plot(fourierExtrapolation(np.array(DS), 0, 52), color = 'purple')    
    
plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(spark['spread'].resample('D').mean())))
plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(pun)))
plt.title('pun periodogram')

plt.figure()
plt.hist(np.array(spark['spread'].resample('D').mean()))
plt.figure()
plt.plot(np.array(spark['spread'].resample('D').mean()))
plt.plot(fourierExtrapolation(np.array(spark['spread'].resample('D').mean()), 0, 9), color = 'red')        
    
    