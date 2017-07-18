# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:20:27 2017

@author: utente

Fit Levy processes for Trading Simulation
"""

import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from sklearn.linear_model import LinearRegression

####################################################################################################
def hurst(ts):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0
####################################################################################################



ger = pd.read_excel('C:/Users/utente/Documents/Trading/Chiusura GER_15_16.xlsx')
ger.columns = [['Data', 'CAL', 'YEAR']]

ger2015 = ger.ix[ger['YEAR'] == 2015]
ger2016 = ger.ix[ger['YEAR'] == 2016]


levy_ger15 = scipy.stats.levy.fit(ger2015['CAL'])

plt.figure()
ger2015['CAL'].plot()
plt.figure()
ger2015['CAL'].plot(kind = 'hist', bins = 20)

sim15 = scipy.stats.levy.rvs(levy_ger15[0], levy_ger15[1], size = 5*ger2015.shape[0])

sim15[np.where(sim15 > ger2015['CAL'].max())[0]] = ger2015['CAL'].max()

plt.figure()
plt.hist(sim15, bins = 20, color = 'teal')
plt.figure()
plt.plot(sim15, color = 'teal')

print hurst(ger2015['CAL'].values.ravel())
print hurst(ger2016['CAL'].values.ravel())

#### Try fitting an Ornstein–Uhlenbeck process

plt.figure()
plotting.autocorrelation_plot(ger2015['CAL'].values.ravel())
plt.figure()
plotting.lag_plot(ger2015['CAL'])

X = ger2015['CAL'].values.ravel()[:-1]
y = ger2015['CAL'].values.ravel()[1:]

lm = LinearRegression(fit_intercept = True)

lm.fit(X.reshape(-1,1), y)

a = lm.coef_[0]
b = lm.intercept_

Sxy = np.sum(X*y)
Sx = np.sum(X)
Sy = np.sum(y)
Sxx = np.sum(X**2)
Syy = np.sum(y**2)
n = X.size

a = (n*Sxy - Sx*Sy)/(n*Sxx - Sx**2)
b = (Sy - a*Sx)/(n)
sdepsilon = np.sqrt((n*Syy - Sy**2 - a*(n*Sxy - Sx*Sy))/(n*(n-2)))

residuals = y - lm.predict(X.reshape(-1,1))

lam = -np.log(a)
mu = b/(1 - a)
sigma = sdepsilon*np.sqrt(2*lam/(1 - a**2))

S0 = np.mean(X)
S = [S0]
for i in range(1,229):
    rnv = scipy.stats.norm.rvs()
    s = lam * (mu - S[i-1]) + sigma * rnv + np.mean(X)
    S.append(s)
    
plt.figure()
plt.plot(np.array(S), color = 'lawngreen')