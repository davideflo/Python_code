# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:57:23 2016

@author: utente

analysis dependence pun - load
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

fabb = pd.read_csv('C:/Users/utente/Documents/PUN/demand_curves.csv')

fabb.mean(axis = 1).plot()

data = pd.read_excel("C:/Users/utente/Documents/PUN/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')

data = data.set_index(data['Date'])
data = data[data.columns[0:32]]

data = data.ix[data.index.month >= 9]
data = data.dropna()
data = data.ix[144:]

data = data.resample('D').mean()

plt.figure()
plt.scatter(fabb.mean(axis = 1), data[data.columns[6]])

lm = linear_model.LinearRegression(fit_intercept = True).fit(np.array(fabb.mean(axis = 1)).reshape(-1,1), np.array(data[data.columns[6]]))

yhat = lm.predict(np.array(fabb.mean(axis = 1)).reshape(-1,1))

plt.figure()
plt.plot(np.array(fabb.mean(axis = 1)),yhat)
plt.scatter(fabb.mean(axis = 1), data[data.columns[6]])

R2 = 1 - np.sum((np.array(data[data.columns[6]]) - yhat)**2)/(np.sum((np.array(data[data.columns[6]]) - np.mean(np.array(data[data.columns[6]])))**2))

res = np.array(data[data.columns[6]]) - yhat
np.mean(res)
np.std(res)

lm.coef_
beta_0 = lm.predict(0)

plt.figure()
plt.plot(np.linspace(-1, 42, num = 100),lm.predict(np.linspace(-1, 42, num = 100).reshape(-1,1)))
plt.scatter(fabb.mean(axis = 1), data[data.columns[6]])
