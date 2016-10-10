# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:08:34 2016

@author: d_floriello

BTC analysis
"""

import pandas as pd
import statsmodels.api
import numpy as np
import matplotlib.pyplot as plt

btc = pd.read_csv('C:/Users/d_floriello/Documents/bitcoin.csv')

btc = btc.set_index(btc['Date'])
btc.head()
btc = btc[['Open', 'High', 'Low', 'Close']]

plt.figure()
btc['Close'].plot()

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(btc['Close'])))
plt.title('BTC periodogram')