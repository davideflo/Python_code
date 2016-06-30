# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:57:54 2016

@author: d_floriello
"""

## Time Series Analysis of PUN in Python

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa import stattools
from statsmodels.graphics import tsaplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions_for_TSP
import statsmodels.tsa.arima_model  

data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")

import temp

pun = data["PUN"]
dates = temp.dates(data[data.columns[0]])
prova = pd.to_datetime(dates)
df = pd.DataFrame(pun)
df = df.set_index(prova)

dec = sm.tsa.seasonal_decompose(df, freq=24)
dec.plot()

D,Y = temp.create_dataset(data, "ven")
Y = np.array(Y)

acf, Q, P, = statsmodels.tsa.stattools.acf(Y, nlags = 48, qstat = True)
statsmodels.graphics.tsaplots.plot_acf(Y, lags=1000)

per = statsmodels.tsa.stattools.periodogram(Y)

plt.plot(per)

S_per = pd.Series(per)
S_per.describe()

peaks = Functions_for_TSP.find_peaks(per, 10)

FE = Functions_for_TSP.fourierExtrapolation(Y, n_predict = 24)

fitted_FE = FE[0:8736]

diff = Y - fitted_FE
np.mean(diff)
np.var(diff)

RMSE = np.sqrt(np.mean(diff**2))

sp_y = Functions_for_TSP.Signum_Process(Y)
sp_f = Functions_for_TSP.Signum_Process(fitted_FE)

sp_p = sp_y * sp_f

perc_err_andamento = sp_p[sp_p <= 0].size/sp_p.size

aic = statsmodels.tsa.stattools.arma_order_select_ic(Y, ic = 'aic')


#D2 = D.convert_objects(convert_numeric=True)
D3 = D[D.columns[0:215]].convert_objects(convert_numeric=True)
#Dnp = float(D.as_matrix(columns=D.columns[0:215]))
arma_y = statsmodels.tsa.arima_model.ARIMA(endog=Y, order=[4,1,2], exog=D3.as_matrix()).fit(trend = 'c', method = 'mle', solver = 'newton', maxiter = 100)

#### when ARMA-like models are used, DO NOT use the DataFrame coming from create_dataset:
#### ARMA-like methods require only the time series (the lags are computed internally)
#### the DataFrame from create_dataset has to be used only in NN-like methods





