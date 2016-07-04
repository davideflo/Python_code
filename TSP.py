# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:57:54 2016

@author: d_floriello
"""

## Time Series Analysis of PUN in Python

import statsmodels
import statsmodels.api as sm
#from statsmodels.tsa import stattools
#from statsmodels.graphics import tsaplots
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

#RMSE = np.sqrt(np.mean(diff**2))

sp_y = Functions_for_TSP.Signum_Process(Y)
sp_f = Functions_for_TSP.Signum_Process(fitted_FE)

sp_p = sp_y * sp_f

perc_err_andamento = sp_p[sp_p <= 0].size/sp_p.size

aic = statsmodels.tsa.stattools.arma_order_select_ic(Y, ic = 'aic')


#D2 = D.convert_objects(convert_numeric=True)
D3 = D[D.columns[0:215]].convert_objects(convert_numeric=True)
#Dnp = float(D.as_matrix(columns=D.columns[0:215]))

ardata = pd.DataFrame(data[data.columns[[2,3,7,10,11,18,20]]])
ardata = ardata.convert_objects(convert_numeric=True)
ardata = ardata.set_index(prova)

slovsviz = ardata['SLOV'] - ardata['SVIZ']
slovfran = ardata['SLOV'] - ardata['FRAN']
austfran = ardata['AUST'] - ardata['FRAN']

## strana perfetta correlazione tra FRAN, SVIZ, SLOV e AUST.

aic = statsmodels.tsa.stattools.arma_order_select_ic(ardata["PUN"], ic = ['aic', 'bic'])
arma_y = statsmodels.tsa.arima_model.ARIMA(endog=ardata["PUN"], order=[4,1,2], 
                                           exog=ardata[ardata.columns[[2,4]]].as_matrix()).fit(trend = 'c', 
                                           method = 'mle', solver = 'newton', maxiter = 100)

#### when ARMA-like models are used, DO NOT use the DataFrame coming from create_dataset:
#### ARMA-like methods require only the time series (the lags are computed internally)
#### the DataFrame from create_dataset has to be used only in NN-like methods

arma_y = statsmodels.tsa.arima_model.ARIMA(endog=ardata["PUN"], order=[4,1,2]).fit(trend = 'c', method = 'mle', maxiter = 100)

arma_y.resid

RMSE = np.sqrt(np.mean(arma_y.resid**2))
arma_y.forecast(steps=24)

### provo modello PUN, ORA, GIORNO, HOLIDAY

vac = temp.add_holidays(dates) ## <-
ad = temp.associate_days(data[data.columns[1]], 'ven')
yd = temp.generate_days(data[data.columns[1]], 'ven')
anglesd = np.array([temp.convert_day_to_angle(v) for v in yd]) ## <- 
ora = np.sin(np.array(data[data.columns[1]])*np.pi/24) ## <-

arg = {'holiday' : vac, 'day' : anglesd, 'ora' : ora}
arg = pd.DataFrame(arg)

arfit = statsmodels.tsa.arima_model.ARIMA(endog=ardata["PUN"], order=[4,1,2],exog = arg.as_matrix()).fit(trend = 'c', method = 'mle', maxiter = 100)
rmse_fit = Functions_for_TSP.RMSE(arfit.resid) ## 7.7520042757584031

trainset = list(range(8016))
testset = list(range(8016,8760)) 

artfit = statsmodels.tsa.arima_model.ARIMA(endog=ardata["PUN"].ix[trainset], order=[4,1,2],exog = arg.ix[trainset].as_matrix()).fit(trend = 'c', method = 'mle', maxiter = 100)

art_forecast = artfit.forecast(steps = 744, exog = arg.ix[testset].as_matrix())

### http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.arima_model.ARMAResults.html ####
#RMSE(art_forecast[1]-ardata["PUN"].ix[testset])

############################################################
############################################################
###### prova su tutti i dataset ############################
############################################################

data2 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2011.xlsx")
data3 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2012.xlsx")
data4 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2013.xlsx")
data5 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2014.xlsx")
data6 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2015.xlsx")

vector_dates = np.concatenate([np.array(data[data.columns[0]]), np.array(data2[data2.columns[0]]), 
                               np.array(data3[data3.columns[0]]),
                         np.array(data4[data4.columns[0]]), np.array(data5[data5.columns[0]]), 
                            np.array(data6[data6.columns[0]])])
                            
vector_ore = np.concatenate([np.array(data[data.columns[1]]), np.array(data2[data2.columns[1]]), 
                               np.array(data3[data3.columns[1]]),
                         np.array(data4[data4.columns[1]]), np.array(data5[data5.columns[1]]), 
                            np.array(data6[data6.columns[1]])])

pun = np.concatenate([np.array(data[data.columns[2]]), np.array(data2[data2.columns[2]]), 
                               np.array(data3[data3.columns[2]]),
                         np.array(data4[data4.columns[2]]), np.array(data5[data5.columns[2]]), 
                            np.array(data6[data6.columns[2]])])


global_dates = temp.dates(pd.Series(vector_dates))
vac_glob = temp.add_holidays(global_dates) ###

all_days = temp.generate_days(vector_ore, 'ven')

aad = np.array([temp.convert_day_to_angle(v) for v in all_days]) ## <- 
aore = np.sin(np.array(vector_ore)*np.pi/24) ## <-


all_dict=  {'holiday' : vac_glob, 'day' : aad, 'ora' : aore}
tot_data = pd.DataFrame(all_dict)

aicg = statsmodels.tsa.stattools.arma_order_select_ic(pun, ic = ['aic','bic'])


tot_model = statsmodels.tsa.arima_model.ARIMA(endog=pun, order=[4,1,2],exog = tot_data.as_matrix()).fit(trend = 'c', method = 'mle', maxiter = 100)


dataf = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_04.xlsx")

datesf = temp.dates(dataf[dataf.columns[0]])
vacf = temp.add_holidays(datesf) ## <-
adf = temp.associate_days(dataf[dataf.columns[1]], 'ven')
ydf = temp.generate_days(dataf[dataf.columns[1]], 'ven')
anglesdf = np.array([temp.convert_day_to_angle(v) for v in ydf]) ## <- 
oraf = np.sin(np.array(dataf[dataf.columns[1]])*np.pi/24) ## <-

argf = {'holiday' : vacf, 'day' : anglesdf, 'ora' : oraf}
argf = pd.DataFrame(argf)

forecast_04_16 = tot_model.forecast(steps = 2903, exog = argf.as_matrix())

predicted_pun_04_16 = forecast_04_16[1]

pundf = pd.DataFrame(pun)
pundf = pundf.set_index(pd.to_datetime(global_dates))

dec_pun = sm.tsa.seasonal_decompose(pundf, freq=24)
dec_pun.plot()

min_per_seasonality = np.array(dec_pun.seasonal.ix[0:24])
plt.plot(min_per_seasonality)

diff_pred = predicted_pun_04_16 - dataf[dataf.columns[2]]

spp = Functions_for_TSP.Signum_Process(predicted_pun_04_16)
asp = Functions_for_TSP.Signum_Process(dataf[dataf.columns[2]])

ppp = spp*asp
ppp[ppp <= 0].size/ppp.size


## a little bit of tuning ##
params_d = [1,2,3,4,6,12,24]

for d in params_d:
    print("model with d = ", d)
    fit = statsmodels.tsa.arima_model.ARIMA(endog=pun, order=[4,d,2],exog = tot_data.as_matrix()).fit(trend = 'c', method = 'mle', maxiter = 100)
    pred = tot_model.forecast(steps = 2903, exog = argf.as_matrix())
    dd = pred[1] - dataf[dataf.columns[2]]
    print("computed RMSE:", RMSE(dd))
    spp = Functions_for_TSP.Signum_Process(pred[1])
    asp = Functions_for_TSP.Signum_Process(dataf[dataf.columns[2]])
    ppp = spp*asp
    print("error on sign process:", ppp[ppp <= 0].size/ppp.size)


##### eliminate trend and or seasonality ###
### ref: http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
### trying eliminate global seasonality:
pun_des = pun - np.repeat(min_per_seasonality, pun.size/24)
des_pun = sm.tsa.seasonal_decompose(pun_des, freq=24)
des_pun.plot()

params_d = [1,2]

for d in params_d:
    print("model with d = ", d)
    fit = statsmodels.tsa.arima_model.ARIMA(endog=pun_des, order=[24,d,24],exog = tot_data.as_matrix()).fit(trend = 'c', method = 'mle', maxiter = 100)
    pred = tot_model.forecast(steps = 2903, exog = argf.as_matrix())
    dd = pred[1] - dataf[dataf.columns[2]]
    print("computed RMSE:", RMSE(dd))
    spp = Functions_for_TSP.Signum_Process(pred[1])
    asp = Functions_for_TSP.Signum_Process(dataf[dataf.columns[2]])
    ppp = spp*asp
    print("error on sign process:", ppp[ppp <= 0].size/ppp.size)

pun2 = pd.Series(pun_des)
pun2.rolling(center = True, window= 24).mean()

####################################################################
############### new test ###########################################

fit = statsmodels.tsa.arima_model.ARIMA(endog=pun, order=[24,2,24],exog = tot_data.as_matrix()).fit(trend = 'c', maxiter = 100)
fit_des = statsmodels.tsa.arima_model.ARIMA(endog=pun_des, order=[24,2,24],exog = tot_data.as_matrix()).fit(trend = 'c', maxiter = 100)
