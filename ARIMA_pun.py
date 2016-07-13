# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:32:30 2016

@author: d_floriello
"""

import statsmodels
import statsmodels.api as sm
#from statsmodels.tsa import stattools
#from statsmodels.graphics import tsaplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions_for_TSP
import statsmodels.tsa.arima_model  

import temp


data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")
data2 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2011.xlsx")
data3 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2012.xlsx")
data4 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2013.xlsx")
data5 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2014.xlsx")
data6 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2015.xlsx")

data7 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_06.xlsx")


roma = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_roma.txt", sep="\t")
milano = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_milano.txt", sep="\t")
torino = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_torino.txt", sep="\t")
#milano = pd.read_table("storico_milano_aggiornato.txt", sep="\t")
ca = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_cagliari.txt", sep="\t")
pa = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_palermo.txt", sep="\t")
rc = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_reggiocalabria.txt", sep="\t")
fi = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_firenze.txt", sep="\t")


###################################################################
###### find missing dates #########################################
###################################################################

vector_dates = np.concatenate([np.array(data[data.columns[0]]), np.array(data2[data2.columns[0]]), 
                               np.array(data3[data3.columns[0]]),
                         np.array(data4[data4.columns[0]]), np.array(data5[data5.columns[0]]), 
                            np.array(data6[data6.columns[0]])])
                            
global_dates = temp.dates(pd.Series(vector_dates))

missing = Functions_for_TSP.finding_missing_dates(global_dates, milano)
missing2 = Functions_for_TSP.finding_missing_dates(global_dates, torino)

test = Functions_for_TSP.update_meteo(milano, torino)
test.to_csv('storico_milano_aggiornato.txt', sep="\t", index = False)
missing2 = Functions_for_TSP.finding_missing_dates(global_dates, test)

missingca = Functions_for_TSP.finding_missing_dates(global_dates, ca)
missingpa = Functions_for_TSP.finding_missing_dates(global_dates, pa)
missingrc = Functions_for_TSP.finding_missing_dates(global_dates, rc)
missingfi = Functions_for_TSP.finding_missing_dates(global_dates, fi)

diffmiro = Functions_for_TSP.simulate_meteo(milano,roma)
difffiro = Functions_for_TSP.simulate_meteo(fi,roma)
diffparo = Functions_for_TSP.simulate_meteo(pa,roma)
diffcaro = Functions_for_TSP.simulate_meteo(ca,roma)
diffrcro = Functions_for_TSP.simulate_meteo(rc,roma)

################ study and simulation meteo #####################

milano_date = pd.to_datetime(milano[milano.columns[0]])
mmedia = pd.DataFrame(milano['Tmedia'])
mmedia = mmedia.set_index(milano_date)

milano_dec = sm.tsa.seasonal_decompose(mmedia, freq=365)
milano_dec.plot()

roma_date = pd.to_datetime(roma[roma.columns[0]])
rmedia = pd.DataFrame(roma['Tmedia'])
rmedia = rmedia.set_index(roma_date)

roma_dec = sm.tsa.seasonal_decompose(rmedia, freq=365)
roma_dec.plot()

plt.plot(roma['Tmedia'])

upfi = Functions_for_TSP.generate_simulated_meteo_dataset(fi,roma)

#################################################################
variables = data.columns[[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20]]

DF = pd.concat([data[variables],data2[variables],data3[variables],data4[variables],
               data5[variables]], axis=0)

data6 = data6[variables]
tdf,ty = Functions_for_TSP.generate_dataset_ARIMA(data6,"gio",roma, "CSUD")
     
tdfcov = pd.concat([tdf,pd.Series(ty)],axis=1)          
np.linalg.det(tdfcov.corr().as_matrix())

df, y = Functions_for_TSP.generate_dataset_ARIMA(DF,"ven",roma, "CSUD")

dfcov = pd.concat([df,pd.Series(y)],axis=1)
np.linalg.det(dfcov.corr().as_matrix())

aicg = statsmodels.tsa.stattools.arma_order_select_ic(y, ic = ['aic','bic'],max_ar=24, max_ma=12)

tot_model = statsmodels.tsa.arima_model.ARIMA(endog=y, order=[24,1,12],exog = df.as_matrix()).fit(trend = 'c', maxiter = 100)

#for i in range(1,20,1):
#    plt.plot(pd.ewma(pd.Series(y), span=8670))

forecast = tot_model.forecast(steps = y.size, exog = df.as_matrix())
dec = sm.tsa.seasonal_decompose(forecast[0], freq=24)
dec.plot()
dec2 = sm.tsa.seasonal_decompose(y, freq=24)
dec2.plot()

## maybe with this model it's better to learn models and forecast for things like max and min

X, Y = temp.create_dataset_arima(DF, 'ven', 'CSUD', roma)
dec_y = sm.tsa.seasonal_decompose(Y, freq=5)
dec_y.plot()


aicg = statsmodels.tsa.stattools.arma_order_select_ic(Y, ic = ['aic','bic'],max_ar=24, max_ma=12)
