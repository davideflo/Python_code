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

meteo = pd.read_table("C:/Users/d_floriello/Documents/PUN/storico_roma.txt", sep="\t")

variables = data.columns[[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20]]

DF = pd.concat([data[variables],data2[variables],data3[variables],data4[variables],
               data5[variables]], axis=0)

data6 = data6[variables]
tdf,ty = Functions_for_TSP.generate_dataset_ARIMA(data6,"gio",meteo, "CSUD")
               

df, y = Functions_for_TSP.generate_dataset_ARIMA(DF,"ven",meteo, "CSUD")

aicg = statsmodels.tsa.stattools.arma_order_select_ic(y, ic = ['aic','bic'],max_ar=24, max_ma=12)

tot_model = statsmodels.tsa.arima_model.ARIMA(endog=y, order=[24,1,12],exog = df.as_matrix()).fit(trend = 'c', maxiter = 100)
