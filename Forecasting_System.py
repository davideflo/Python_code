# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:32:36 2017

@author: d_floriello

FORECASTING SYSTEM
"""

import FFDE
import datetime
import pandas as pd

today = datetime.datetime.now()

######
###### CRPP update ###### do it in python 2.X: 3 to 2 has problem with hdf 
DF6, DF7 = FFDE.updateCRPP()

#### manual completion
DF6 = pd.read_excel('C:/Users/d_floriello/Documents/CRPP2016.xlsx')
DF7 = pd.read_excel('C:/Users/d_floriello/Documents/CRPP2017.xlsx')
####
DF6.to_hdf('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/CRPP_2016.h5', 'DF2016')
DF7.to_hdf('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/CRPP_2017.h5', 'DF2017')
######
###### Files extraction and aggregation
FFDE.ZIPExtractor()

###### MO updating
mdf = FFDE.Aggregator(today)
mdf.to_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")

###### Web meteo scraping




