# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:32:36 2017

@author: d_floriello

FORECASTING SYSTEM
"""

import FFDE
import datetime

today = datetime.datetime.now()

######
###### Files extraction and aggregation
FFDE.ZIPExtractor()

###### MO updating
mdf = FFDE.Aggregator(today)
mdf.to_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")

###### Web meteo scraping




