# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:01:48 2016

@author: d_floriello

stand-alone self-pun-computizer
"""

import pandas as pd
import SPC
import sys

print('Insert YEAR MONTH DAY')
#path = sys.argv[1]
year = sys.argv[1]
month = sys.argv[2]
day =sys.argv[3]

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
pun = data['PUN [€/MWH]'].dropna().resample('D').mean()

print('values for tomorrow = {}'.format(SPC.Forecast_(pun, int(year), int(month), int(day))))
