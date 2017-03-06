# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:37:14 2017

@author: d_floriello

DataAggregator
"""


import pandas as pd
#import unidecode

data1 = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Anno '+str(2014)+'.xlsx', sheetname = 'Prezzi-Prices')
data2 = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2015/06. MI/DB_Borse_Elettriche.xlsx", sheetname = 'DB_Dati')
data3 = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')
data4 = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2017/06. MI/DB_Borse_Elettriche_PER MI_17_conMacro - Copy.xlsm", sheetname = 'DB_Dati')
#
#data1.columns = [unidecode.unidecode(x) for x in data1.columns.tolist()]
#data2.columns = [unidecode.unidecode(x) for x in data2.columns.tolist()]
#data3.columns = [unidecode.unidecode(x) for x in data3.columns.tolist()]
#data4.columns = [unidecode.unidecode(x) for x in data4.columns.tolist()]


pun = []
pun.append(data1['PUN'].values.ravel())
pun.append(data2[data2.columns[12]].values.ravel())
pun.append(data3[data3.columns[12]].dropna().values.ravel())
pun.append(data4[data4.columns[12]].dropna().values.ravel())


unlisted =  [item for sublist in pun for item in sublist]


df = pd.DataFrame(unlisted) ######### to: 2 DAYS AHEAD OF LAST PUN
df = df.set_index(pd.date_range('2014-01-01', '2018-01-02', freq = 'H')[:df.shape[0]])


df.to_excel('C:/Users/d_floriello/Documents/R/dati_2014-2017.xlsx')



