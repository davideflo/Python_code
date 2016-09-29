# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:15:20 2016

@author: d_floriello

Analysis of French and Swiss prices
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime

fs = pd.read_excel('C:/Users/d_floriello/Documents/Prezzi Francia e Svizzera (2015 -2016).xlsx', sheetname = '2016')
fs = fs[fs.columns[[2,3]]].set_index(fs['Data'])
fs.plot()

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
pun = data['PUN [€/MWH]'].resample('D').mean()

days = np.unique(fs.index) 

fsm = OrderedDict()

fr = []
sv = []
for d in days:
    fr.append(fs['Francia (EPEX)'].ix[fs.index == d].mean())
    sv.append(fs['Svizzera (EPEX)'].ix[fs.index == d].mean())
    
fsm['francia'] = fr
fsm['svizzera'] = sv
fsm['pun'] = pun[:273]

fsm = pd.DataFrame.from_dict(fsm)

fsm.plot() ### from this I'm very doubtful that flows actually correlate wth prices... 
           ### except in the last week where something anomalous is definitely happening

swiss_flows = pd.read_excel('C:/Users/d_floriello/Documents/EnergieUebersichtCH_2016.xlsx', sheetname = 'Zeitreihen0h15')
sflows = pd.DataFrame((1/1000)*swiss_flows[['Verbundaustausch CH->FR\nCross Border Exchange CH->FR',
                      'Verbundaustausch FR->CH\nCross Border Exchange FR->CH',
                      'Verbundaustausch CH->IT\nCross Border Exchange CH->IT',
                      'Verbundaustausch IT->CH\nCross Border Exchange IT->CH']].ix[1:])
rng = pd.date_range('2016-01-01', '2016-08-01', freq = '15T')
rng_true = rng[1:8264] + rng[8268:]

sflows = sflows.set_index(rng_true)
sflowsm = sflows.resample('D').sum()/sflows.resample('D').count()
sflowsm.plot(lw = 2)

ch_to_fr = pd.Series(sflowsm[sflows.columns[0]], dtype='float64')
fr_to_ch = pd.Series(sflowsm[sflows.columns[1]], dtype='float64')
ch_to_it = pd.Series(sflowsm[sflows.columns[2]], dtype='float64')
it_to_ch = pd.Series(sflowsm[sflows.columns[3]], dtype ='float64')

ch_to_it.corr(it_to_ch)
ch_to_fr.corr(fr_to_ch)
ch_to_it.corr(ch_to_fr)
ch_to_it.corr(fr_to_ch)

fran = pd.Series(fsm['francia'], dtype = 'float64')
sviz = pd.Series(fsm['svizzera'], dtype = 'float64')
pun = pd.Series(fsm['pun'], dtype = 'float64')

ch_to_it.corr(pun[:214])
ch_to_it.corr(sviz[:214])
ch_to_it.corr(fran[:214])

ch_to_fr.corr(fran[:214])
ch_to_fr.corr(sviz[:214])
ch_to_fr.corr(pun[:214])

it_to_ch.corr(fran[:214])
it_to_ch.corr(sviz[:214])
it_to_ch.corr(pun[:214])

fr_to_ch.corr(fran[:214])
fr_to_ch.corr(sviz[:214])
fr_to_ch.corr(pun[:214])

pun.corr(fran)
pun.corr(sviz)
sviz.corr(fran)

#################################
def weekly_(ts):
    dow = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    diz = OrderedDict()
    dm = ts.resample('D').mean()
    for d in dow:
        daily = []
        for i in range(ts.shape[0]):
            if dow[datetime.date(int(str(dm.index[i])[:4]),int(str(dm.index[i])[5:7]),int(str(dm.index[i])[8:10])).weekday()] == d:
                daily.append(dm.ix[i])
        diz[d] = daily
    diz = pd.DataFrame.from_dict(diz)
    return diz
#################################
wp = weekly_(pun)    
wf = weekly_(fran)
ws = weekly_(sviz)    

#### are we in a volatility cluster? it doesn't seem so for pun:
wp.T.plot(legend = False)
plt.figure()
plt.plot(wp.mean(axis=1))

plt.figure()
plt.plot(wp.var(axis = 1))

(wp.var(axis = 1) - np.mean(wp.var(axis = 1)))/np.std(wp.var(axis = 1))
####### fran is more variable than pun
wf.plot(legend = False)
plt.figure()
plt.plot(wf.mean(axis=1))

plt.figure()
plt.plot(wf.var(axis = 1))

(wf.var(axis = 1) - np.mean(wf.var(axis = 1)))/np.std(wf.var(axis = 1))
####### sviz is more similar to pun (in terms of variance)
ws.plot(legend = False)
plt.figure()
plt.plot(ws.mean(axis=1))

plt.figure()
plt.plot(ws.var(axis = 1))

(ws.var(axis = 1) - np.mean(ws.var(axis = 1)))/np.std(ws.var(axis = 1))

### "H1 norm" between processes ###
np.sqrt(np.mean((pun - sviz)**2) + np.mean((np.diff(pun) - np.diff(sviz))**2))
np.sqrt(np.mean((fran - sviz)**2) + np.mean((np.diff(fran) - np.diff(sviz))**2))
np.sqrt(np.mean((pun - fran)**2) + np.mean((np.diff(pun) - np.diff(fran))**2))

der_diz = OrderedDict()
der_diz['fran'] = np.diff(fran)
der_diz['sviz'] = np.diff(sviz)
der_diz['pun'] = np.diff(pun)

DD = pd.DataFrame.from_dict(der_diz)
DD.plot(lw = 2)

np.sign(DD['fran']) - np.sign(DD['sviz'])
np.sign(DD['pun']) - np.sign(DD['sviz'])
np.sum(np.sign(DD['fran']) - np.sign(DD['sviz']))
np.sum(np.sign(DD['pun']) - np.sign(DD['sviz']))
