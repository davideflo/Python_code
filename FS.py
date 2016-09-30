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
from lxml import objectify
from os import listdir
from os.path import join, isdir
from sklearn import linear_model

##########################################################################################################
def modify_XML(path):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    file = open(path, 'w')
    todelete = ['<xs', '</xs']
    for line in lines:
        if todelete[0] in line:
            print('skip')
        elif todelete[1] in line:
            print('skip this too')
        else:
            print('write')
            file.write(line.replace(',','.'))
    file.close()
################################################    
def read_XML(path):
#    var = ['NORD', 'FRAN', 'SVIZ']
    xml = objectify.parse(open(path))
    root = xml.getroot()
    nord_fran = []
    fran_sviz = []
    sviz_nord = []
    nord_sviz = []
    sviz_fran = []
    fran_nord= []
    diz = OrderedDict()
    for i in range(len(root.getchildren())):
        child = root.getchildren()[i].getchildren()
        if child[3] == 'NORD' and child[4] == 'FRAN':
            nord_fran.append(float(child[5]))
        elif child[3] == 'NORD' and child[4] == 'SVIZ':
            nord_sviz.append(float(child[5]))
        elif child[3] == 'FRAN' and child[4] == 'SVIZ':
            fran_sviz.append(float(child[5]))
        elif child[3] == 'FRAN' and child[4] == 'NORD':
            fran_nord.append(float(child[5]))
        elif child[3] == 'SVIZ' and child[4] == 'NORD':
            sviz_nord.append(float(child[5]))
        elif child[3] == 'SVIZ' and child[4] == 'FRAN':
            sviz_fran.append(float(child[5]))
        else:
            pass
    diz['nord-fran'] = np.nanmean(nord_fran)
    diz['nord-sviz'] = np.nanmean(nord_sviz)
    diz['fran-nord'] = np.nanmean(fran_nord)
    diz['fran-sviz'] = np.nanmean(fran_sviz)
    diz['sviz-fran'] = np.nanmean(sviz_fran)
    diz['sviz-nord'] = np.nanmean(sviz_nord)
    return diz
#################################################
def get_flowsXML(prepath, lof, start, end):
    diz = OrderedDict()
    nf = []
    ns = []
    for p in lof:
        path = prepath+p
        modify_XML(path)
        res = read_XML(path)
        nf.append(res['nord-fran'])        
        ns.append(res['nord-sviz'])
    diz['nord-fran'] = nf
    diz['nord-sviz'] = ns
    df = pd.DataFrame.from_dict(diz).set_index(pd.date_range(start, end,freq='D'))        
    return df
##########################################################################################################

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

###############################################################################
#### importing all transit datasets:
list_of_dirs = [f for f in listdir('C:/Users/d_floriello/Documents/Transiti') if isdir(join('C:/Users/d_floriello/Documents/Transiti', f))]

DF = pd.DataFrame()
for lod in list_of_dirs:
    start = lod[12:16]+'/'+lod[16:18]+'/'+lod[18:20]
    end = lod[20:24]+'/'+lod[24:26]+'/'+lod[26:28]
    print(start)
    print(end)    
    list_of_files = listdir('C:/Users/d_floriello/Documents/Transiti/'+lod)
    df = get_flowsXML('C:/Users/d_floriello/Documents/Transiti/'+lod+'/',list_of_files, start, end)
    DF = DF.append(df)

DF.plot(lw = 2)
DF.head()
pun.size

nf = pd.Series(DF['nord-fran'], dtype='float64')
ns = -pd.Series(DF['nord-sviz'], dtype='float64')

nf.corr(pun[:nf.size])
ns.corr(pun[:ns.size])

plt.figure()
plt.scatter(nf,pun[:nf.size])

lr = linear_model.LinearRegression(fit_intercept = True)
lin_model = lr.fit(np.array(nf).reshape(-1,1), np.array(pun[:nf.size]).reshape(-1,1))

y_hat = lin_model.predict(np.array(nf).reshape(-1,1))
lin_model.coef_ #### not much impact of transits on pun

plt.figure()
plt.scatter(nf,pun[:nf.size], color = 'black')
plt.plot(nf, y_hat, color = 'red', lw = 2)

y_hat_mean = lin_model.predict(np.array(0).reshape(-1,1))
np.mean(y_hat)
pun[:nf.size].mean()

##################################################################
std_nf = (nf - nf.mean())/nf.std()

pun_s = pun[:nf.size]

more = pun_s.ix[std_nf > 1]
less = pun_s.ix[std_nf < -1]
bet1 = pun_s.ix[-1 <= std_nf]
bet = bet1.ix[std_nf <= 1]

bet.mean()
more.mean()
less.mean()
grand_mean = pun_s.mean()

(more.mean() - bet.mean())/bet.mean()
(more.mean() - grand_mean)/grand_mean
(less.mean() - bet.mean())/bet.mean()
(less.mean() - grand_mean)/grand_mean

############ correlation pun - gas
gas = pd.read_excel('C:/Users/d_floriello/Documents/Annotermico2015-2016_08.xlsx', sheetname = 'PB-GAS G+1')
gas = gas.set_index(gas['Giorno gas'])
gas.tail()
gas = pd.DataFrame(gas['Prezzo'].ix[2:])

gas.plot()
gas.describe()

mask = (gas.index > datetime.datetime(2015, 12, 31)) & (gas.index <= datetime.datetime(2016, 8, 31))
gas_ts = pd.Series(gas['Prezzo'].loc[mask], dtype='float64')
gas_ts.shape

gas_ts.corr(pun[:gas_ts.shape[0]])

plt.figure()
plt.scatter(np.array(gas_ts), np.array(pun[:gas_ts.shape[0]]))

pun_g = pun[:gas_ts.shape[0]]
lr_g = linear_model.LinearRegression(fit_intercept = True)
lin_model_gas = lr.fit(np.array(gas_ts).reshape(-1,1), np.array(pun_g).reshape(-1,1))

g_hat = lin_model_gas.predict(np.array(gas_ts).reshape(-1,1))
lin_model_gas.coef_ #### not much impact of transits on pun

plt.figure()
plt.scatter(np.array(gas_ts),np.array(pun_g), color = 'blue')
plt.plot(np.array(gas_ts), g_hat, color = 'red', lw = 2)
