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
import scipy.stats

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
pun = data['PUN [€/MWH]'].dropna().resample('D').mean()

days = np.unique(fs.index) 

fsm = OrderedDict()

fr = []
sv = []
for d in days:
    fr.append(fs['Francia (EPEX)'].ix[fs.index == d].mean())
    sv.append(fs['Svizzera (EPEX)'].ix[fs.index == d].mean())
    
###  EPEX FR from 2016-09-30 to 2016-10-27:
nd = [41.78, 38.19, 32.48, 36.04, 42.02,42.68 ,48.28,
      57.29208333	, 44.35375,	36.705,56.70208333, 71.21208333, 62.81041667, 64.25, 64.10, 44.28, 40.02, 56.41,66.94,
      67.69,	76.30,	72.95,	55.72,	44.57,	72.63,
      79.92, 70.53, 61.49]
      
for n in nd:
    fr.append(n)
fsm['francia'] = fr
fsm['svizzera'] = sv
fsm['pun'] = pun[:287]

fsm = pd.DataFrame.from_dict(fsm)

fsm.plot() ### from this I'm very doubtful that flows actually correlate wth prices... 
           ### except in the last week where something anomalous is definitely happening

############## analysis of correlations ###############
fsm = fsm.set_index(pun.index)
cors = []
for i in range(2,pun.shape[0],1):
    cors.append(np.corrcoef(np.array(pun)[:i],np.array(fsm).ravel()[:i])[1,0])

compl_cors = []
for i in range(2,pun.shape[0],1):
    compl_cors.append(np.corrcoef(np.array(pun)[pun.shape[0] - i:],np.array(fsm).ravel()[pun.shape[0] - i:])[1,0])
    
plt.figure()
plt.plot(np.array(cors))
plt.figure()
plt.plot(np.array(compl_cors))

ottp = pun.ix[pun.index.month == 10]
ottf = fsm.ix[fsm.index.month == 10]
nottp = pun.ix[pun.index.month < 10]
nottf = fsm.ix[fsm.index.month < 10]

corr_ottobre = ottp.corr(ottf['francia'])
corr_else = []
corr_upto = []
for i in range(1, 10, 1):
    corr_else.append(nottp.ix[nottp.index.month == i].corr(nottf['francia'].ix[nottf.index.month == i]))
    corr_upto.append(nottp.ix[nottp.index.month <= i].corr(nottf['francia'].ix[nottf.index.month <= i]))

corr_else = []
corr_upto = []
for i in range(1, 11, 1):
    corr_else.append(pun.ix[pun.index.month == i].corr(fsm['francia'].ix[fsm.index.month == i]))
    corr_upto.append(pun.ix[pun.index.month <= i].corr(fsm['francia'].ix[fsm.index.month <= i]))

plt.figure()
plt.plot(np.array(corr_else), marker = 'o')
plt.plot(np.array(corr_upto), marker = '*')
plt.scatter(np.array([9]), np.array([corr_ottobre]), color = 'black', marker = 'D')

dpun = pd.Series(np.diff(pun), index = pd.date_range('2016-01-02', '2016-10-26', freq = 'D'))
dfran = pd.Series(np.diff(fsm['francia']), index = pd.date_range('2016-01-02', '2016-10-26', freq = 'D'))
   
dcorr_else = []
dcorr_upto = []
for i in range(1, 11, 1):
    dcorr_else.append(dpun.ix[dpun.index.month == i].corr(dfran.ix[dfran.index.month == i]))
    dcorr_upto.append(dpun.ix[dpun.index.month <= i].corr(dfran.ix[dfran.index.month <= i]))

plt.figure()
plt.plot(np.array(dcorr_else), marker = 'o', color = 'grey')
plt.plot(np.array(dcorr_upto), marker = '*', color = 'magenta')   
################# monthwise volatility and percentage increment ###############
volp = pun.resample('M').std()
volf = fsm['francia'].resample('M').std()

plt.figure()
plt.plot(volp, marker = 'o')
plt.plot(volf, marker = '*')

meanp = pun.resample('M').mean()
meanf = fsm['francia'].resample('M').mean()

percp = []
percf = []
for i in range(meanp.size - 1):
    percp.append((meanp[i+1] - meanp[i])/meanp[i])
    percf.append((meanf[i+1] - meanf[i])/meanf[i])
    
plt.figure()
plt.plot(np.array(percp), marker = 'o')
plt.plot(np.array(percf), marker = '*')
###################################

plt.figure()
plt.plot(fsm['pun'])
plt.plot(fsm['francia'])
plt.plot(fsm['pun']-fsm['francia'])
plt.axhline(y=0)

qd = scipy.stats.mstats.mquantiles(np.array(fsm['pun']-fsm['francia']), prob = [0.025, 0.975])
qp = scipy.stats.mstats.mquantiles(np.array(fsm['pun']), prob = 0.95)
qf = scipy.stats.mstats.mquantiles(np.array(fsm['francia']), prob = 0.95)

plt.figure()
plt.plot(np.array(fsm['pun']))
plt.scatter(np.where(np.array(fsm['pun']) > qp)[0], np.array(fsm['pun'])[np.where(np.array(fsm['pun']) > qp)[0]], color = 'black', marker = 'o')
plt.plot(np.array(fsm['francia']))
plt.scatter(np.where(np.array(fsm['francia']) > qf)[0], np.array(fsm['francia'])[np.where(np.array(fsm['francia']) > qf)[0]], color = 'black', marker = '*')
plt.plot(np.array(fsm['pun']-fsm['francia']))
plt.scatter(np.where(np.array(fsm['pun']-fsm['francia']) > qd[1])[0], np.array(fsm['pun']-fsm['francia'])[np.where(np.array(fsm['pun']-fsm['francia']) > qd[1])[0]], color = 'black', marker = 'o')
plt.scatter(np.where(np.array(fsm['pun']-fsm['francia']) < qd[0])[0], np.array(fsm['pun']-fsm['francia'])[np.where(np.array(fsm['pun']-fsm['francia']) < qd[0])[0]], color = 'magenta', marker = 'o')
plt.grid()
plt.axhline(y=0)

rel_diff = (fsm['pun']-fsm['francia'])/fsm['pun']
fsm['francia'].ix[rel_diff <= 0].corr(fsm['pun'].ix[rel_diff <= 0])

np.diff(np.array(fsm['francia'].ix[rel_diff <= 0]))
np.diff(np.array(fsm['pun'].ix[rel_diff <= 0]))

plt.figure()
plt.scatter(np.array(fsm['pun']-fsm['francia']),np.array(fsm['pun']))
################
s_pun = (fsm['pun'] - fsm['pun'].mean())/fsm['pun'].std() 
s_fran = (fsm['francia'] - fsm['francia'].mean())/fsm['francia'].std() 

import scipy as sp
qsp = sp.stats.mstats.mquantiles(s_pun, prob = 0.95)[0]
qsf = sp.stats.mstats.mquantiles(s_fran, prob = 0.95)[0]

out_p = fsm['pun'].ix[np.abs(s_pun) > qsp]
out_f = fsm['francia'].ix[np.abs(s_fran) > qsf]

plt.figure()
plt.plot(np.array(fsm['pun']))
plt.plot(np.array(fsm['francia']))
plt.scatter(np.where(np.abs(s_pun) > qsp)[0],out_p, color = 'purple', marker = 'o')
plt.scatter(np.where(np.abs(s_fran) > qsf)[0],out_f, color = 'black', marker = '*')
plt.title('(global) outliers')

def find_outliers_rolling(ts):
    index = []
    for i in range(1, ts.size, 1):
        st_ts_i = (ts[:i] - np.mean(ts[:i]))/np.std(ts[:i])        
        qu = sp.stats.mstats.mquantiles(st_ts_i, prob = 0.95)[0]
        index.append(np.where(np.abs(st_ts_i) > qu)[0].tolist())
    return index

rop_i = find_outliers_rolling(np.array(fsm['pun'])) 

unp = set(rop_i[-1])

for i in range(len(rop_i)-1):
    unp = unp.union(rop_i[i])
 
unp = list(unp)

plt.figure()
plt.plot(np.array(fsm['pun']))
plt.scatter(np.array(unp), np.array(fsm['pun'].ix[unp]), color = 'black', marker = 'o')

rof_i = find_outliers_rolling(np.array(fsm['francia'])) 

unf = set(rof_i[-1])

for i in range(len(rof_i)-1):
    unf = unf.union(rof_i[i])
 
unf = list(unf)

plt.figure()
plt.plot(np.array(fsm['francia']), color = 'red')
plt.scatter(np.array(unf), np.array(fsm['francia'].ix[unf]), color = 'purple', marker = 'o')

plt.figure()
plt.plot(np.array(fsm['pun']))
plt.plot(np.array(fsm['francia']), color = 'red')
plt.scatter(np.array(unp), np.array(fsm['pun'].ix[unp]), color = 'black', marker = 'o')
plt.scatter(np.array(unf), np.array(fsm['francia'].ix[unf]), color = 'purple', marker = 'o')
plt.title('rolling outliers')


##########################################################################################################
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

wpq = wp.quantile([0.0275, 0.975])
wfq = wf.quantile([0.0275, 0.975])

lun = []
mar = []
mer = []
gio = []
ven = []
sab = []
dom = []

dow = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
diz = OrderedDict()
first = datetime.date(int(str(pun.index[0])[:4]),int(str(pun.index[0])[5:7]),int(str(pun.index[0])[8:10]))
for d in dow:
    daily = []
    for i in range(pun.shape[0]):
        dt = datetime.date(int(str(pun.index[i])[:4]),int(str(pun.index[i])[5:7]),int(str(pun.index[i])[8:10]))
        if dow[dt.weekday()] == d:
            if pun.ix[i] <= wpq[d].ix[0.0275] or pun.ix[i] >= wpq[d].ix[0.975]:             
                daily.append((dt - first).days)
    diz[d] = daily
    
markers = ["1", "2", "3", "4", "8", "s", "p"]
plt.figure()
plt.plot(np.array(pun), color = 'purple')
for k in diz.keys():
    m = markers[dow.index(k)]
    plt.scatter(np.array(diz[k]), np.array(fsm['pun'].ix[diz[k]]), color = 'black', marker = m)


dow = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
diz = OrderedDict()
first = datetime.date(int(str(fran.index[0])[:4]),int(str(fran.index[0])[5:7]),int(str(fran.index[0])[8:10]))
for d in dow:
    daily = []
    for i in range(fran.shape[0]):
        dt = datetime.date(int(str(fran.index[i])[:4]),int(str(fran.index[i])[5:7]),int(str(fran.index[i])[8:10]))
        if dow[dt.weekday()] == d:
            if fran.ix[i] <= wfq[d].ix[0.0275] or fran.ix[i] >= wfq[d].ix[0.975]:             
                daily.append((dt - first).days)
    diz[d] = daily
    
markers = ["1", "2", "3", "4", "8", "s", "p"]
plt.figure()
plt.plot(np.array(fran), color = 'grey')
for k in diz.keys():
    m = markers[dow.index(k)]
    plt.scatter(np.array(diz[k]), np.array(fsm['francia'].ix[diz[k]]), color = 'blue', marker = m)
   
def weekly_area(df):
    wa = []
    for i in range(df.shape[0]):
        area = 0        
        week = np.array(df.ix[i])
        dweek = np.diff(week)
        for j in range(5):
            area += (week[j] + week[j+1])*(1/2) + (dweek[j] + dweek[j+1])*(1/2)
        area += (week[5] + week[6])*(1/2)
        wa.append(area)
    return np.array(wa)
 
wap = weekly_area(wp)   
waf = weekly_area(wf)    
    
    
q_wap = sp.stats.mstats.mquantiles(wap, prob = [0.05, 0.95])
q_waf = sp.stats.mstats.mquantiles(waf, prob = [0.05, 0.95])

plt.figure()
plt.plot(wap)  
plt.axhline(q_wap[0], color = 'blue')
plt.axhline(q_wap[1], color = 'blue')
plt.plot(waf)    
plt.axhline(q_waf[0], color = 'green')
plt.axhline(q_waf[1], color = 'green')

    
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
lin_model_gas = lr_g.fit(np.array(gas_ts).reshape(-1,1), np.array(pun_g).reshape(-1,1))

g_hat = lin_model_gas.predict(np.array(gas_ts).reshape(-1,1))
lin_model_gas.coef_

plt.figure()
plt.scatter(np.array(gas_ts),np.array(pun_g), color = 'blue')
plt.plot(np.array(gas_ts), g_hat, color = 'red', lw = 2)

from scipy import stats

result = stats.linregress(gas_ts, pun_g)
result.slope

err = pun_g - g_hat.ravel()
plt.figure()
plt.scatter(list(range(err.size)),err) 
err.mean()
err.median()

stud_err = (err - err.mean())/(err.std())
plt.figure()
plt.scatter(list(range(stud_err.size)),stud_err) 

plt.figure()
plt.scatter(np.array(gas_ts), stud_err)

plt.figure()
plt.scatter(g_hat.ravel(), stud_err)

gas_ts
plt.figure()
plt.plot(gas_ts)

########## complete gas prices

gp = pd.read_excel('C:/Users/d_floriello/Documents/Gas prices (1).xlsx', sheetname = 'Spot indices')
gp = gp.ix[1:]
gp = gp.set_index(gp[gp.columns[0]])
bal = gp['Balancing market price (within day)'].dropna()

bal.plot()

plt.figure()
plt.plot(np.array(bal))
############# prezzo sbilanciamento

gpb = pd.read_excel('C:/Users/d_floriello/Documents/Prezzo di bilanciamento gas.xlsx', sheetname = 'val')
gpb = gpb.set_index(pd.to_datetime(gpb[gpb.columns[0]]))
bil = gpb['sbil']

plt.figure()
plt.plot(np.array(bil))
bil6 = bil.ix[306:550]

pun_g.corr(bil6)

pun_g.size
bil6.size

plt.figure()
plt.scatter(np.array(bil6), pun_g, color = 'red')

res6 = stats.linregress(bil6, pun_g)
pg_hat = res6.intercept + bil6 * res6.slope

plt.figure()
plt.scatter(np.array(bil6), pun_g, color = 'red')
plt.plot(np.array(bil6), pg_hat, color = 'blue')

res = pun_g - pg_hat
res_stud = (res - np.mean(res))/np.std(res)

plt.figure()
plt.scatter(list(range(res_stud.size)),res_stud) 
plt.scatter(list(range(stud_err.size)),stud_err, color = 'lime') 


np.mean(res)
np.std(res)

pun_in = pun_g.ix[np.where((pun_g - np.mean(pun_g))/np.std(pun_g) <= 3)]
bil_in6 = bil6.ix[np.where((pun_g - np.mean(pun_g))/np.std(pun_g) <= 3)]

stats.linregress(bil_in6, pun_in)

#########################
diff_pf = fsm['pun'] - fsm['francia']

plt.figure()
plt.plot(diff_pf)
plt.plot(pun)

from pandas.tools import plotting

plt.figure()
plotting.lag_plot(fsm['pun'])
plt.figure()
plotting.lag_plot(diff_pf)

diff_pf.corr(pun)

plt.figure()
plt.scatter(np.array(diff_pf), np.array(pun[1:281]))

##################################

plt.figure()
plt.scatter(np.array(fsm['francia']), np.array(fsm['pun']))


fplm = linear_model.LinearRegression(fit_intercept = True).fit(np.array(fsm['francia']).reshape(-1,1),np.array(fsm['pun']))

yhat = fplm.predict(np.array(fsm['francia']).reshape(-1,1))

plt.figure()
plt.scatter(np.array(fsm['francia']), np.array(fsm['pun']))
plt.plot(np.array(fsm['francia']).reshape(-1,1), yhat)

R2 = 1 - np.sum((fsm['pun'] - yhat)**2)/np.sum((fsm['pun'] - fsm['pun'].mean())**2)

np.mean(fsm['pun'] - yhat)
np.std(fsm['pun'] - yhat)

###############################################################################

fpspread = np.array(pun) - fsm['francia']

plt.figure()
plt.plot(fpspread)

cm = []
for i in range(1, fpspread.size, 1):
    cm.append(np.mean(fpspread[:i]))

plt.figure()
plt.plot(np.array(cm), color = 'black')
plt.figure()
plt.hist(np.array(cm))

CM = pd.DataFrame(cm)
from pandas.tools import plotting

plt.figure()
plotting.lag_plot(CM)

cmpun = []
cvpun = []
for i in range(1, pun.size, 1):
    cmpun.append(np.mean(pun.ix[:i]))
    cvpun.append(np.std(pun.ix[:i]))

plt.figure()
plt.plot(np.array(cmpun))
plt.plot(np.array(cmpun) + np.array(cvpun), color = 'black')
plt.plot(np.array(cmpun) - np.array(cvpun), color = 'black')


CMP = pd.DataFrame(cmpun)
plt.figure()
plotting.lag_plot(CMP)

lmcmp = linear_model.LinearRegression(fit_intercept = True).fit(np.array(CMP.ix[0:289]).reshape(-1,1), np.array(CMP[1:291]))
lmcmp.coef_

yhat = lmcmp.predict(np.array(CMP.ix[0:289]).reshape(-1,1))

R2 = 1- np.sum((np.array(CMP[1:291]) - yhat)**2)/np.sum((np.array(CMP[1:291]) - np.mean(np.array(CMP[1:291])))**2)

plt.figure()
plotting.lag_plot(CMP)
plt.plot(np.array(CMP.ix[0:289]), yhat, color = 'red')

################## tuning ##########################
for d in range(1,6,1):
    mean_trend = np.polyfit(np.linspace(0, 291, 291), CMP, d)
    pol = np.poly1d(mean_trend.ravel())
    plt.figure()
    plt.plot(np.array(CMP))
    plt.plot(np.linspace(0, 291, 1000), pol(np.linspace(0, 291, 1000)), color = 'grey')
    plt.title('polynomial of degree{}'.format(d))
#################################################### d = 5
    
diff_spread = np.diff(fpspread)

plt.figure()
plt.plot(diff_spread)

plt.figure()
plt.hist(np.array(CMP))
plt.axvline(x = 38.963584, color = 'black', lw = 2)

np.median(CMP)
np.mean(CMP)
scipy.stats.skew(CMP)
scipy.stats.kurtosis(CMP)

from sklearn.neighbors.kde import KernelDensity

bws = [0.1,0.2,0.5,0.8,1,2,3,4]
for h in bws:
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(CMP)
    ss = np.exp(kde.score_samples(CMP))
    plt.figure()
    plt.plot(np.array(CMP),ss)
    plt.title('kde with bandwidth = {}'.format(h))
##### take  h = 2
#### candidate distributions: 
#### translated gamma, exponential, chi2    
fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(np.array(CMP)) 

sampled = scipy.stats.gamma.rvs(fit_alpha, loc = fit_loc, scale = fit_beta, size = CMP.size)
   
plt.figure()
plt.hist(np.array(CMP))
plt.hist(sampled)

diz = {'cmp': np.array(CMP).ravel(), 'sam': sampled}
plt.figure()
pd.DataFrame.from_dict(diz).hist()    

scipy.stats.gamma.cdf(np.inf, a = fit_alpha, loc = fit_loc, scale = fit_beta) 
1 - scipy.stats.gamma.cdf(38.963584, a = fit_alpha, loc = fit_loc, scale = fit_beta)

###############################################################################
def get_prob_mu(x, emp_sigma):
    sup = scipy.stats.gamma.cdf(x + emp_sigma, a = fit_alpha, loc = fit_loc, scale = fit_beta)    
    inf = scipy.stats.gamma.cdf(x, a = fit_alpha, loc = fit_loc, scale = fit_beta)
    return sup - inf
###############################################################################
###### test for 2016-10-19

mean_trend = np.polyfit(np.linspace(0, 291, 291), CMP, 5)
pol = np.poly1d(mean_trend.ravel())

get_prob_mu(pol(292), cvpun[-1])

###############################################################################
def divide_in_days(CMP):
    
    day = OrderedDict()
    
    lun = [] ## 0
    mar = []
    mer = []
    gio = []
    ven = []
    sab = []
    dom = [] ## 6
    
    for i in range(CMP.shape[0]):    
        dt = datetime.date(int(str(CMP.index[i])[:4]),int(str(CMP.index[i])[5:7]),int(str(CMP.index[i])[8:10]))
        if dt.weekday() == 0:
            lun.append(CMP.ix[i].values[0])
        elif dt.weekday() == 1:
            mar.append(CMP.ix[i].values[0])
        elif dt.weekday() == 2:
            mer.append(CMP.ix[i].values[0])
        elif dt.weekday() == 3:
            gio.append(CMP.ix[i].values[0])
        elif dt.weekday() == 4:
            ven.append(CMP.ix[i].values[0])
        elif dt.weekday() == 5:
            sab.append(CMP.ix[i].values[0])
        else:
            dom.append(CMP.ix[i].values[0])
    
    szs = [len(lun), len(mar), len(mer), len(gio), len(ven), len(sab), len(dom)]
    M = max(szs)
    
    if len(lun) < M:
        lun.append(np.nan)
    if len(mar) < M:
        mar.append(np.nan)
    if len(mer) < M:
        mer.append(np.nan)
    if len(gio) < M:
        gio.append(np.nan)
    if len(ven) < M:
        ven.append(np.nan)
    if len(sab) < M:
        sab.append(np.nan)
    if len(dom) < M:
        dom.append(np.nan)
            
    day['lun'] = lun        
    day['mar'] = mar        
    day['mer'] = mer        
    day['gio'] = gio        
    day['ven'] = ven        
    day['sab'] = sab        
    day['dom'] = dom        
    
    DBD = pd.DataFrame.from_dict(day)

    return DBD
###############################################################################
rng = pd.date_range('2016-01-02', '2016-10-18', freq = 'D')
CMP = CMP.set_index(rng)

day = OrderedDict()

lun = [] ## 0
mar = []
mer = []
gio = []
ven = []
sab = []
dom = [] ## 6

for i in range(CMP.shape[0]):    
    dt = datetime.date(int(str(CMP.index[i])[:4]),int(str(CMP.index[i])[5:7]),int(str(CMP.index[i])[8:10]))
    if dt.weekday() == 0:
        lun.append(CMP.ix[i].values[0])
    elif dt.weekday() == 1:
        mar.append(CMP.ix[i].values[0])
    elif dt.weekday() == 2:
        mer.append(CMP.ix[i].values[0])
    elif dt.weekday() == 3:
        gio.append(CMP.ix[i].values[0])
    elif dt.weekday() == 4:
        ven.append(CMP.ix[i].values[0])
    elif dt.weekday() == 5:
        sab.append(CMP.ix[i].values[0])
    else:
        dom.append(CMP.ix[i].values[0])

mer.append(np.nan)
gio.append(np.nan)
ven.append(np.nan)
        
day['lun'] = lun        
day['mar'] = mar        
day['mer'] = mer        
day['gio'] = gio        
day['ven'] = ven        
day['sab'] = sab        
day['dom'] = dom        

DBD = pd.DataFrame.from_dict(day)
DBD.plot.box()

DP = divide_in_days(pd.DataFrame(pun))
DP.plot.box()

import seaborn as sns

plt.figure()
sns.pairplot(DP.fillna(0), vars = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom'], kind = 'reg')
DP.corr()


mp = pun.resample('M').mean()
plt.figure()
mp.plot()

fitted_trend = np.polyfit(np.linspace(0,10,10), np.array(mp), 5)
yh = np.poly1d(fitted_trend)

plt.figure()
plt.plot(np.array(mp))
plt.plot(np.linspace(0,10, 1000), yh(np.linspace(0,10, 1000)), color = 'black')

err_4 = np.array(mp) - yh(np.linspace(0,10,10))
err5 = np.array(mp) - yh(np.linspace(0,10,10))

np.mean(err_4)
np.mean(err5)
np.std(err_4)
np.std(err5)

plt.figure()
pd.DataFrame([err_4,err5]).T.hist()

###############################################################################
def de_trend(df, trend):
    dt = []
    yh = trend(np.linspace(0,df.shape[0],df.shape[0]))
    for i in range(df.shape[0]):
        mon = yh[(df.index.month[i] - 1)]
        print('month {} and correction {}'.format(df.index.month[i],mon))
        dt.append(df.ix[i] - mon)
    return pd.DataFrame(dt)
###############################################################################
det = de_trend(pun, yh)

plt.fgure()
det.plot()

###############################################################################
def remainderizer(df):
    ### comupe trend in months:
    diz = OrderedDict()
    dow = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom'] 
    mp = np.array(df.resample('M').mean())
    dt = []
    for i in range(df.shape[0]):
        mon = mp[(df.index.month[i] - 1)]
        print('month {} and correction {}'.format(df.index.month[i],mon))
        dt.append(df.ix[i] - mon)
    dt = pd.DataFrame(dt).set_index(df.index)
    ### remove monthly seasonality:
    MONTHS = np.unique(dt.index.month)
    des = []
    for m in MONTHS:
        lm = []
        loc_dt = dt.ix[dt.index.month == m]
        dd = divide_in_days(loc_dt)
        diz[str(m)+'_mean'] = dd.mean()
        diz[str(m)+'_std'] = dd.std()
        for j in range(loc_dt.shape[0]):
            die = datetime.date(int(str(loc_dt.index[j])[:4]),int(str(loc_dt.index[j])[5:7]),int(str(loc_dt.index[j])[8:10]))
            giorno = die.weekday()
            x = (loc_dt.ix[j] - dd[dow[giorno]].mean())/dd[dow[giorno]].std()
            lm.append(x)
        des.append(lm)
    flattened_des = [item for sublist in des for item in sublist]
    rem = pd.DataFrame(flattened_des)
    seas = pd.DataFrame.from_dict(diz).set_index([dow])
    return mp, seas, rem
###############################################################################
### for modelling process, look at:
# 1) http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/#cox-ingersoll-ross
# 2) https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing

mm, sea, remn = remainderizer(pun)
remn.plot()
plt.figure()
sea.plot()

plt.figure()
plotting.lag_plot(remn)
plt.figure()
plotting.autocorrelation_plot(remn)

plt.figure()
sns.lmplot(x = 'x', y = 'y',data = pd.DataFrame({'x': np.array(remn.ix[0:(remn.shape[0]-1)].values.ravel()), 'y': np.array(remn.ix[1:(remn.shape[0])].values.ravel())}))

import statsmodels.api
rem_per = statsmodels.api.tsa.periodogram(remn.values.ravel()).ravel()
scipy.stats.mstats.mquantiles(rem_per, prob = 0.95)
scipy.stats.mstats.mquantiles(remn, prob = 0.95)
### is remn stationary?
statsmodels.api.tsa.adfuller(remn.values.ravel()) ## looks like it

### fit an autoregressive model
remn = remn.values.ravel()
plt.figure()
#### trying different ARMA models
arma_remn = statsmodels.api.tsa.ARMA(remn, (1,0)).fit()
arma_remn.summary()
print(arma_remn.params)
arma_pred = arma_remn.predict()

R22 = 1 - np.sum((remn - arma_pred)**2)/np.sum((remn - np.mean(remn))**2)

plt.figure()
plt.plot(remn)
plt.plot(arma_pred, color = 'red')

resid = remn - arma_pred
np.mean(resid)
np.std(resid)

###############################################################################
def Forecast(dataset, steps_ahead):
    forecasted = []
    arma1 = statsmodels.api.tsa.ARMA(dataset, (1,0)).fit()
    forecasted.append(arma1.params[0] + arma1.params[1] * dataset[-1])
    dataset = np.concatenate((dataset, np.array(forecasted)))
    for j in range(1, steps_ahead, 1):    
        print(j)
        arma2 = statsmodels.api.tsa.ARMA(dataset, (1,0)).fit()
        forecasted.append(arma2.params[0] + arma2.params[1] * dataset[-1])
        dataset = np.hstack((dataset, np.array(forecasted[-1])))
    return dataset
###############################################################################
forecasted = Forecast(remn, 60)

rf = np.concatenate((remn, forecasted))

plt.figure()
#plt.plot(rf)
plt.plot(forecasted, color = 'black', lw = 2)
plt.plot(arma_pred, color = 'red')

plt.figure()
plt.plot(rem_per)
np.random.seed(123)
plt.figure()
plt.plot(np.random.normal(size = remn.shape[0]))
plt.plot(statsmodels.api.tsa.periodogram(np.random.normal(size = remn.shape[0])))

import scipy.integrate as integrate
###############################################################################
def FourierCoefficients(freq, f):
    xc = lambda x, freq, f: f * np.cos(freq * x)
    xs = lambda x, freq, f: f * np.sin(freq * x)
    return (1/np.pi) * integrate.quad(xc, 0, f.size, args = (f, freq)), (1/np.pi) * integrate.quad(xs, 0, f.size, args = (f, freq)) 
###############################################################################
def cn(y, n, time, period):
   c = y[time]*np.exp(-1j*2*time[n]*np.pi*time/period)
   return c.sum()/c.size
###############################################################################
def cn2(y, omega, period):
   c = y*np.exp(-1j*omega*np.pi*np.arange(y.size)/period)
   return c.sum()/np.sqrt(c.size)
###############################################################################
def f(x, y, time, period):
   f = np.array([2*cn(y, i, time , period)*np.exp(1j*2*i*np.pi*x/period) for i in range(time.size)]) #range(1,Nh+1)])
   return f.sum()
###############################################################################
def DFT(y, freq):
    N = y.size
    c = y * np.exp((-1j)*(2*np.pi)*freq*np.arange(y.size)/N)
    return c.sum()
###############################################################################   
def IDFT(Y, freq):
    N = Y.size
    y = Y * np.exp((2*np.pi*1j)*freq*np.arange(Y.size)/N)
    return y.sum()/N
###############################################################################
def hurst(ts):
	"""Returns the Hurst Exponent of the time series vector ts"""
	# Create the range of lag values
	lags = range(2, 100)

	# Calculate the array of the variances of the lagged differences
	tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

	# Use a linear fit to estimate the Hurst Exponent
	poly = np.polyfit(np.log(lags), np.log(tau), 1)

	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0
###############################################################################
print("Hurst(remainderized PUN): {}".format(hurst(remn))) ### mean reverting ###

surv = np.where(rem_per > 1)[0]
Ccos = []
Csin = []
coeffs = []
cn(remn.values, 1, surv[:2], 1)
cn2(remn.values, surv[0], 1)
for x in np.linspace(0, remn.values.ravel().size, remn.values.ravel().size):
    #xc, xs = FourierCoefficients(surv[i], remn.values.ravel())
    #Ccos.append(xc)
    #Csin.append(xs)
    fc = f(x,  remn.values.ravel(), surv,1) 
    coeffs.append(fc)
    
for i in range(1, surv.size, 1):
    print(cn2(remn.values, surv[i], 1))    
    coeffs.append(cn2(remn.values, surv[i], 1)) 
    
plt.figure()
plt.plot(np.array(coeffs))    

#### test DFT - IDFT ####    
y = np.array(remn).ravel()
res = [] 
for s in rem_per:
    res.append(DFT(y, s))
plt.figure()
plt.plot(np.array(res))
rec = []
for s in rem_per:
    rec.append(IDFT(np.array(res), s))
       
plt.figure()
plt.plot(np.array(remn.values.ravel()).ravel())
plt.plot(np.array(rec), color = 'red')    
    
rec_error = np.array(remn.values.ravel()).ravel() - np.array(rec).real
 
np.mean(rec_error)   
np.std(rec_error)

### forecasted value for 25/10/2016 with this un-tuned method:
P_new = mm[-1] + 2.590220 + rec[0].real
sigma_sum = np.std(pun.ix[pun.index.month == 10] - mm[-1]) + 8.104067 + np.std(rec_error)
lu =(P_new.real - sigma_sum, P_new.real + sigma_sum)

rngr = pd.date_range('2016-01-01', '2016-10-24', freq='D')
rem = pd.DataFrame(remn.values.ravel()).set_index(rngr)
drem = divide_in_days(rem)

plt.figure()
drem.plot()
drem.T.plot()

###############################################################################
def remainderizer2(df):
    ### comupe trend in months:
    diz = OrderedDict()
    dow = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom'] 
    mp = np.array(df.resample('M').mean())
    dt = []
    for i in range(df.shape[0]):
        mon = mp[(df.index.month[i] - 1)]
        print('month {} and correction {}'.format(df.index.month[i],mon))
        dt.append(df.ix[i] - mon)
    dt = pd.DataFrame(dt).set_index(df.index)
    ### remove global seasonality:
    dd = divide_in_days(dt)
    diz['mean'] = dd.mean()
    diz['std'] = dd.std()
    des= []
    for j in range(dt.shape[0]):
        die = datetime.date(int(str(dt.index[j])[:4]),int(str(dt.index[j])[5:7]),int(str(dt.index[j])[8:10]))
        giorno = die.weekday()
        x = (dt.ix[j] - dd[dow[giorno]].mean())/dd[dow[giorno]].std()
        des.append(x)
    rem = pd.DataFrame(des)
    seas = pd.DataFrame.from_dict(diz).set_index([dow])
    return mp, seas, rem
###############################################################################

mm, seas2, rem2 = remainderizer2(pun)    
    
plt.figure()
rem2.plot()    
    
plt.figure()    
plotting.lag_plot(rem2)

per2 = statsmodels.api.tsa.periodogram(np.array(rem2.values.ravel()).ravel())    
plt.figure()
plt.plot(per2)
hurst(np.array(rem2.values.ravel()).ravel())
hurst(pun)

###############################################################################
def Forecast_(pun, year, month, day):
    ts = np.array(pun)
    dow = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom'] 
    dt = datetime.datetime(year, month, day)
    mm, sea, remn = remainderizer(pun)
    arma = statsmodels.api.tsa.ARMA(remn.values.ravel(), (1,0)).fit()
    resid = remn.values.ravel() - arma.predict()
    pred = arma.predict(start = ts.size, end = ts.size)
    forecasted = mm[-1] + sea[str(month)+'_mean'].ix[dow[dt.weekday()]] + pred
    #### sampled sigma is a bit overestimated
    sigma_hat = np.std(pun.ix[pun.index.month == 10] - mm[-1]) + sea[str(month)+'_std'].ix[dow[dt.weekday()]] + np.std(resid)
    return (forecasted - 2*sigma_hat, forecasted - sigma_hat, forecasted, forecasted + sigma_hat, forecasted + 2*sigma_hat)
###############################################################################

Forecast_(pun, 2016, 10, 26)
Forecast_(pun, 2016, 10, 27)
 
import arch

arch_model = arch.arch_model(remn, mean = 'AR', vol='garch', p=1, q=1).fit()
arch_model.summary()

plt.figure()
arch_model.plot()
arch_model.params
arch_model.rsquared
np.mean(arch_model.resid)
arch_model.conditional_volatility
arch_model.hedgehog_plot(horizon = 40, step = 40)
arch_model.forecast()

fitted = remn + arch_model.resid

plt.figure()
plt.plot(fitted, color = 'black')
plt.plot(remn)

R2 = 1 - np.sum((remn - fitted)**2)/np.sum((remn)**2)

cv = arch_model.conditional_volatility
np.mean(cv)

ccv = np.cumsum(cv)/np.arange(1,cv.size+1,1)

plt.figure()
plt.plot(ccv)

cremn = np.cumsum(remn)/np.arange(1, remn.size + 1, 1)

plt.figure()
plt.plot(cremn)