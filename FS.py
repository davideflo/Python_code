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
    
###  EPEX FR from 2016-09-30 to 2016-10-17:
nd = [41.78, 38.19, 32.48, 36.04, 42.02,42.68 ,48.28,
      57.29208333	, 44.35375,	36.705,56.70208333, 71.21208333, 62.81041667, 64.25, 64.10, 44.28, 40.02, 56.41,66.94]
for n in nd:
    fr.append(n)
fsm['francia'] = fr
fsm['svizzera'] = sv
fsm['pun'] = pun[:287]

fsm = pd.DataFrame.from_dict(fsm)

fsm.plot() ### from this I'm very doubtful that flows actually correlate wth prices... 
           ### except in the last week where something anomalous is definitely happening

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

