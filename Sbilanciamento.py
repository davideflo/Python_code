# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:34:42 2016

@author: d_floriello

Analisi Sbilanciamento
"""

import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import statsmodels.api
import matplotlib.pyplot as plt
from pandas.tools import plotting
import scipy
import dateutil
from collections import OrderedDict
import datetime

###############################################################################
def DateParser(dt):
    dto = datetime.datetime(year = int(dt[6:10]), month = int(dt[3:5]), day = int(dt[:2]), hour = int(dt[11:13]))
    return dto
###############################################################################
def ConvertDates(df):
    dts = []
    for i in range(df.shape[0]):
        dts.append(DateParser(df.ix[i]))
    return dts
###############################################################################


#path2 = "H:/Energy Management/04. WHOLESALE/18. FATTURAZIONE WHOLESALE/2016/TERNA_2016/01_TERNA_2016_SETTLEMENT/TERNA_2016.09/FP/2016.09_Sbilanciamento_UC_2016761743A.csv"

#sbil = pd.read_csv(path2,sep = ';', skiprows = [0,1], error_bad_lines=False)

mon = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
years = [2015, 2016]

#path = "H:/Energy Management/04. WHOLESALE/18. FATTURAZIONE WHOLESALE/"
path = 'H:/Energy Management/Davide_per_sbilanciamento/'

sbil_tot = pd.DataFrame()
for y in years:    
    for m in mon:
        print(m)
        if y == 2016 and m in ['10', '11', '12']:
            break
        else:
            #pp = path+str(y)+'/TERNA_'+str(y)+'/01_TERNA_'+str(y)+'_SETTLEMENT/TERNA_'+str(y)+'.'+m+'/FA/'
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            nof = [onlyfiles[i] for i in range(len(onlyfiles)) if onlyfiles[i].startswith(str(y)+'.'+m+'_Sbilanciamento_UC_'+str(y))]
            #nof2 = [nof[i] for i in range(len(nof)) if nof[i].endswith('.csv')]        
            sbil = pd.read_csv(path+nof[0], sep = ';', skiprows = [0,1], error_bad_lines=False)
            sbil_tot = sbil_tot.append(sbil[['CODICE RUC', 'DATA RIFERIMENTO CORRISPETTIVO', 'PV [MWh]', 'SEGNO SBILANCIAMENTO AGGREGATO ZONALE']], ignore_index = True)                
        
sbil_tot.to_excel('aggregato_sbilanciamento.xlsx')        

ST = pd.read_excel('aggregato_sbilanciamento.xlsx')
#ST = ST.set_index(pd.date_range('2015-01-01', '2016-09-30', freq = 'H'))

cnlist = (ST[['CODICE RUC']].values == 'UC_DP1608_CNOR').ravel().tolist()
cnor = ST.ix[cnlist]
cnor = cnor.reset_index(drop = True)
cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[:1000].plot(ylim = (-2,2))

plt.plot(statsmodels.api.tsa.acf(np.array(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[0:2000])))
plotting.autocorrelation_plot(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[0:2000])

plt.plot(np.diff(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

###############################################################################
def get_cons_hours(ts):
    ch = []
    locc = 0
    for i in range(ts.size-1):
        if ts[i+1] == ts[i] and ts[i] < 0:
            locc -=1
        elif ts[i+1] == ts[i] and ts[i] > 0:
            locc +=1
        else:
            ch.append(locc)
            locc = 0
    return ch
###############################################################################
chcnor = get_cons_hours(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel())   

   
plt.figure()
plt.plot(np.array(chcnor))
plt.axhline(y = scipy.stats.mstats.mquantiles(chcnor, prob = 0.95))    
plt.axhline(y = scipy.stats.mstats.mquantiles(chcnor, prob = 0.025))    


np.mean(chcnor)
np.median(chcnor)
np.std(chcnor)
    
plt.figure()
plt.plot(statsmodels.api.tsa.acf(chcnor))
    
plt.figure()
plt.hist(np.array(chcnor))

scipy.stats.shapiro(np.array(chcnor))

np.where(np.logical_and(0 < np.array(chcnor), np.array(chcnor) <= 14))[0].size/len(chcnor)
np.where(np.logical_and(-17 < np.array(chcnor), np.array(chcnor) <= 0))[0].size/len(chcnor)

dt = dateutil.parser.parse(cnor[cnor.columns[1]].ix[0])

si = []
for i in range(cnor.shape[0]):
    si.append(dateutil.parser.parse(cnor[cnor.columns[1]].ix[i]))

cnor = cnor.set_index(pd.to_datetime(si))

for h in range(24):
    cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cnor.index.hour == h].hist()
    plt.title(h)


cnor.ix[cnor.index.year == 2015].plot(ylim = (-2,2))
cnor.ix[cnor.index.year == 2016].plot(ylim = (-2,2))

cnor.ix[cnor.index.year == 2015].hist()
cnor.ix[cnor.index.year == 2016].hist()


plotting.lag_plot(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']])
plt.figure()
plt.scatter(chcnor[:len(chcnor)-1], chcnor[1:])


diz5 = OrderedDict()
diz6 = OrderedDict()
cn5 = cnor.ix[cnor.index.year == 2015]
cn6 = cnor.ix[cnor.index.year == 2016]
for h in range(24):
    diz5[h] = cn5[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cn5.index.hour == h].values.ravel()
    diz6[h] = cn6[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cn6.index.hour == h].values.ravel()

CN5 = pd.DataFrame.from_dict(diz5, orient = 'index')
CN6 = pd.DataFrame.from_dict(diz6, orient = 'index')

CN5.mean().plot(ylim = (-2,2))

plt.figure()
CN6.mean().plot(color = 'red',ylim = (-2,2))

plt.figure()
CN5.mean(axis = 1).plot()
CN6.mean(axis = 1).plot(color = 'black')

CN6.mean().size

plt.figure()
plt.scatter(CN5.mean()[:274],CN6.mean())

np.corrcoef(CN5.mean()[:274],CN6.mean())


f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(CN5.mean()[:274], lw = 2)
axarr[1].plot(CN6.mean(), color = 'red', lw = 2)

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(CN5.std()[:274], lw = 2)
axarr[1].plot(CN6.std(), color = 'red', lw = 2)

CN6.std().max()
CN5.std().max()

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(np.diff(CN5.mean()[:274]), lw = 2)
axarr[1].plot(np.diff(CN6.mean()), color = 'red', lw = 2)

###############################################################################
def hurst(ts, n=100):
	"""Returns the Hurst Exponent of the time series vector ts"""
	# Create the range of lag values
	lags = range(2, n)

	# Calculate the array of the variances of the lagged differences
	tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

	# Use a linear fit to estimate the Hurst Exponent
	poly = np.polyfit(np.log(lags), np.log(tau), 1)

	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0
###############################################################################
hurst(CN5.mean()) 
hurst(CN6.mean()) 

###############################################################################
def conditionalDistribution(df, h1, h2):
    cdiz = OrderedDict()
    df = df.T
    dfp = df.ix[df[df.columns[h1]] > 0]
    dfn = df.ix[df[df.columns[h1]] < 0]
    cdiz[str(h1)+'pos'] = dfp[dfp.columns[h2]].values.ravel()
    cdiz[str(h1)+'neg'] = dfn[dfn.columns[h2]].values.ravel()
    return pd.DataFrame.from_dict(cdiz, orient = 'index')
###############################################################################
    
H2 = conditionalDistribution(CN5, 1, 2)    
    
plt.figure()
plt.hist(np.array(H2.ix['1pos'].dropna()))    
plt.hist(np.array(0.5 + H2.ix['1neg']), color = 'red')    

###############################################################################    
def Influence(df):
    inf = []
    df = df.T
    for h in range(24):
        condcorr = 0
        if h < 23:
            dfp = df.ix[df[df.columns[h]] > 0]
            dfn = df.ix[df[df.columns[h]] < 0]
            psize = dfp.shape[0]
            nsize = dfn.shape[0]
            condcorr += (np.sum(np.abs(dfp[dfp.columns[h]].values.ravel() - dfp[dfp.columns[(h+1)]].values.ravel()))/2)/psize
            condcorr += (np.sum(np.abs(dfn[dfn.columns[h]].values.ravel(),dfn[dfn.columns[(h+1)]].values.ravel()))/2)/nsize
            inf.append(condcorr)
        else:
            dfp = df.ix[df[df.columns[h]] > 0]
            dfn = df.ix[df[df.columns[h]] < 0]
            psize = dfp.shape[0]
            nsize = dfn.shape[0]
            condcorr += (np.sum(np.abs(dfp[dfp.columns[h]].values.ravel()-dfp[dfp.columns[0]].values.ravel()))/2)/psize
            condcorr += (np.sum(np.abs(dfn[dfn.columns[h]].values.ravel(),dfn[dfn.columns[0]].values.ravel()))/2)/nsize
            inf.append(condcorr)
    return np.array(inf)
###############################################################################
    
inf15 = Influence(CN5)
inf16 = Influence(CN6)

plt.figure()
plt.plot(inf15)    
plt.plot(inf16, color = 'black')

for h in range(23):
    H2 = conditionalDistribution(CN6, h, h+1)    
    plt.figure()
    plt.hist(np.array(H2.ix[str(h)+'pos'].dropna()))    
    plt.hist(np.array(0.5 + H2.ix[str(h)+'neg'].dropna()), color = 'red')    
    plt.title(str(h) + ' --> ' + str(h+1))

sm15 = CN5.mean().values.ravel()
sm16 = CN6.mean().values.ravel()

np.where(sm15 > 0)[0].size/sm15.size
np.where(sm16 > 0)[0].size/sm16.size


sd15 = CN5.std().values.ravel()
sd16 = CN6.std().values.ravel()

print(np.mean(sd15))
print(np.mean(sd16))
print(np.std(sd15))
print(np.std(sd16))


P5 = np.poly1d(np.polyfit(np.linspace(1,sm15.size,sm15.size), sm15, 3))
P6 = np.poly1d(np.polyfit(np.linspace(1,sm16.size,sm16.size), sm16, 3))

plt.figure()
plt.plot(sm15)
plt.plot(np.linspace(1,sm15.size,1000), P5(np.linspace(1,sm15.size,1000)))

plt.figure()
plt.plot(sm16, color = 'red')
plt.plot(np.linspace(1,sm16.size,1000), P6(np.linspace(1,sm16.size,1000)), color = 'black')

#################################################################################################################
def getStatistics(zona):
    cnlist = (ST[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()
    cnor = ST.ix[cnlist]
    cnor = cnor.reset_index(drop = True)


    #plt.plot(np.diff(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

    chcnor = get_cons_hours(cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel())   

   
    plt.figure()
    plt.plot(np.array(chcnor))
    plt.axhline(y = scipy.stats.mstats.mquantiles(chcnor, prob = 0.95))    
    plt.axhline(y = scipy.stats.mstats.mquantiles(chcnor, prob = 0.025))    


    print('mean consecutive days with same sign: {}'.format(np.mean(chcnor)))
    print('median consecutive days with same sign: {}'.format(np.median(chcnor)))
    print('std consecutive days with same sign: {}'.format(np.std(chcnor)))    

    
    plt.figure()
    plt.hist(np.array(chcnor))

    print('shapiro t4est: {}'.format(scipy.stats.shapiro(np.array(chcnor))))

    print('#########################################################################################')
    print(np.where(np.logical_and(0 < np.array(chcnor), np.array(chcnor) <= scipy.stats.mstats.mquantiles(chcnor, prob = 0.95)))[0].size/len(chcnor))
    print(np.where(np.logical_and(scipy.stats.mstats.mquantiles(chcnor, prob = 0.025) < np.array(chcnor), np.array(chcnor) <= 0))[0].size/len(chcnor))
    print('#########################################################################################')


    si = []
    for i in range(cnor.shape[0]):
        si.append(dateutil.parser.parse(cnor[cnor.columns[1]].ix[i]))
    
    cnor = cnor.set_index(pd.to_datetime(si))

#    for h in range(24):
#        cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cnor.index.hour == h].hist()
#        plt.title(h)


#    cnor.ix[cnor.index.year == 2015].plot(ylim = (-2,2))
#    cnor.ix[cnor.index.year == 2016].plot(ylim = (-2,2))
#    
    #cnor.ix[cnor.index.year == 2015].hist()
    #cnor.ix[cnor.index.year == 2016].hist()


    diz5 = OrderedDict()
    diz6 = OrderedDict()
    cn5 = cnor.ix[cnor.index.year == 2015]
    cn6 = cnor.ix[cnor.index.year == 2016]
    for h in range(24):
        diz5[h] = cn5[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cn5.index.hour == h].values.ravel()
        diz6[h] = cn6[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cn6.index.hour == h].values.ravel()
    
    CN5 = pd.DataFrame.from_dict(diz5, orient = 'index')
    CN6 = pd.DataFrame.from_dict(diz6, orient = 'index')

    plt.figure()
    CN5.mean().plot(ylim = (-2,2))

    plt.figure()
    CN6.mean().plot(color = 'red',ylim = (-2,2))

    plt.figure()
    CN5.mean(axis = 1).plot()
    CN6.mean(axis = 1).plot(color = 'black')
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(CN5.mean()[:274], lw = 2)
    axarr[1].plot(CN6.mean(), color = 'red', lw = 2)
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(CN5.std()[:274], lw = 2)
    axarr[1].plot(CN6.std(), color = 'red', lw = 2)

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(np.diff(CN5.mean()[:274]), lw = 2)
    axarr[1].plot(np.diff(CN6.mean()), color = 'red', lw = 2)
    
    H2 = conditionalDistribution(CN5, 1, 2)    
    
    plt.figure()
    plt.hist(np.array(H2.ix['1pos'].dropna()))    
    plt.hist(np.array(0.5 + H2.ix['1neg']), color = 'red')    
    
#    for h in range(23):
#        H2 = conditionalDistribution(CN6, h, h+1)    
#        plt.figure()
#        plt.hist(np.array(H2.ix[str(h)+'pos'].dropna()))    
#        plt.hist(np.array(0.5 + H2.ix[str(h)+'neg'].dropna()), color = 'red')    
#        plt.title(str(h) + ' --> ' + str(h+1))
#    
    sm15 = CN5.mean().values.ravel()
    sm16 = CN6.mean().values.ravel()
    
    np.where(sm15 > 0)[0].size/sm15.size
    np.where(sm16 > 0)[0].size/sm16.size
    
    
    sd15 = CN5.std().values.ravel()
    sd16 = CN6.std().values.ravel()
    
    print(np.mean(sd15))
    print(np.mean(sd16))
    print(np.std(sd15))
    print(np.std(sd16))


    P5 = np.poly1d(np.polyfit(np.linspace(1,sm15.size,sm15.size), sm15, 3))
    P6 = np.poly1d(np.polyfit(np.linspace(1,sm16.size,sm16.size), sm16, 3))
    
    plt.figure()
    plt.plot(sm15)
    plt.plot(np.linspace(1,sm15.size,1000), P5(np.linspace(1,sm15.size,1000)))
    
    plt.figure()
    plt.plot(sm16, color = 'red')
    plt.plot(np.linspace(1,sm16.size,1000), P6(np.linspace(1,sm16.size,1000)), color = 'black')
    return 0
################################################################################################################    

getStatistics('NORD')
getStatistics('CSUD')
getStatistics('SUD')
getStatistics('SICI')
getStatistics('SARD')

######### independence of non overlapping days?
###############################################################################
def TestIndipendence(st, zona):
    
    cnlist = (st[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()
    cnor = st.ix[cnlist]
    cnor = cnor.reset_index(drop = True)     
    d = np.random.randint(cnor.shape[0], size = 50)
    rem = set(range(cnor.shape[0])).difference(d)
    sam1 = cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[d]
    sam2 = cnor[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[np.random.choice(list(rem), size = 50)]
    plt.figure()
    plt.hist(sam1.values.ravel())
    plt.title('sample 1')
    plt.figure()
    plt.hist(sam2.values.ravel())
    plt.title('sample 2')

    si = []
    for i in range(cnor.shape[0]):
        si.append(dateutil.parser.parse(cnor[cnor.columns[1]].ix[i]))
    
    CN = cnor.set_index(pd.to_datetime(si))
    
    sm = CN[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].resample('D').mean()    
    
    d = np.random.randint(sm.shape[0], size = 50)
    rem = set(range(sm.shape[0])).difference(d)
    sam1 = sm.ix[d].dropna()
    sam2 = sm.ix[np.random.choice(list(rem), size = 50)].dropna()
    plt.figure()
    plt.hist(sam1.values.ravel())
    plt.title('sample 1')
    plt.figure()
    plt.hist(sam2.values.ravel())
    plt.title('sample 2')

    plt.figure()
    plotting.lag_plot(sm)
    plt.title('lag = 1')
    plt.figure()
    plotting.lag_plot(sm, lag = 2)
    plt.title('lag = 2')
    plt.figure()
    plotting.lag_plot(sm, lag = 5)
    plt.title('lag = 5')
    plt.figure()
    plotting.lag_plot(sm, lag = 10)
    plt.title('lag = 10')
    plt.figure()
    plotting.lag_plot(sm, lag = 30)
    plt.title('lag = 30')

    
    return 0
###############################################################################

sii = []
for i in range(ST.shape[0]):
    sii.append(dateutil.parser.parse(ST[ST.columns[1]].ix[i]))
    
ST_2 = ST.set_index(pd.to_datetime(ConvertDates(ST[ST.columns[1]])))

ST16 = ST_2.ix[ST_2.index.year == 2016]
ST15 = ST_2.ix[ST_2.index.year == 2015]

#ConvertDates(ST[ST.columns[1]])



TestIndipendence(ST16, 'NORD')
TestIndipendence(ST15, 'NORD')
TestIndipendence(ST16, 'CNOR')
TestIndipendence(ST15, 'CNOR')
TestIndipendence(ST16, 'CSUD')
TestIndipendence(ST15, 'CSUD')
TestIndipendence(ST16, 'SUD')
TestIndipendence(ST15, 'SUD')
TestIndipendence(ST16, 'SICI')
TestIndipendence(ST15, 'SICI')
TestIndipendence(ST16, 'SARD')
TestIndipendence(ST15, 'SARD')

#### qualitativamente: NORD non sembra troppo indipendente: chiaro cluster attorno a (1,1)
#### le altre zone sembrano un po' piu indipendenti (gli sbilanciamenti medi giornalieri)
#### reminder: 24*media_giornaliera_segni_sbilanciamento_orari ~ Binomiale

##### distribution 0-variance days #####

###############################################################################
def CountZeroVariance(st, zona):    
    
    cnlist = (st[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()
    cnor = st.ix[cnlist]
    cnor = cnor.reset_index(drop = True)  

    CN = cnor.set_index(pd.to_datetime(ConvertDates(cnor[cnor.columns[1]])))
    
    CV = CN.resample('D').std()
    
    counter = []    
    
    for m in range(1,13,1):
        print(m)
        print('there are {} days'.format(np.sum(st.index.month == m)))
        lm = CV.ix[CV.index.month == m].values.ravel()
        counter.append(np.where(lm == 0)[0].size)
    
    return counter
###############################################################################
    
CountZeroVariance(ST16, 'NORD')
CountZeroVariance(ST15, 'NORD')
CountZeroVariance(ST16, 'CNOR')
CountZeroVariance(ST15, 'CNOR')
CountZeroVariance(ST16, 'CSUD')
CountZeroVariance(ST15, 'CSUD')
CountZeroVariance(ST16, 'SUD')
CountZeroVariance(ST15, 'SUD')
CountZeroVariance(ST16, 'SICI')
CountZeroVariance(ST15, 'SICI')
CountZeroVariance(ST16, 'SARD')
CountZeroVariance(ST15, 'SARD')

