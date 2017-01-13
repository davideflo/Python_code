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
import sklearn

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
###############################################################################
def DateParser2(dt):
    dto = datetime.datetime(year = int(dt[6:10]), month = int(dt[3:5]), day = int(dt[:2]))
    return dto
###############################################################################
def ConvertDates2(df):
    dts = []
    for i in range(df.shape[0]):
        dts.append(DateParser2(df.ix[i]))
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
        if y == 2016 and m in ['12']:
            break
        else:
            #pp = path+str(y)+'/TERNA_'+str(y)+'/01_TERNA_'+str(y)+'_SETTLEMENT/TERNA_'+str(y)+'.'+m+'/FA/'
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            nof = [onlyfiles[i] for i in range(len(onlyfiles)) if onlyfiles[i].startswith(str(y)+'.'+str(m)+'_Sbilanciamento_UC_'+str(y))]
            #nof2 = [nof[i] for i in range(len(nof)) if nof[i].endswith('.csv')]        
            sbil = pd.read_csv(path+nof[0], sep = ';', skiprows = [0,1], error_bad_lines=False)
            sbil_tot = sbil_tot.append(sbil[['CODICE RUC', 'DATA RIFERIMENTO CORRISPETTIVO', 'MO [MWh]', 'PV [MWh]', 'SBILANCIAMENTO FISICO [MWh]','SEGNO SBILANCIAMENTO AGGREGATO ZONALE']], ignore_index = True)                
        
sbil_tot.to_excel('aggregato_sbilanciamento.xlsx')        

ST = pd.read_excel('aggregato_sbilanciamento.xlsx')
#ST = ST.set_index(pd.date_range('2015-01-01', '2016-09-30', freq = 'H'))

cnlist = (ST[['CODICE RUC']].values == 'UC_DP1608_NORD').ravel().tolist()
cnor = ST.ix[cnlist]
cnor = cnor.reset_index(drop = True)


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


    cnor = cnor.set_index(pd.to_datetime(ConvertDates(cnor[cnor.columns[1]])))

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

ST_2 = ST.set_index(pd.to_datetime(ConvertDates(ST[ST.columns[1]])))

ST16 = ST_2.ix[ST_2.index.year == 2016]
ST15 = ST_2.ix[ST_2.index.year == 2015]

#ConvertDates(ST[ST.columns[1]])

###############################################################################
#### dataset to be used in R ####
def ToDataFrame(st, zona):
    stz = st.ix[(st[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()]
    diz = OrderedDict()
    rng = pd.date_range(start = '2015-01-01', end = '2016-11-01', freq = 'D')
    for r in rng:
        hl = []
        hl2 = 0
        sub1 = stz.ix[stz.index.year == r.year]
        sub2 = sub1.ix[sub1.index.month == r.month]
        sub3 = sub2.ix[sub2.index.day == r.day]
        for h in range(24):
            if sub3['FABBISOGNO REALE'].ix[sub3.index.hour == h].values.ravel().size == 0:
                hl.append(0)
            elif sub3['FABBISOGNO REALE'].ix[sub3.index.hour == h].values.ravel().size == 2:
                hl.append([np.sum(sub3['FABBISOGNO REALE'].ix[sub3.index.hour == h].values.ravel())])
            else:
                hl.append(sub3['FABBISOGNO REALE'].ix[sub3.index.hour == h].values.ravel())
        if isinstance(hl[0], list):
            hl2 = [float(item) for sublist in hl for item in sublist]
        else:
            hl2 = hl
        diz[r] = np.array(hl2).ravel()
    df = pd.DataFrame.from_dict(diz, orient = 'index')
    df.columns = ['1','2','3','4','5','6','7','8','9','10','11','12',
                  '13','14','15','16','17','18','19','20','21','22','23','24']
    return df
###############################################################################
def ToDataFrameMO(st, zona):
    stz = st.ix[(st[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()]
    diz = OrderedDict()
    rng = pd.date_range(start = '2015-01-01', end = '2016-11-01', freq = 'D')
    for r in rng:
        hl = []
        hl2 = 0
        sub1 = stz.ix[stz.index.year == r.year]
        sub2 = sub1.ix[sub1.index.month == r.month]
        sub3 = sub2.ix[sub2.index.day == r.day]
        for h in range(24):
            if sub3['MO [MWh]'].ix[sub3.index.hour == h].values.ravel().size == 0:
                hl.append(0)
            elif sub3['MO [MWh]'].ix[sub3.index.hour == h].values.ravel().size == 2:
                hl.append([np.sum(float(sub3['MO [MWh]'].ix[sub3.index.hour == h].values.ravel()[0].replace(',','.')) )])
            else:
                hl.append(float(sub3['MO [MWh]'].ix[sub3.index.hour == h].values.ravel()[0].replace(',','.')))
        if isinstance(hl[0], list):
            hl2 = [float(item) for sublist in hl for item in sublist]
        else:
            hl2 = hl
        diz[r] = hl2
    df = pd.DataFrame.from_dict(diz, orient = 'index')
    df.columns = ['1','2','3','4','5','6','7','8','9','10','11','12',
                  '13','14','15','16','17','18','19','20','21','22','23','24']
    return df
###############################################################################
ex = ToDataFrame(ST_2, 'CNOR')
ex2 = ToDataFrame(ST_2, 'NORD')
ex3 = ToDataFrame(ST_2, 'CSUD')
ex4 = ToDataFrame(ST_2, 'SUD')
ex5 = ToDataFrame(ST_2, 'SICI')
ex6 = ToDataFrame(ST_2, 'SARD')

exmo = ToDataFrameMO(ST_2, 'CNOR')
ex2 = ToDataFrameMO(ST_2, 'NORD')

exmo.to_excel('MOcnord2.xlsx')

ex.to_csv('cnord.csv', sep = ';')
ex.to_excel('cnord.xlsx')
ex2.to_excel('nord.xlsx')


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
def DistBetweenZeroVarDays(vec):
    dist = []
    x = 0
    for i in range(vec.size):
        if vec[i] == 0:
            dist.append(i - x)
            x = i
    return dist
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

nord16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_NORD').ravel().tolist()]
nord15 = ST15.ix[(ST15[['CODICE RUC']].values == 'UC_DP1608_NORD').ravel().tolist()]

f, axarr = plt.subplots(2)
axarr[0].plot(nord16.resample('D').mean(), lw = 2)
axarr[1].plot(nord15.resample('D').mean(), color = 'red', lw = 2)

f, axarr = plt.subplots(2)
axarr[0].plot(nord16.resample('D').std(), lw = 2)
axarr[1].plot(nord15.resample('D').std(), color = 'red', lw = 2)

var_nord16 = np.array(nord16.resample('D').std()).ravel()
var_nord15 = np.array(nord15.resample('D').std()).ravel()

plt.figure()
plt.hist(np.array(var_nord16), bins = 20)
plt.figure()
plotting.autocorrelation_plot(pd.Series(var_nord16))
plt.figure()
plotting.autocorrelation_plot(pd.Series(np.random.sample(size = len(var_nord16))))

d16nord = DistBetweenZeroVarDays(var_nord16)
d15nord = DistBetweenZeroVarDays(var_nord15)

plt.figure()
plt.hist(np.array(d16nord))
plt.figure()
plt.hist(np.array(d15nord))

np.mean(d16nord)
np.mean(d15nord)
np.std(d16nord)
np.std(d15nord)
np.median(d16nord)
np.median(d15nord)

import Fourier

plt.figure()
plt.plot(np.array(var_nord16), lw = 2)
plt.plot(Fourier.fourierExtrapolation(var_nord16, 0), lw = 2, color = 'black')

data = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2016/06. MI/DB_Borse_Elettriche_PER MI.xlsx", sheetname = 'DB_Dati')
data = data.set_index(data['Date'])
data = data.ix[data.index.month <= 9]
pnord = data['MGP NORD [€/MWh]']
pnord = pnord.ix[:pnord.shape[0]-1]
cnord = data['MGP CNOR [€/MWh]']
cnord = cnord.ix[:pnord.shape[0]-1]
csud = data['MGP CSUD [€/MWh]']
csud = csud.ix[:csud.shape[0]-1]
sud = data['MGP SUD [€/MWh]']
sud = sud.ix[:sud.shape[0]-1]
sici = data['MGP SICI [€/MWh]']
sici = sici.ix[:sici.shape[0]-1]
sard = data['MGP SARD [€/MWh]']
csud = sard.ix[:sard.shape[0]-1]

pun = data['PUN [€/MWH]']
#pun = data['PUN [€/MWH]'].dropna().resample('D').mean()

pd.Series(pnord.values.ravel()).corr(pd.Series(nord16[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

pind = pnord.index.tolist()
cind = nord16.index.tolist()

plt.figure()
plt.scatter(np.array(pnord.resample('D').mean()), np.array(nord16.resample('D').mean()))
plt.figure()
plt.scatter(np.array(pnord.resample('D').mean()), np.array(nord16.resample('D').std()), color = 'red')
plt.figure()
plt.scatter(np.array(pnord.resample('D').std()), np.array(nord16.resample('D').std()), color = 'green')
    


cnord16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_CNOR').ravel().tolist()]

plt.figure()
plt.scatter(np.array(cnord.resample('D').mean()), np.array(cnord16.resample('D').mean()))

pd.Series(cnord.values.ravel()).corr(pd.Series(cnord16[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

csud16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_CSUD').ravel().tolist()]

plt.figure()
plt.scatter(np.array(csud.resample('D').mean()), np.array(csud16.resample('D').mean()))

pd.Series(csud.values.ravel()).corr(pd.Series(csud16[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

sud16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_SUD').ravel().tolist()]

plt.figure()
plt.scatter(np.array(sud.resample('D').mean()), np.array(sud16.resample('D').mean()))

pd.Series(csud.values.ravel()).corr(pd.Series(csud16[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

sici16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_SICI').ravel().tolist()]

plt.figure()
plt.scatter(np.array(sici.resample('D').mean()), np.array(sici16.resample('D').mean()))

pd.Series(sici.values.ravel()).corr(pd.Series(sici16[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

sard16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_SARD').ravel().tolist()]

plt.figure()
plt.scatter(np.array(sard.resample('D').mean()), np.array(sard16.resample('D').mean()))

pd.Series(sard.values.ravel()).corr(pd.Series(sard16[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].values.ravel()))

###############################################################################
def RBinomialTree(st, zona, B):
    dix = OrderedDict()
    #######################
    diz = OrderedDict()
    stz = st.ix[(st[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()]
    cn = stz.ix[stz.index.year == 2016]
    for h in range(24):
        diz[h] = cn[['SEGNO SBILANCIAMENTO AGGREGATO ZONALE']].ix[cn.index.hour == h].values.ravel()
    
    CN = pd.DataFrame.from_dict(diz, orient = 'index')
    #######################
    pstart = np.where(CN.ix[0] == 1)[0].size/CN.shape[1]
    
    for b in range(B):
        r = scipy.stats.bernoulli.rvs(pstart, size=1)
        tree = [r[0]]
        for h in range(1, 23, 1):
            H = conditionalDistribution(CN, h, h+1)
            ph = np.where(H.ix[0] == 1)[0].size/H.shape[1]
            tree.append(scipy.stats.bernoulli.rvs(ph, size=1)[0])
        dix[b] = tree
    
    return pd.DataFrame.from_dict(dix, orient = 'index')
###############################################################################
import time
    
start = time.time()
t1 = RBinomialTree(ST16, 'NORD', 1000)    
(time.time() - start)/60    
    
###############################################################################
def converter(ser):
    res = []
    for i in range(ser.size):
        x = float(ser.ix[i].replace(',', '.'))
        res.append(x)
    return np.array(res).ravel()
###############################################################################
def PlotImbalance(st15, st16, zona):
    df16 = st16.ix[(st16[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()]
    df15 = st15.ix[(st15[['CODICE RUC']].values == 'UC_DP1608_'+zona).ravel().tolist()]
    plt.figure()
    (-df16[['PV [MWh]']]).plot()
    plt.figure()
    df16[['SBILANCIAMENTO FISICO [MWh]']].plot()
    plt.figure()
    (-df15[['PV [MWh]']]).plot(color = 'black')
    plt.figure()
    df15[['SBILANCIAMENTO FISICO [MWh]']].plot(color = 'black')
    
    psdf15 = df15[['SBILANCIAMENTO FISICO [MWh]']].values.ravel()/np.abs(df15[['PV [MWh]']].values.ravel())
    plt.figure()
    plt.plot(psdf15,color = 'green')
    plt.axhline(y=0, color = 'black')
    plt.title(zona+' Imbalance Percentage 2015')
    
    psdf16 = df16[['SBILANCIAMENTO FISICO [MWh]']].values.ravel()/np.abs(df16[['PV [MWh]']].values.ravel())
    plt.figure()
    plt.plot(psdf16,color = 'indigo')
    plt.axhline(y=0, color = 'black')
    plt.title(zona+' Imbalance Percentage 2016')        
    return 0
###############################################################################    
    
plt.figure()
plt.plot(converter(nord16[nord16.columns[2]]))
plt.figure()
plt.plot(converter(cnord16[cnord16.columns[2]]))
plt.figure()
plt.plot(converter(csud16[csud16.columns[2]]))
plt.figure()
plt.plot(converter(sud16[sud16.columns[2]]))
plt.figure()
plt.plot(converter(sici16[sici16.columns[2]]))
plt.figure()
plt.plot(converter(sard16[sard16.columns[2]]))

plt.figure()
(-nord16[['PV [MWh]']]).plot()
plt.figure()
nord16[['SBILANCIAMENTO FISICO [MWh]']].plot()
plt.figure()
(-nord15[['PV [MWh]']]).plot(color = 'black')
plt.figure()
nord15[['SBILANCIAMENTO FISICO [MWh]']].plot(color = 'black')

psnord15 = nord15[['SBILANCIAMENTO FISICO [MWh]']].values.ravel()/np.abs(nord15[['PV [MWh]']].values.ravel())
plt.figure()
plt.plot(psnord15,color = 'green')
plt.axhline(y=0, color = 'black')

psnord16 = nord16[['SBILANCIAMENTO FISICO [MWh]']].values.ravel()/np.abs(nord16[['PV [MWh]']].values.ravel())
plt.figure()
plt.plot(psnord16,color = 'indigo')
plt.axhline(y=0, color = 'black')


psnord15ts = pd.Series(psnord15, index = nord15.index)
dec = statsmodels.api.tsa.seasonal_decompose(psnord15ts, freq = 24)
plt.figure()
dec.plot()
plt.figure()
plt.plot(dec.seasonal[0:23])

plotting.autocorrelation_plot(psnord15ts)

scipy.stats.kurtosis(psnord15)
scipy.stats.skew(psnord15)
scipy.stats.kurtosis(psnord16)
scipy.stats.skew(psnord16)

###############################################################################
def BestApproximatingPolynomial(vec):
    best_d = 0
    P1 = np.mean(vec)
    qerr1 = np.mean((vec - P1)**2)
    best_error = qerr1
    print('0-degree error: {}'.format(qerr1))
    for d in range(1, 11, 1):
        BP = np.poly1d(np.polyfit(np.linspace(1,vec.size,vec.size), vec, d))
        qerr = np.mean((vec - BP(np.linspace(1,vec.size,vec.size)))**2)
        print(str(d)+'-degree error: {}'.format(qerr))
        if qerr < best_error:
            best_d = d
            best_error = qerr
    return best_d
###############################################################################
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
###############################################################################
def deseasonalise(hts): ## TS with temporal timestamp
    dix = OrderedDict()
    for h in range(24):
        dix[h] = hts.ix[hts.index.hour == h].values.ravel()
    return pd.DataFrame.from_dict(dix, orient = 'columns')
###############################################################################
def deseasonalised(hts, seas):
    res = []
    for i in range(0, hts.shape[0]-24, 24):
        day = hts[i:i+24].values.ravel()
        res.append(day - seas)
    day = hts[hts.shape[0] - 24:].values.ravel()
    res.append(day - seas)
    return np.array(res).ravel()
###############################################################################

psnord15ts = pd.DataFrame(psnord15ts)
trend = psnord15ts.rolling(24).mean()

BestApproximatingPolynomial(psnord15)

Ptrend = np.poly1d(np.polyfit(np.linspace(1,psnord15.size,psnord15.size), psnord15, 6))

plt.figure()
plt.plot(trend.values.ravel())
plt.plot(np.linspace(1,psnord15.size,1000), Ptrend(np.linspace(1,psnord15.size,1000)),lw=2, color = 'black')

trendN15 = Ptrend(np.linspace(1,psnord15.size,psnord15.size))

det_psnord15 = psnord15 + trendN15

plt.figure()
plt.plot(psnord15)
plt.plot(exponential_smoothing(psnord15,0.9), color = 'black')

Ds = deseasonalise(psnord15ts)
Ds.plot()
plt.figure()
Ds.mean().plot()

plt.figure()
plt.plot(deseasonalised(psnord15,Ds.mean()))

np.mean(deseasonalised(psnord15,Ds.mean()))
np.mean(psnord15)

############ Kalman - filter toy usage ########################################
n_iter = psnord15.size
sz = (n_iter,) # size of array
x = 0.0 # truth value
z = psnord15 # observations 

Q = 1e-5 # process variance

# allocate space for arrays
xhat=np.zeros(n_iter)      # a posteri estimate of x
P=np.zeros(n_iter)         # a posteri error estimate
xhatminus=np.zeros(n_iter) # a priori estimate of x
Pminus=np.zeros(n_iter)    # a priori error estimate
K=np.zeros(n_iter)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(z,'k-o',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Percentage Imbalance')

plt.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()

plt.figure()
plt.plot(dec.trend.values.ravel(), color = 'navy')
plt.plot(xhat, color = 'coral')
plt.title('difference of estimations')
###############################################################################

des = deseasonalised(psnord15ts, dec.seasonal[0:24])

residuals = des - dec.trend.values.ravel() 

plt.figure()
plt.plot(residuals)

np.nanmean(residuals)
np.nanstd(residuals)
np.nanmax(residuals)/np.nanstd(residuals)
np.nanmin(residuals)/np.nanstd(residuals)
scipy.stats.skew(residuals[np.logical_not(np.isnan(residuals))])
scipy.stats.kurtosis(residuals[np.logical_not(np.isnan(residuals))])
scipy.stats.shapiro(residuals[np.logical_not(np.isnan(residuals))])


plt.figure()
plt.hist(residuals[np.logical_not(np.isnan(residuals))])

meteo = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Milano 2016.xlsx')
meteo = meteo.set_index(pd.to_datetime(ConvertDates2(meteo['DATA'])))

nlug16 = nord16.ix[nord16.index.month <= 7]

snlug16 = nlug16['SBILANCIAMENTO FISICO [MWh]'].resample('D').sum()
snlug16.plot()
plt.figure()
meteo['Tmedia'].plot()

mnlug16 = nlug16['SBILANCIAMENTO FISICO [MWh]'].resample('D').mean()
plt.figure()
mnlug16.plot()

plt.figure()
plt.scatter(meteo['Tmedia'].values.ravel(), snlug16.values.ravel())
plt.scatter(meteo['Tmedia'].values.ravel(), mnlug16.values.ravel(), color = 'red')

Mnlug16 = nlug16['SBILANCIAMENTO FISICO [MWh]'].resample('D').max()
plt.figure()
Mnlug16.plot()

psnordlug16 = nord16[['SBILANCIAMENTO FISICO [MWh]']].ix[nord16.index.month <= 7].resample('D').sum().values.ravel()/np.abs(nord16[['PV [MWh]']].ix[nord16.index.month <= 7].resample('D').sum().values.ravel())
psnordlug16 = pd.Series(nord16[['SBILANCIAMENTO FISICO [MWh]']].ix[nord16.index.month <= 7].values.ravel()/np.abs(nord16[['PV [MWh]']].ix[nord16.index.month <= 7].values.ravel()), index = nord16.index[nord16.index.month <= 7])
psnordlug16 = psnordlug16.resample('D').sum().values.ravel()

plt.figure()
plt.scatter(meteo['Tmedia'].values.ravel(), psnordlug16, color = 'green')
plt.axhline(y = scipy.stats.mstats.mquantiles(psnordlug16, prob = 0.95), color = 'turquoise')
plt.axhline(y = scipy.stats.mstats.mquantiles(psnordlug16, prob = 0.025), color = 'turquoise')
plt.axvline(x = scipy.stats.mstats.mquantiles(meteo['Tmedia'].values.ravel(), prob = 0.95), color = 'yellow')
plt.axvline(x = scipy.stats.mstats.mquantiles(meteo['Tmedia'].values.ravel(), prob = 0.025), color = 'yellow')

P2 = np.poly1d(np.polyfit(meteo['Tmedia'].values.ravel(), psnordlug16, 2))
#X = np.array([meteo['Tmedia'].values.ravel(), meteo['Tmedia'].values.ravel()**2])
#ols_model = statsmodels.regression.linear_model.OLS(psnordlug16, X.T)
#res = ols_model.fit()
#res.params

plt.figure()
plt.scatter(meteo['Tmedia'].values.ravel(), psnordlug16, color = 'green')
plt.plot(np.linspace(-5,35,1000), P2(np.linspace(-5,35,1000)))

## R2 with outliers
R2 = 1 - np.sum((psnordlug16 - P2(meteo['Tmedia'].values.ravel()))**2)/np.sum((psnordlug16 - np.mean(psnordlug16))**2)

np.mean(psnordlug16 - P2(meteo['Tmedia'].values.ravel()))
np.std(psnordlug16 - P2(meteo['Tmedia'].values.ravel()))

## R2 without outliers

less = np.where(psnordlug16 < scipy.stats.mstats.mquantiles(psnordlug16, prob = 0.025))[0]

nooutlug16 = psnordlug16[list(set(list(range(psnordlug16.size))).difference(less))]
nooutmeteo = meteo['Tmedia'].values.ravel()[list(set(list(range(meteo['Tmedia'].values.ravel().size))).difference(less))]

P2out = np.poly1d(np.polyfit(nooutmeteo, nooutlug16, 2))
R2out = 1 - np.sum((nooutlug16 - P2(nooutmeteo))**2)/np.sum((nooutlug16 - np.mean(nooutlug16))**2)

plt.figure()
plt.scatter(nooutmeteo, nooutlug16, color = 'gold')
plt.plot(np.linspace(-5,35,1000), P2(np.linspace(-5,35,1000)))
plt.figure()
plt.plot(psnordlug16)

################### correlation meteo 2015

met15 = pd.read_table('C:/Users/d_floriello/Documents/PUN/storico_milano_aggiornato.txt')
met15 = met15.set_index(pd.to_datetime(ConvertDates2(met15['Data'])))

met15 = met15.ix[met15.index.year == 2015]

plt.figure()
plt.scatter(met15['Tmedia'].values.ravel(), psnord15ts.resample('D').sum().values.ravel())

plt.figure()
plt.plot(met15['Tmedia'].values.ravel())
plt.figure()
plt.plot(psnord15ts.resample('D').sum().values.ravel())

plt.figure()
plt.plot(met15['Tmedia'].values.ravel())
plt.plot(psnord15ts.resample('D').sum().values.ravel())


n_iter = psnord15ts.resample('D').sum().values.ravel().size
sz = (n_iter,) # size of array
x = 0.0 # truth value
z = psnord15ts.resample('D').sum().values.ravel() # observations 

Q = 1e-5 # process variance

# allocate space for arrays
xhat=np.zeros(n_iter)      # a posteri estimate of x
P=np.zeros(n_iter)         # a posteri error estimate
xhatminus=np.zeros(n_iter) # a priori estimate of x
Pminus=np.zeros(n_iter)    # a priori error estimate
K=np.zeros(n_iter)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(z,'k-o',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Percentage Imbalance')


###############################################################################

plt.figure()
plt.scatter(met15['Tmedia'].values.ravel(), xhat)


n_iter = met15['Tmedia'].values.ravel().size
sz = (n_iter,) # size of array
x = 0.0 # truth value
z = met15['Tmedia'].values.ravel() # observations 

Q = 1e-5 # process variance

# allocate space for arrays
yhat=np.zeros(n_iter)      # a posteri estimate of x
P=np.zeros(n_iter)         # a posteri error estimate
yhatminus=np.zeros(n_iter) # a priori estimate of x
Pminus=np.zeros(n_iter)    # a priori error estimate
K=np.zeros(n_iter)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
yhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    yhatminus[k] = yhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    yhat[k] = yhatminus[k]+K[k]*(z[k]-yhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(z,'k-o',label='noisy measurements')
plt.plot(yhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Percentage Imbalance')


###############################################################################

plt.figure()
plt.scatter(yhat, xhat, color = 'crimson')

##### correlation with meteo 2015 CSUD

roma15 = pd.read_table('C:/Users/d_floriello/Documents/PUN/storico_roma.txt')
roma15 = roma15.set_index(pd.to_datetime(ConvertDates2(roma15['Data'])))

roma15 = roma15.ix[roma15.index.year == 2015]

csud16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_CSUD').ravel().tolist()]
csud15 = ST15.ix[(ST15[['CODICE RUC']].values == 'UC_DP1608_CSUD').ravel().tolist()]
pcsud15 = csud15[['SBILANCIAMENTO FISICO [MWh]']].values.ravel()/np.abs(csud15[['PV [MWh]']].values.ravel())
pcsud15 = pd.Series(pcsud15, index = csud15.index)

plt.figure()
roma15['Tmedia'].plot()
plt.figure()
pcsud15.resample('D').sum().plot()

corrs = []
for h in range(24):
    ath = csud15['SBILANCIAMENTO FISICO [MWh]'].ix[csud15.index.hour == h]
    corrs.append(np.corrcoef(ath.values.ravel(), roma15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), roma15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs), color = 'yellow')

corrs2 = []
for h in range(24):
    ath = (-1)*csud15['PV [MWh]'].ix[csud15.index.hour == h]
    corrs2.append(np.corrcoef(ath.values.ravel(), roma15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), roma15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs2), color = 'gold')

##### NORD ######

mi15 = pd.read_table('C:/Users/d_floriello/Documents/PUN/storico_milano_aggiornato.txt')
mi15 = mi15.set_index(pd.to_datetime(ConvertDates2(mi15['Data'])))

mi15 = mi15.ix[mi15.index.year == 2015]

nord16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_NORD').ravel().tolist()]
nord15 = ST15.ix[(ST15[['CODICE RUC']].values == 'UC_DP1608_NORD').ravel().tolist()]


corrs = []
for h in range(24):
    ath = nord15['SBILANCIAMENTO FISICO [MWh]'].ix[nord15.index.hour == h]
    corrs.append(np.corrcoef(ath.values.ravel(), mi15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), mi15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs), color = 'lime')

corrs2 = []
for h in range(24):
    ath = (-1)*nord15['PV [MWh]'].ix[nord15.index.hour == h]
    corrs2.append(np.corrcoef(ath.values.ravel(), mi15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), mi15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs2), color = 'silver')

###############################################################################

PlotImbalance(ST15, ST16, 'CNOR')
PlotImbalance(ST15, ST16, 'CSUD')
PlotImbalance(ST15, ST16, 'SUD')
PlotImbalance(ST15, ST16, 'SICI')
PlotImbalance(ST15, ST16, 'SARD')

###### SARD ####

sard16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_SARD').ravel().tolist()]
sard15 = ST15.ix[(ST15[['CODICE RUC']].values == 'UC_DP1608_SARD').ravel().tolist()]

(-sard15['PV [MWh]']).plot()
plt.figure()
sard15['SBILANCIAMENTO FISICO [MWh]'].plot()

ca15 = pd.read_table('C:/Users/d_floriello/Documents/PUN/storico_cagliari_aggiornato.txt')
ca15 = ca15.set_index(pd.to_datetime(ConvertDates2(ca15['Data'])))

ca15 = ca15.ix[ca15.index.year == 2015]

corrs = []
for h in range(24):
    ath = sard15['SBILANCIAMENTO FISICO [MWh]'].ix[sard15.index.hour == h]
    corrs.append(np.corrcoef(ath.values.ravel(), ca15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), ca15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs))

corrs2 = []
for h in range(24):
    ath = (-1)*sard15['PV [MWh]'].ix[sard15.index.hour == h]
    corrs2.append(np.corrcoef(ath.values.ravel(), ca15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), ca15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs2), color = 'coral')

ca16 = pd.read_excel('C:/Users/d_floriello/Documents/PUN/Cagliari 2016.xlsx')
ca16 = ca16.set_index(pd.date_range('2016-01-01', '2016-10-31', freq = 'D'))
ca16 = ca16.ix[ca16.index.month <= 9]

corrs6 = []
for h in range(24):
    ath = sard16['SBILANCIAMENTO FISICO [MWh]'].ix[sard16.index.hour == h]
    if h == 2: 
        corrs6.append(np.corrcoef(ath.values.ravel(), np.delete(ca16['Tmedia'].values.ravel(),57))[0,1])
        print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), np.delete(ca16['Tmedia'].values.ravel(),57))[0,1]))
    else:
        corrs6.append(np.corrcoef(ath.values.ravel(), ca16['Tmedia'].values.ravel())[0,1])
        print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), ca16['Tmedia'].values.ravel())[0,1]))


plt.figure()
plt.bar(list(range(24)),np.array(corrs6), color = 'turquoise')


###### CNORD #######

fi15 = pd.read_table('C:/Users/d_floriello/Documents/PUN/storico_firenze_aggiornato.txt')
fi15 = fi15.set_index(pd.to_datetime(ConvertDates2(fi15['Data'])))

fi15 = fi15.ix[fi15.index.year == 2015]

cnord15 = ST15.ix[(ST15[['CODICE RUC']].values == 'UC_DP1608_CNOR').ravel().tolist()]
cnord16 = ST16.ix[(ST16[['CODICE RUC']].values == 'UC_DP1608_CNOR').ravel().tolist()]


corrs = []
for h in range(24):
    ath = cnord15['SBILANCIAMENTO FISICO [MWh]'].ix[cnord15.index.hour == h]
    corrs.append(np.corrcoef(ath.values.ravel(), fi15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), fi15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs), color = 'darkturquoise')

corrs2 = []
for h in range(24):
    ath = (-1)*cnord15['PV [MWh]'].ix[cnord15.index.hour == h]
    corrs2.append(np.corrcoef(ath.values.ravel(), fi15['Tmedia'].values.ravel())[0,1])
    print('correlation between ith hour and mean Temp: {}'. format(np.corrcoef(ath.values.ravel(), fi15['Tmedia'].values.ravel())[0,1]))
#    plt.figure()
#    plt.scatter(ca15['Tmedia'].values.ravel(), ath.values.ravel())
#    plt.title('correlation ' + str(h) + 'th hour')

plt.figure()
plt.bar(list(range(24)),np.array(corrs2), color = 'deepskyblue')


cnord15['SBILANCIAMENTO FISICO [MWh]'].plot()
cnord16['SBILANCIAMENTO FISICO [MWh]'].plot()

cnord15['SBILANCIAMENTO FISICO [MWh]'].resample('D').mean().plot()
cnord15['SBILANCIAMENTO FISICO [MWh]'].resample('M').mean().plot()
cnord15['SBILANCIAMENTO FISICO [MWh]'].resample('D').std().plot()
cnord15['SBILANCIAMENTO FISICO [MWh]'].resample('M').std().plot()


cnord16['SBILANCIAMENTO FISICO [MWh]'].resample('D').mean().plot(lw = 2)
cnord16['SBILANCIAMENTO FISICO [MWh]'].resample('M').mean().plot(lw = 2)
cnord16['SBILANCIAMENTO FISICO [MWh]'].resample('D').std().plot(lw = 2)
cnord16['SBILANCIAMENTO FISICO [MWh]'].resample('M').std().plot(lw = 2)

np.sign(np.diff(cnord15['SBILANCIAMENTO FISICO [MWh]'].resample('M').mean()))
np.sign(np.diff(cnord16['SBILANCIAMENTO FISICO [MWh]'].resample('M').mean()))