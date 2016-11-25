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
ST = ST.set_index(pd.date_range('2015-01-01', '2016-09-30', freq = 'H'))

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



