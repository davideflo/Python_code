# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 14:47:04 2017

@author: d_floriello

Fast Quoting Updating
"""

from __future__ import division
import pandas as pd
import numpy as np


####################################################################################################
def Assembler(ph):
    real = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2017/06. MI/DB_Borse_Elettriche_PER MI_17_conMacro - Copy.xlsm", sheetname = 'DB_Dati')
    real = real[real.columns[[0,1,2,3,12]]]
    real = real.dropna()
    real = real.set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:real.shape[0]])
    ld = real.index[-1]
    phf = ph.ix[ph.index > ld]
    real2 = real[real.columns[4]].append(phf)
    v = np.repeat(1, real.shape[0]).tolist()
    v.extend(np.repeat(0, 8760 - len(v)).tolist())
    real2['real'] = pd.Series(v) 
    return real2
####################################################################################################
def SimpleRedimensioniser(ph, mh, From, To):
    period = ph.ix[ph.index >= From]
    period = period.ix[period.index <= To]
    M = period.shape[0]
    phb = (1/M)*period['pun'].sum()
    pb = 0
    if period.ix[period['real'] ==1].size > 0:
        pb = (1/M)*period['pun'].ix[period['real'] ==1].sum()
    pihat = (mh - pb)/phb
    newperiod = pihat * period['pun'].values.ravel()
    df1 = ph.ix[ph.index < From]
    df2 = pd.DataFrame(newperiod).set_index(period.index)
    df3 = ph.ix[ph.index > To]
    df2 = df2.append(df3)
    df = df1.append(df2)
    df['real'] = ph['real']
    df.columns = [['pun', 'real']] 
    return df
####################################################################################################
def ConstrainedRedimensioniser(ph, mh, odiz):
    """
    @param: odiz is an ordered dict with the months of the particular Q. The Q has to be complete. If a month
    is missing, the component will be odiz['month'] = [0]. For Example:
    odiz = {'Apr': 42, 'Mag':0, 'Giu': 45}
    """
    if ph.shape[1] > 2:
        ph = ph[ph.columns[2:]]
    else:
        pass
    #num_months = len(odiz.keys())
    df = pd.DataFrame()    
    months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    index_months = [months.index(k) + 1 for k in odiz.keys()]
    phb_months = []
    for k in index_months:
        perk = ph.ix[ph.index.month == k]        
        phb_months.append(np.where(perk['real'].ix[perk['real'] == 1].size > 0, perk['pun'].ix[perk['real'] == 1].mean(), 0))
    ok = [i for i in range(len(odiz.values())) if odiz.values()[i] > 0]
    nz_months = [index_months[i] for i in ok]
    z_months = list(set(index_months).difference(set(nz_months)))    
    beta = 3*(mh - ((1/3)*sum(odiz.values()))) ##### remove the real values
    den = 0
    for i in range(len(z_months)):
       den += ph['pun'].ix[ph.index.month == z_months[i]].mean()
    beta = beta/den
    #df = df.append(ph.ix[ph.index.month < min(index_months)])
    for i in range(1,13):
        if i in z_months:
            dd = pd.DataFrame()
            dd['pun'] = beta * ph['pun'].ix[ph.index.month == i].values.ravel()
            dd['real'] = ph['real'].ix[ph.index.month == i].values.ravel()
            dd = dd.set_index(ph.ix[ph.index.month == i].index)
            df = df.append(dd)
        else:
            df = df.append(ph.ix[ph.index.month == i])
    df['real'] = ph['real'].values.ravel()        
    return df
####################################################################################################
    
    
    