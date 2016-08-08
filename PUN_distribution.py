# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 10:30:49 2016

@author: utente

estimating a markov matrix for the monthwise conditional distributions of the PUN process.
"""

import numpy as np
import pandas as pd

data = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2010.xlsx", sheetname=1)
data2 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2011.xlsx", sheetname=0)
data3 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2012.xlsx", sheetname=0)
data4 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2013.xlsx", sheetname=0)
data5 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2014.xlsx", sheetname=0)
data6 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2015.xlsx", sheetname=0)
data7 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2016_06.xlsx", sheetname=0)


varn = "PUN"
pun1 = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
                               np.array(data3[varn]),
                         np.array(data4[varn]), np.array(data5[varn]), 
                            np.array(data6[varn]), np.array(data7[varn])])
                            
pun = {"PUN": pun1}
rng2 = pd.date_range(start="01-01-2010", periods = pun1.size,freq = 'H')
ixx = np.arange(56951)
rng = pd.DataFrame([rng2,ixx])
rng.columns = [["rng", "ixx"]]
rng.columns = ["rng"]
dd = {"ixx": ixx, "pun": pun1}
ap = pd.DataFrame(dd).set_index(rng2)

ap2 = pd.DataFrame(pun1)
ap2.columns = ["pun"]

months = np.unique(rng.month)

#########################################################
def CountCorrespondences(df, v, varn):
    ## df is already divided by month
    wv = df.ix[df[varn] == v]
    uv = np.unique(df[varn])
    iwv = df.ix[df[varn] == v].index
    res = np.zeros(uv.size)
    for i in iwv:
        if i < df.shape[0]-1:
            res[i] = np.sum(df[varn].ix[i+1] == uv[i])/wv.shape[0]
    return res
##########################################################
def FindMarkovMatrix(df, df2, month, varn):
    atmonth2 = df['ixx'].ix[df.index.month == month].tolist()
    atmonth = df2.ix[atmonth2]
    uv = np.unique(atmonth[varn])
    res = np.zeros(shape=[uv.size,uv.size])
    for v in uv:
        i = uv.tolist().index(v)
        res[i,:] = CountCorrespondences(atmonth,v,varn)    
    return res
###########################################################