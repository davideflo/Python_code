# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:13:45 2016

@author: utente

Auxiliary script for MarkovMatrix.py
"""
import numpy as np
import seaborn as sns

#########################################################
def ExpectedLossInf(ell, Dx, kde):
    xm = Dx[Dx < ell]
    mxm = np.abs(xm[xm.size-1] - xm[0])
    deltax = mxm/xm.size
    Eloss = 0
    for i in range(xm.size):
        Eloss += ((ell - xm[i])**2) * kde[i] * deltax
    return Eloss/mxm
########################################################
def ExpectedLossSup(ell, Dx, kde):
    xm = Dx[Dx > ell]
    mxm = np.abs(xm[xm.size-1] - xm[0])
    deltax = mxm/xm.size
    Eloss = 0
    for i in range(xm.size):
        Eloss += ((ell - xm[i])**2) * kde[i] * deltax
    return Eloss/mxm
#########################################################
def ExpectedLoss(ell, Dx, kde):
    return ExpectedLossInf(ell,Dx,kde) + ExpectedLossSup(ell,Dx,kde)
#########################################################
def CountCorrespondences(df, v, varn):
    ## df is already divided by month
    wv = df.ix[np.round(df[varn]) == v]
    uv2 = np.unique(np.round(df[varn]))
    iwv = list(df.ix[np.round(df['pun']) == v].index)    
    #print iwv    
    res2 = np.zeros(uv2.size)
    for j in iwv:
        #print 'index j = {}'.format(j)
        for x in uv2:
            j2 = uv2.tolist().index(x)
            if j < df.shape[0] - 1:
                #print 'values df and uv2: {} and {}'.format(df[varn].ix[j+1], x)
                res2[j2] = np.sum(np.round(df[varn].ix[j+1]) == x)/wv.shape[0]
            else:
                pass
    return res2
##########################################################
def FindMarkovMatrix(df, df2, month, varn):
    atmonth2 = df['ixx'].ix[df.index.month == month].tolist()
    atmonth = df2.ix[atmonth2]
    
    uv = np.unique(np.round(atmonth[varn]))
    res = np.zeros(shape=[uv.size,uv.size])
    for v in uv:
        #print 'v in uv = {}'.format(v)
        i = uv.tolist().index(v)
        #print 'index = {}'.format(i)
        res[i,:] = CountCorrespondences(atmonth,v,varn)    
    return res
###########################################################
def Heatmap(res, uv):
    sns.heatmap(res, annot = True, xticklabels = uv, yticklabels = uv, cmap = "YlGnBu")