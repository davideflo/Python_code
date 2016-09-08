# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 11:57:49 2016

@author: utente

To compute probability of PUN going upwards
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.neighbors.kde import KernelDensity
import scipy.integrate as integrate

data = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2010.xlsx", sheetname = 1)
data2 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2011.xlsx", sheetname = 0)
data3 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2012.xlsx", sheetname = 0)
data4 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2013.xlsx", sheetname = 0)
data5 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2014.xlsx", sheetname = 0)
data6 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2015.xlsx", sheetname = 0)
data7 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2016_08.xlsx", sheetname = 1)
#########################################################################################################
def hkde(bandwidth, hour, ln):
    wh = []
    if not isinstance(ln, str):
        for n in ln:
            D = globals()[n]
            wh.append(D['PUN'].ix[D[D.columns[1]] == hour])
        wh2 = [val for sublist in wh for val in sublist]    
        wh = np.array(wh2)
        kdew = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(wh.reshape(-1,1))
        return kdew
    else:
        D = globals()[ln]
        wh.append(D['PUN'].ix[D[D.columns[1]] == hour])
        wh2 = [val for sublist in wh for val in sublist]    
        wh = np.array(wh2)
        kdew = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(wh.reshape(-1,1))
        return kdew
######################################################################################
######################################################################################
def compute_probability(low, up, distr):

    def distribution(x,distr):
        x = np.array(x)
        return np.exp(distr.score_samples(x.reshape(-1,1)))
        
    J = integrate.quad(distribution,low,up, args = (distr,))  

    return J
######################################################################################
def Expected_Loss_inf(v, distr):
    
    def f(x,v,distr):
        x = np.array(x)
        return ((x - v) ** 2) * np.exp(distr.score_samples(x.reshape(-1,1)))
        
    J = integrate.quad(f, 0, v, args = (v,distr))
    return J
######################################################################################
def Expected_Loss_sup(v, distr):
    
    def f(x,v,distr):
        x = np.array(x)
        return ((x - v) ** 2) * np.exp(distr.score_samples(x.reshape(-1,1)))
        
    J = integrate.quad(f, v, np.inf, args = (v,distr))
    return J
######################################################################################
######################################################################################
def Expected_Loss(v, distr):
    return Expected_Loss_sup(v, distr)[0] + Expected_Loss_inf(v, distr)[0]
######################################################################################
def get_Probabilities(path):
    pred = pd.read_excel(path)
    diz = OrderedDict({})
    for h in range(1,25,1):
        hdsitr = hkde(4, h, ['data', 'data2', 'data3', 'data4', 'data5', 'data6'])
        m = pred[pred.columns[1]].ix[h-1]
        p = compute_probability(m, np.inf, hdsitr)
        diz['ora-'+str(h)] = p
    probs = pd.DataFrame.from_dict(diz)
    return probs
######################################################################################
def get_Probabilities2(path, dataset):
    pred = pd.read_excel(path)
    diz = OrderedDict({})
    for h in range(1,25,1):
        hdsitr = hkde(4, h, dataset)
        m = pred[pred.columns[1]].ix[h-1]
        p = compute_probability(m, np.inf, hdsitr)
        diz['ora-'+str(h)] = p
    probs = pd.DataFrame.from_dict(diz)
    return probs    
######################################################################################    
def Find_Anomalies(df):
    counter = 0
    for i in range(df.shape[1]):
        if df[df.columns[i]].ix[0] >= 0.75:
            print '{} is at risk'.format(df.columns[i])
            counter += 1
    if counter == 0:
        print 'nothing to note'
######################################################################################
def get_ExpectedLoss(path, dataset):
    pred = pd.read_excel(path)
    diz = OrderedDict()
    for h in range(1,25,1):
        hdistr = hkde(4, h, dataset)
        m = pred[pred.columns[1]].ix[h-1]
        loss = Expected_Loss(m, hdistr)
        diz['ora-'+str(h)] = [loss]
    DD = pd.DataFrame.from_dict(diz)
    return(DD)
######################################################################################    