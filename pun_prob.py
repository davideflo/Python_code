# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:37:17 2016

@author: d_floriello

probability of moving upwards or downwards, given pun predictions
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
#from scipy.interpolate import interp1d
#import mpmath as mp



data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")
data2 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2011.xlsx")
data3 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2012.xlsx")
data4 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2013.xlsx")
data5 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2014.xlsx")
data6 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2015.xlsx")

data7 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_07.xlsx", sheetname = 1)

from sklearn.neighbors.kde import KernelDensity
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
##########################################################################################################
import scipy.integrate as integrate
######################################################################################
def compute_probability(low, up, distr):
#    x = np.linspace(start=low,stop=up,num=1000)
#    logy = distr.score_samples(x.reshape(-1,1))
    def distribution(x,distr):
        x = np.array(x)
        return np.exp(distr.score_samples(x.reshape(-1,1)))
#     quad wants a single value as first argument???
    J = integrate.quad(distribution,low,up, args = (distr,))  
#    I = integrate.quad(lambda x: np.exp(logy),low,up, args = x)
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
def get_Probabilities(path):
    pred = pd.read_excel(path)
    diz = OrderedDict({})
    for h in range(1,25,1):
        hdsitr = hkde(4, h, 'data7')
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
    
    
    
    