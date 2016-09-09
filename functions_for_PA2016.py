# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 11:22:40 2016

@author: utente

Pattern Analysis on 2016
"""

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
#from scipy.interpolate import interp1d
from sklearn import linear_model
#import mpmath as mp
from matplotlib.legend_handler import HandlerLine2D
import statsmodels
from pandas.tools.plotting import lag_plot


#data7 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_08.xlsx", sheetname = 1)

###############################################
def deseasonalise(x, min_s, freq):
    x_ds = []
    for i in range(0, x.size, freq):
        x_ds.append(x[i:i+24] - min_s)
    x_ds = np.array(x_ds).flatten()
    return x_ds
##############################################


####### empirical risk analysis ######
#(np.max(pun) - np.mean(pun))/np.std(pun)
# 
#index_max = pun.tolist().index(np.max(pun)) 
#rng[index_max]
# 
## how many times the values are > mean+x*sigma?
#################################################
def freq_greater_than(ts, sig, flag):
    greater = []
    for x in ts:
        #print (x - np.mean(ts))/np.std(ts) > sig
        greater.append(int((x - np.mean(ts))/np.std(ts) > sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
################################################# 
def freq_smaller_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int((x - np.mean(ts))/np.std(ts) < -sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
#################################################
def abs_freq_greater_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int(abs(x - np.mean(ts))/np.std(ts) > sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
#################################################
################################################
def rolling_mean_at(ts, time_interval):
    tsm = []
    for i in range(0, ts.size, time_interval):
        tsm.append(np.mean(ts[i:i+time_interval]))
    return np.array(tsm)
################################################
################################################
### NOT RUN
def adaptive_trend(ts, m, n, I, O, coefs,epsilon=1e-06):
    y0 = np.array(ts[m:n+1])
    x0 = np.linspace(m,n,y0.size)
    model0 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 20)
    model0.fit(x0.reshape([x0.size,1]), y0)
    coefs.append(model0.estimator_.coef_)
    for i in range(n+1, ts.size, 1):
        xnew = ts[i]
        ynew = model0.predict(i)
        error = abs(ynew - xnew)
        if error <= epsilon:
            I.append(i)
        else:
            O.append(i)
            adaptive_trend(ts, i, i+1, I, O, coefs, epsilon)
    return 0
###############################################    
### does there exist a "signal" that something is going to happen? e.g.: is there anything suggesting the trend is changing?
### percentage of two consecutive peaks for all levels of norm. distances:
def glob_perc(ts):
    res = []
    for x in range(1, 10, 1):
        sigma = freq_greater_than(ts, x, True)
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            count = 0
            for i in range(out.size - 1):
                if out[i+1] - out[i] == 1:
                    count += 1
                else:
                    pass
            res.append(float(count/np.sum(sigma)))
            print('at distance {}'.format(x))    
            print('%.6f' % float(count/np.sum(sigma)))
    return np.array(res)
###########################################################        
###########################################################
def glob_perc_neg(ts):
    res = []
    for x in range(1, 10, 1):
        sigma = freq_smaller_than(ts, x, True)
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            count = 0
            for i in range(out.size - 1):
                if out[i+1] - out[i] == 1:
                    count += 1
                else:
                    pass
            res.append(float(count/np.sum(sigma)))
            print('at distance {}'.format(x))    
            print('%.6f' % float(count/np.sum(sigma)))
    return np.array(res) 
###########################################################
def cumulative_glob_perc(ts, period, step):
    perc = []
    year = lambda y: np.ceil(y/step)
    for j in range(0, ts.size, step):
        start = np.choose(j-period >0, [0, j-period])
        ts2 = ts[start:j]
        for x in range(1, 10, 1):
            sigma = freq_greater_than(ts2, x, True)
            out = np.where(sigma > 0)[0]
            if np.sum(sigma) > 0:
                count = 0
                for i in range(out.size - 1):
                    if out[i+1] - out[i] == 1:
                        count += 1
                    else:
                        pass
                    #print(mp.mpf(count))
                    #print(mp.mpf(np.sum(sigma)))
                print('after {} year, at distance {}'.format(year(j), x)) 
                print('%.6f' % float(count/np.sum(sigma)))
                perc.append(float(count/np.sum(sigma)))
    return np.array(perc)
                    #print('%.6f' % float(mp.mpf(count)/mp.mpf(np.sum(sigma))))
#############################################################    

#### what is the average distance between peaks? 
def compute_average_distance_between_peaks(ts, flag_s):
    dist = []
    for x in range(1, 10, 1):
        #sigma = freq_greater_than(ts, x, True)
        sigma = abs_freq_greater_than(ts, x, True)        
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            dist_sigma = []
            for i in range(out.size-1):
                dist_sigma.append(out[i+1]-out[i])
            dist.append(np.nanmean(dist_sigma))
    if flag_s:
        return dist_sigma
    else:
        return np.array(dist)
###############################################
################################################
def fourierExtrapolation(x, n_predict):
    x = np.array(x)
    n = x.size
    n_harm = 100                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t        
################################################
#####################################################################################
#### probability of moving upwards or downwards ####

from sklearn.neighbors.kde import KernelDensity

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
############################################################################################
def Find_Differences_Month_Years(pun, pun2, month):
    od = OrderedDict()
    sixteen = []
    for h in range(24):
        sixteen.append(np.mean(pun2.ix[(pun2.index.month == month) & (pun2.index.hour == h)].reset_index(drop=True)))
    for y in range(2010,2016,1):
        al = []
        for h in range(24):
            al.append(np.mean(pun.ix[(pun.index.year== y) & (pun.index.month == month) & (pun.index.hour == h)].reset_index(drop=True)))
            al2 = [item for sublist in al for item in sublist]
        od[str(y)] = np.array(al2) 
    return pd.DataFrame.from_dict(od), pd.DataFrame.from_dict(sixteen)
###########################################################################################
    
###########################################################################################
def L2norm_standardised_curves(hc, hc2, year):
    yc = (hc[str(year)] - np.mean(hc[str(year)]))/np.std(hc[str(year)])
    cc = (hc2['PUN'] - hc2['PUN'].mean())/hc2['PUN'].std()
    return np.mean((yc - cc)**2)
##########################################################################################

