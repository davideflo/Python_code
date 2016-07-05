# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:42:09 2016

@author: d_floriello
"""

## Functions for TSP.py

import numpy as np
from numpy import fft


def find_peaks(v,al):
    v = v[0:int(v.shape[0]/2)]
    vu = np.unique(v)
    peaks = []
    rm = np.max(vu)    
    peaks.append(rm)
    for i in range(al):
        vu_mod = vu[vu < rm]
        rm = np.max(vu_mod)
        peaks.append(rm)
        if len(peaks) >= al:
            break
    return peaks
#######################################################    
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
######################################################
def Signum_Process(v):
    sp = []
    for i in range(len(v)-1):
        sp.append(np.sign(v[i+1] - v[i]))
    return np.array(sp)
#####################################################
def RMSE(v):
    return np.sqrt(np.mean(v**2))
#####################################################
def Error_Signum_process(v1, v2):
    p = v1*v2
    return p[p <= 0].size/p.size    






    