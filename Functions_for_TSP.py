# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:42:09 2016

@author: d_floriello
"""

## Functions for TSP.py

import numpy as np
from numpy import fft
import pandas as pd

import temp


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
#####################################################
def replicate_meteo_variables(meteo, vd):
    rtmin = []
    rtmax = []
    rtmed = []
    rrain = []
    rvm = []
    date_meteo = np.array(meteo[meteo.columns[0]]).tolist() 
    for i in range(0, np.array(vd).size,24):
        ir = date_meteo.index(vd[i])
        rtmin.append(np.repeat(meteo["Tmin"].ix[ir], 24, axis = 0))
        rtmax.append(np.repeat(meteo["Tmax"].ix[ir], 24, axis = 0))
        rtmed.append(np.repeat(meteo["Tmedia"].ix[ir], 24, axis = 0))
        rrain.append(np.repeat(meteo["Pioggia"].ix[ir], 24, axis = 0))
        rvm.append(np.repeat(meteo["Vento_media"].ix[ir], 24, axis = 0))
    meteodict = {"Tmin": np.array(rtmin).flatten(), "Tmax": np.array(rtmax).flatten(),
                 "Tmedia": np.array(rtmed).flatten(), "Pioggia": np.array(rrain).flatten(),
                 "Wind": np.array(rvm).flatten()}
    
    return meteodict       
#####################################################
def generate_dataset_ARIMA(pun, first_day, meteo, varn):
    vector_date = np.array(pun[pun.columns[0]])
    vector_ore = np.array(pun[pun.columns[1]])
    target = np.array(pun[varn])
    global_dates = temp.dates(pd.Series(vector_date))
    vac_glob = temp.add_holidays(global_dates) 

    all_days = temp.generate_days(vector_ore, first_day)

    MD = replicate_meteo_variables(meteo, global_dates)

    aad = np.array([temp.convert_day_to_angle(v) for v in all_days]) 
    aore = np.sin(vector_ore*np.pi/24) 
    
    all_dict = {'holiday' : vac_glob, 'day' : aad, 'ora' : aore}
    all_dict.update(MD)
    
    FDF = pd.DataFrame(all_dict)
    
    return FDF, target
####################################################
def finding_missing_dates(date, meteo):
    md = []
    date = np.unique(np.array(date))
    npm = np.array(meteo[meteo.columns[0]])
    for d in date:
        if d not in npm:
#            print(d)
#            print(d not in npm)
            md.append(d)
    return md
###################################################
def update_meteo(meteo1, meteo2):
    # meteo2 is supposed to be the most complete one
    date1 = np.unique(np.array(meteo1[meteo1.columns[0]]))
    date2 = np.unique(np.array(meteo2[meteo2.columns[0]]))
    date2list = date2.tolist()
    md = []
    index = []
    for d in date2:
        if d not in date1:
            md.append(d)
            index.append(date2list.index(d))
    upmeteo = pd.DataFrame(meteo2.ix[index])
    met = {'Data': meteo1[meteo1.columns[0]],
           'Tmin': meteo1[meteo1.columns[1]],
           'Tmedia': meteo1[meteo1.columns[2]],
           'Tmax': meteo1[meteo1.columns[3]],
           'Pioggia': meteo1[meteo1.columns[4]],
           'Vento_media': meteo1[meteo1.columns[8]]}
    nmet = {'Data': upmeteo[upmeteo.columns[0]],
           'Tmin': upmeteo[upmeteo.columns[1]],
           'Tmedia': upmeteo[upmeteo.columns[2]],
           'Tmax': upmeteo[upmeteo.columns[3]],
           'Pioggia': upmeteo[upmeteo.columns[4]],
           'Vento_media': upmeteo[upmeteo.columns[8]]}       
           
    meteodf = pd.DataFrame(met)
    nmetdf = pd.DataFrame(nmet)    
       
    updatedmeteo = pd.concat([meteodf, nmetdf]).reset_index(drop=True)
    return updatedmeteo
##########################################################
def simulate_meteo(meteo1, roma):
    index = []
    index2 = []
    vd = np.array(meteo1[meteo1.columns[0]]).tolist()
    vd2 = np.array(roma[roma.columns[0]]).tolist()
    for d in roma[roma.columns[0]]:
        if d in vd:
            index.append(vd.index(d))
            index2.append(vd2.index(d))
        else:
            pass
    found = meteo1[meteo1.columns[[1,2,3,4,8]]].ix[index].reset_index(drop=True)
    roma2 = roma[roma.columns[[1,2,3,4,8]]].ix[index2].reset_index(drop=True)
    diffdf = found - roma2
    return diffdf
##########################################################
def generate_simulated_meteo_dataset(meteo, roma):
    diff = simulate_meteo(meteo, roma)
    date = np.array(roma[roma.columns[0]]).tolist()
    vd = np.array(meteo[meteo.columns[0]]).tolist()
    tmin = []
    tmed = []
    tmax = []
    rain = []
    vm = []
    for d in date:
        if d in vd:
            tmin.append(meteo['Tmin'].ix[vd.index(d)])
            tmed.append(meteo['Tmedia'].ix[vd.index(d)])
            tmax.append(meteo['Tmax'].ix[vd.index(d)])
            rain.append(meteo['Pioggia'].ix[vd.index(d)])
            vm.append(meteo['Vento_media'].ix[vd.index(d)])
        else:
            tmin.append(roma['Tmin'].ix[date.index(d)] + np.mean(diff['Tmin']))
            tmed.append(roma['Tmedia'].ix[date.index(d)] + np.mean(diff['Tmedia']))
            tmax.append(roma['Tmax'].ix[date.index(d)] + np.mean(diff['Tmax']))
            rain.append(roma['Pioggia'].ix[date.index(d)] + np.mean(diff['Pioggia']))
            vm.append(roma['Vento_media'].ix[date.index(d)] + np.mean(diff['Vento_media']))
    pdf = {'Data': date, 'Tmin': np.array(tmin), 'Tmedia': np.array(tmed), 'Tmax': np.array(tmax),
           'Pioggia': np.array(rain), 'Vento_media': np.array(vm)}
    return pd.DataFrame(pdf).ix[:, ['Data', 'Tmin', 'Tmedia', 'Tmax', 'Pioggia', 'Vento_media']]
    
    
    
    

    