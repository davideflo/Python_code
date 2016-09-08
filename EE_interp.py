# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 09:58:05 2016

@author: utente

Iterpolation of electric energy demand

"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d 
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
import matplotlib.pyplot as plt

path = 'C:\Users\utente\Documents\PUN\curve domanda'


#curves = pd.read_csv('C:\\Users\\utente\\Documents\\PUN\\curve domanda\\Default Dataset_2016-09-07.csv',sep=',', header=None)
###################################################################################################
def Extract_Dataset(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    ndict = OrderedDict()
    for of in onlyfiles:
        df = pd.read_csv('C:\\Users\\utente\\Documents\\PUN\\curve_domanda\\'+of, sep=',', header=None)
        for j in range(df.shape[0]):
            if 'h-'+str(j) in ndict:
                ndict['h-'+str(j)].append(df[df.columns[1]].ix[j])
            else:
                ndict['h-'+str(j)] = [df[df.columns[1]].ix[j]]
    DF = pd.DataFrame.from_dict(ndict)
    rng = pd.date_range(start="2016-09-07", periods = DF.shape[0],freq = 'D')
    DF = DF.set_index(rng)
    DF.to_csv('C:\\Users\\utente\\Documents\\PUN\\demand_curves.csv', sep=',', header = True)
    return DF
###################################################################################################
def get_InterpolatedCurves(path):
    DF = Extract_Dataset(path)
    x = range(DF.shape[1])
    new_x = np.linspace(0, 24, 100)
    for i in range(DF.shape[0]):
        fhat = interp1d(x, DF.ix[i], kind = 'cubic')
        fig = plt.figure()
        plt.plot(new_x, fhat(new_x))
        fig.suptitle('electric energy demand for {} '.format(str(DF.index[i])))
###################################################################################################    
#DF = Extract_Dataset(path)      
#        
#x = range(DF.shape[1])        
#new_length = 100
#new_x = np.linspace(0, 23, new_length)
#fhat = interp1d(x, DF.ix[1], kind='cubic')
#plt.figure()
#plt.plot(new_x, fhat(new_x))
