# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:36:33 2017

@author: d_floriello

LETTURE
"""

import pandas as pd
from collections import OrderedDict
#import numpy as np

####################################################################################################
def Extractor1(df):
    ###@BRIEF: extractor for files from XML format ---> first row: '/Prestazione'
    diz = OrderedDict()
    vals = ['/DatiPdR/cod_pdr', '/DatiPdR/matr_mis', '/DatiPdR/matr_conv', '/DatiPdR/data_racc', 
            '/DatiPdR/let_tot_prel', '/DatiPdR/let_tot_conv', '/DatiPdR/freq_let', '/DatiPdR/tipo_lettura']
    col_pres = [x for x in vals if x in df.columns]
    multi_index = [vals.index(x) for x in col_pres]
    for i in range(df.shape[0]):
        pdr = str(df['/DatiPdR/cod_pdr'].ix[i])
        vec = [df[vals[j]].ix[i] if j in multi_index else '' for j in range(8)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas',	
                    	'Frequenza Letture','Tipo Letture']]
    return diz
####################################################################################################
def Extractor2(df):
    ###@BRIEF: extractor for the other monthlies
    diz = OrderedDict()
    vals = ['cod_pdr', 'matr_mis', 'matr_conv', 'data_racc', 
            'let_tot_prel', 'let_tot_conv', 'freq_let', 'tipo_lettura']
    col_pres = [x for x in vals if x in df.columns]
    multi_index = [vals.index(x) for x in col_pres]
    for i in range(df.shape[0]):
        pdr = str(df['cod_pdr'].ix[i])
        vec = [df[vals[j]].ix[i] if j in multi_index else '' for j in range(8)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas',	
                    	'Frequenza Letture','Tipo Letture']]
    return diz
####################################################################################################
def GGExtractor2(df):
    ###@BRIEF: extractor for dailies version 2
    diz = OrderedDict()
    vals = ['cod_pdr', 'matr_mis', 'matr_conv', 'data_racc', 
            'let_tot_prel', 'let_tot_conv', 'freq_let', 'tipo_lettura']
    col_pres = [x for x in vals if x in df.columns]
    multi_index = [vals.index(x) for x in col_pres]
    list_pdr = list(set(df['cod_pdr'].values.ravel.tolist()))
    for pdr in list_pdr:
        dfpdr = str(df['cod_pdr'].ix[df['cod_pdr'] == pdr]).reset_index(drop_index = True)
        i = max(dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].index) if dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].shape[0] > 0 else max(dfpdr.index)
        vec = [dfpdr[vals[j]].ix[i] if j in multi_index else '' for j in range(8)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas',	
                    	'Frequenza Letture','Tipo Letture']]
    return diz
####################################################################################################    


#### 0. is the file a xlsx or a csv?
#### 1. read suitable columns as str
#### 2. if /Prestazione in column_names => re-read the file skipping first row
#### 3. if Extractor2 is needed => rename and lower columns like in ASA

df = pd.read_excel("Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TML/2017/1705 Unareti.xlsx")
Extractor1(df)

df = pd.read_csv("Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TML/2017/1704 ASA.csv", sep = ';')
Extractor2(df)