# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:36:33 2017

@author: d_floriello

LETTURE
"""

import os
import pandas as pd
from collections import OrderedDict
import numpy as np

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
        pdr = '0'*(14 - len(str(pdr))) + str(pdr) 
        vec = [np.where(vals[j] == '/DatiPdR/cod_pdr', pdr, df[vals[j]].ix[i]) if j in multi_index else '' for j in range(8)]
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
        pdr = '0'*(14 - len(str(pdr))) + str(pdr)
        vec = [np.where(vals[j] == 'cod_pdr', pdr, df[vals[j]].ix[i]) if j in multi_index else '' for j in range(8)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas',	
                    	'Frequenza Letture','Tipo Letture']]
    return diz
####################################################################################################
def GGExtractor1(df):
    ###@BRIEF: extractor for dailies version 1
    diz = OrderedDict()
    vals = ['/DatiPdR/cod_pdr', '/DatiPdR/matr_mis', '/DatiPdR/matr_conv', '/DatiPdR/data_racc', 
            '/DatiPdR/let_tot_prel', '/DatiPdR/let_tot_conv', '/DatiPdR/freq_let', '/DatiPdR/tipo_lettura']
    col_pres = [x for x in vals if x in df.columns]
    multi_index = [vals.index(x) for x in col_pres]
    list_pdr = list(set(df['cod_pdr'].values.ravel.tolist()))
    for pdr in list_pdr:
        dfpdr = str(df['/DatiPdR/cod_pdr'].ix[df['/DatiPdR/cod_pdr'] == pdr]).reset_index(drop_index = True)
        pdr = '0'*(14 - len(str(pdr))) + str(pdr)
        i = max(dfpdr.ix[dfpdr['/DatiPdR/tipo_lettura'] == 'E'].index) if dfpdr.ix[dfpdr['/DatiPdR/tipo_lettura'] == 'E'].shape[0] > 0 else max(dfpdr.index)
        vec = [np.where(vals[j] == '/DatiPdR/cod_pdr', pdr, df[vals[j]].ix[i]) if j in multi_index else '' for j in range(8)]
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
        pdr = '0'*(14 - len(str(pdr))) + str(pdr)
        i = max(dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].index) if dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].shape[0] > 0 else max(dfpdr.index)
        vec = [np.where(vals[j] == 'cod_pdr', pdr, df[vals[j]].ix[i]) if j in multi_index else '' for j in range(8)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas',	
                    	'Frequenza Letture','Tipo Letture']]
    return diz
####################################################################################################    


#### 0. is the file an xlsx or a csv?
#### 1. read suitable columns as str
#### 2. if /Prestazione in column_names => re-read the file skipping first row
#### 3. if Extractor2 is needed => rename and lower columns like in ASA

df = pd.read_excel("Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TML/2017/1705 Unareti.xlsx")
Extractor1(df)

df = pd.read_csv("Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TML/2017/1704 ASA.csv", sep = ';')
Extractor2(df)

###### Lettura file mensili #####

files = os.listdir('Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TML/2017')
directory = 'Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TML/2017/'
files_month = [f for f in files if '1705' in f]  ### <<-- chancge month #############################################

dfM = pd.DataFrame()
for fm in files_month:
    if '.csv' in fm:
        df = pd.read_csv(directory + fm, sep = ";", dtype = object)
        if '/Prestazione' in df.columns:
            df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
            tdf = Extractor1(df)
        else:
            df.columns = [['cod_servizio','cod_flusso','piva_utente'	,'piva_distr','cod_pdr','matr_mis',	
                           'matr_conv','coeff_corr'	,'freq_let','acc_mis','data_racc','let_tot_prel',	
                           'let_tot_conv','tipo_lettura','val_dato','num_tentativi','esito_raccolta','causa_manc_raccolta',
                           'mod_alt_racc','dir_indennizzo','pros_fin']]
            tdf = Extractor2(df)
    else:
        df = pd.read_excel(directory + fm)
        if '/Prestazione' in df.columns:
            df = pd.read_excel(directory + fm, skiprows = [0], converters = {'/DatiPdR/cod_pdr': str})
            tdf = Extractor1(df)
        else:
            df = pd.read_excel(directory + fm, skiprows = [0], converters = {'cod_pdr': str})
            df.columns = [['cod_servizio','cod_flusso','piva_utente'	,'piva_distr','cod_pdr','matr_mis',	
                           'matr_conv','coeff_corr'	,'freq_let','acc_mis','data_racc','let_tot_prel',	
                           'let_tot_conv','tipo_lettura','val_dato','num_tentativi','esito_raccolta','causa_manc_raccolta',
                           'mod_alt_racc','dir_indennizzo','pros_fin']]
            tdf = Extractor2(df)
    dfM = dfM.append(tdf, ignore_index = True)


####### lettura file giornalieri #############

files = os.listdir('Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TGL/2017')
directory = 'Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TGL/2017/'
files_month = [f for f in files if '1705' in f] ### <<-- change month ################################################


for fm in files_month:
    if '.csv' in fm:
        df = pd.read_csv(directory + fm, sep = ";", dtype = object)
        if '/Prestazione' in df.columns:
            df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
            tdf = GGExtractor1(df)
        else:
            df.columns = [['cod_servizio','cod_flusso','piva_utente'	,'piva_distr','cod_pdr','matr_mis',	
                           'matr_conv','coeff_corr'	,'freq_let','acc_mis','data_racc','let_tot_prel',	
                           'let_tot_conv','tipo_lettura','val_dato','num_tentativi','esito_raccolta','causa_manc_raccolta',
                           'mod_alt_racc','dir_indennizzo','pros_fin']]
            tdf = GGExtractor2(df)
    else:
        df = pd.read_excel(directory + fm)
        if '/Prestazione' in df.columns:
            df = pd.read_excel(directory + fm, skiprows = [0], converters = {'/DatiPdR/cod_pdr': str})
            tdf = GGExtractor1(df)
        else:
            df = pd.read_excel(directory + fm, skiprows = [0], converters = {'cod_pdr': str})
            df.columns = [['cod_servizio','cod_flusso','piva_utente'	,'piva_distr','cod_pdr','matr_mis',	
                           'matr_conv','coeff_corr'	,'freq_let','acc_mis','data_racc','let_tot_prel',	
                           'let_tot_conv','tipo_lettura','val_dato','num_tentativi','esito_raccolta','causa_manc_raccolta',
                           'mod_alt_racc','dir_indennizzo','pros_fin']]
            tdf = GGExtractor2(df)
    dfM = dfM.append(tdf, ignore_index = True)
        
dfM.to_excel('C:/Users/d_floriello/Documents/Letture_GAS_1705.xlsx')
        
