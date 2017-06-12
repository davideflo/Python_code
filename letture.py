# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:36:33 2017

@author: d_floriello

LETTURE
"""

from __future__ import division
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
def ExtractorPar(df):
    ###@BRIEF: extractor for the other monthlies
    diz = OrderedDict()
    vals = ['cod_pdr','matr_mis','matr_conv','coeff_corr','freq_let','acc_mis','data_racc','let_tot_prel',
            'let_tot_conv','tipo_lettura','val_dato','num_tentativi','esito_raccolta','causa_manc_raccolta','mod_alt_racc',	
            'dir_indennizzo','pros_fin']
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
    vals = ['/cod_pdr', '/matr_mis', '/matr_conv', '/data_racc', '/data_comp',
            '/let_tot_prel', '/let_tot_conv','/tipo_lettura']
    col_pres = []
    for i,c in enumerate(df.columns):
        tl = [(i,c) for x in vals if x in c]
        col_pres.extend(tl)
        
    multi_index = []
    for C in col_pres:
        m_i = [C[1] for x in vals if x in C[1] and '#' not in C[1]]
        multi_index.extend(m_i)
    
    multi_index2 = []        
    for x in vals:
        mi2 = [mi for mi in multi_index if x in mi]
        multi_index2.extend(mi2)

    
    list_pdr = list(set(df[multi_index2[0]].values.ravel().tolist()))
    for p in list_pdr:
        dfpdr = df.ix[df[multi_index2[0]] == p].reset_index(drop = True)
        pdr = '0'*(14 - len(str(p))) + str(p)
        i = max(dfpdr.ix[dfpdr[multi_index2[-1]] == 'E'].index) if dfpdr.ix[dfpdr[multi_index2[-1]] == 'E'].shape[0] > 0 else max(dfpdr.index)
        
#        vec = [np.where(vals[j] == '/cod_pdr', pdr, dfpdr[multi_index2[j]].ix[i]) if j in multi_index2 else '' for j in range(8)]
        vec = [dfpdr[multi_index2[j]].ix[i] for j in range(len(multi_index2))]
        diz[pdr] = vec

    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas','Tipo Letture']]
    return diz
####################################################################################################    
def GGExtractor2(df):
    ###@BRIEF: extractor for dailies version 2
    diz = OrderedDict()
    vals = ['cod_pdr', 'matr_mis', 'matr_conv', 'data_racc','data_comp', 
            'let_tot_prel', 'let_tot_conv','tipo_lettura']
    col_pres = [x for x in vals if x in df.columns]
    multi_index = [vals.index(x) for x in col_pres]
    list_pdr = list(set(df['cod_pdr'].values.ravel().tolist()))
    for p in list_pdr:
        dfpdr = df.ix[df['cod_pdr'] == p].reset_index(drop = True)
        pdr = '0'*(14 - len(str(p))) + str(p)
        i = max(dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].index) if dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].shape[0] > 0 else max(dfpdr.index)
        vec = [np.where(vals[j] == 'cod_pdr', pdr, dfpdr[vals[j]].ix[i]).item() if j in multi_index else '' for j in range(7)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas','Tipo Letture']]
    return diz
####################################################################################################    
def GGExtractorPar(df):
    ###@BRIEF: extractor for dailies version 2
    diz = OrderedDict()
    vals = ['cod_pdr','matr_mis','matr_conv','val_dato_mens','esito_raccolta','data_comp','let_tot_prel',
            'let_tot_conv','tipo_lettura']
    col_pres = [x for x in vals if x in df.columns]
    multi_index = [vals.index(x) for x in col_pres]
    list_pdr = list(set(df['cod_pdr'].values.ravel().tolist()))
    for p in list_pdr:
        dfpdr = df.ix[df['cod_pdr'] == p].reset_index(drop = True)
        pdr = '0'*(14 - len(str(p))) + str(p)
        i = max(dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].index) if dfpdr.ix[dfpdr['tipo_lettura'] == 'E'].shape[0] > 0 else max(dfpdr.index)
        vec = [np.where(vals[j] == 'cod_pdr', pdr, dfpdr[vals[j]].ix[i]).item() if j in multi_index else '' for j in range(7)]
        diz[pdr] = vec
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'Matricola misuratore gas','Matricola convertitore gas','Data Lettura',
                    'Lettura misuratore gas','Lettura convertitore gas','Tipo Letture']]
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
files_month = [f for f in files if '1705' in f or '1706' in f]  ### <<-- chancge month #############################################

dfM = pd.DataFrame()
for fm in files_month:
    print fm
    if '.csv' in fm.lower():
        df = pd.read_csv(directory + fm, sep = ";", dtype = object)
        if '/Prestazione' in df.columns:
            df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
            tdf = Extractor1(df)
        elif 'Italgas' in fm or 'Umbria' in fm or 'Toscana' in fm or 'Napoletana' in fm:
            df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
            tdf = ExtractorPar(df)
        else:
            if df.columns.size < 21:
                df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
            df = df[df.columns[:21]]
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
            df = df[df.columns[:21]]
            df.columns = [['cod_servizio','cod_flusso','piva_utente'	,'piva_distr','cod_pdr','matr_mis',	
                           'matr_conv','coeff_corr'	,'freq_let','acc_mis','data_racc','let_tot_prel',	
                           'let_tot_conv','tipo_lettura','val_dato','num_tentativi','esito_raccolta','causa_manc_raccolta',
                           'mod_alt_racc','dir_indennizzo','pros_fin']]
            tdf = Extractor2(df)
    dfM = dfM.append(tdf, ignore_index = True)


####### lettura file giornalieri #############

files = os.listdir('Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TGL/2017')
directory = 'Z:/AREA ENERGY MANAGEMENT GAS/Misura/Letture Gas/TGL/2017/'
files_month = [f for f in files if '1705' in f or '1706' in f]  ### <<-- change month ################################################

missing = []
dfG = pd.DataFrame()

for fm in files_month:
    print fm
    try:
        if '.csv' in fm.lower():
            df = pd.read_csv(directory + fm, sep = ";", dtype = object)
            if '/Prestazione' in df.columns:
                df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
                tdf = GGExtractor1(df)
            elif 'Italgas' in fm or 'Umbria' in fm or 'Toscana' in fm or 'Napoletana' in fm:
                df = pd.read_csv(directory + fm, sep = ";", skiprows = [0], dtype = object)
                tdf = GGExtractorPar(df)
            else:
                if df.columns.size >= 14 and '2i Rete Gas' not in fm:
                    df = df[df.columns[:14]]
                df.columns = [['cod_servizio','cod_flusso','piva_utente','piva_distr','mese_comp','cod_pdr',
                               'matr_mis','matr_conv','val_dato_mens','esito_raccolta','data_comp','let_tot_prel',
                               'let_tot_conv','tipo_lettura']]
                tdf = GGExtractor2(df)
        else:
            df = pd.read_excel(directory + fm)
            if '/Prestazione' in df.columns:
                df = pd.read_excel(directory + fm, skiprows = [0], converters = {'/DatiPdR/cod_pdr': str})
                tdf = GGExtractor1(df)
            else:
                if df.columns.size >= 14:
                    df = df[df.columns[:14]]
                df.columns = [['cod_servizio','cod_flusso','piva_utente','piva_distr','mese_comp','cod_pdr',
                               'matr_mis','matr_conv','val_dato_mens','esito_raccolta','data_comp','let_tot_prel',
                               'let_tot_conv','tipo_lettura']]
                tdf = GGExtractor2(df)
        dfG = dfG.append(tdf, ignore_index = True)
    except:
        missing.append(fm)

print 'percentage not processed: {}'.format(len(missing)/len(files_month))

dfM.to_excel('C:/Users/d_floriello/Documents/Letture_GAS_1706.xlsx')
missed = pd.DataFrame.from_dict({'mancanti': missing}, orient = 'columns')        
missed.to_excel('C:/Users/d_floriello/Documents/Letture_GAS_1706_mancanti.xlsx')