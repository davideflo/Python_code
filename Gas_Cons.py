# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 09:44:34 2017

@author: d_floriello

GAS Consumption

"""

from __future__ import division
import pandas as pd
#import numpy as np
from collections import OrderedDict

####################################################################################################
def DaConferire(l):
    ### l = [cons_contr, cons_distr, sii, VAF, vaf]
    if l[3] > 0:
        return l[3]
    elif l[3] == 0 and l[2] > 0:
        return l[2]
    elif l[3] == 0 and l[2] == 0 and l[1]> 0:
        return l[1]
    else:
        return l[0]
####################################################################################################

doc1 = 'Z:/AREA ENERGY MANAGEMENT GAS/Transizione shipper/AT 2016-2017/20170201 Report Fatturato Gas_Dicembre.xlsx'
doc2 = 'C:/Users/d_floriello/Downloads/170206-101449-218.xls'
doc3 = 'Z:/AREA ENERGY MANAGEMENT GAS/Aggiornamento Anagrafico/1702/Anagrafica TIS EVOLUTION.xlsm'


df181 = pd.read_excel(doc1, sheetname = 'Report fatturato GAS', skiprows = [0,1], converters={'PDR': str,'REMI': str,
                      'COD_CLIENTE': str})
df218 = pd.read_excel(doc2, sheetname = 'Simil Template_Globale', skiprows = [0,1], converters={'FORNITURA_POD': str, 
                      'CLIENTE_CODICE': str, 'COD_REMI': str})
dfA = pd.read_excel(doc3, sheetname = 'Importazione', converters={'COD_PDR': str, 'COD_REMI': str})

years = [2016, 2017]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

cod = list(set(df218['CLIENTE_CODICE']))

res = OrderedDict()
for c in cod:
    cdf218 = df218.ix[df218['CLIENTE_CODICE'] == c]
    pdrs = list(set(cdf218['FORNITURA_POD'].values.tolist()))
    remi = cdf218['COD_REMI'].values.tolist()[-1]
    cons_contr = cdf218['CONSUMO_CONTR_ANNUO'].values.tolist()[-1]
    cons_distr = cdf218['CONSUMO_DISTRIBUTORE'].values.tolist()[-1]
    pp = cdf218['PROFILO_PRELIEVO'].values.tolist()[-1]
    if len(pdrs) > 0:
        for p in pdrs:
            mcount = 0
            vaf = 0
            pdf181 = df181.ix[df181['PDR'] == p]
            for y in years:
                ydf = pdf181.ix[pdf181['ANNO_COMPETENZA'] == y]
                if ydf.shape[0] > 0:
                    for m in months:
                        mydf = ydf.ix[ydf['MESE_COMP'] == m]
                        if mydf.shape[0] > 0:
                            mcount += 1
                            m_cons = mydf['CONSUMO_SMC'].sum() 
                            if mydf['CONSUMO_SMC'].sum() < 0:                        
                                vaf += 0
                            else:
                                vaf += m_cons
            fm12 = vaf
            sii = 0
            if len(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_PDR'] == p].values.tolist()) > 0:
                sii = float(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_PDR'] == p].values.tolist()[0])
            VAF = 0
            if mcount == 12:
                VAF = vaf
            res[p] = [remi, pp, cons_contr, cons_distr, sii, VAF, vaf, DaConferire([cons_contr, cons_distr, sii, VAF, vaf])]
    else:
        mcount = 0
        vaf = 0
        pdf181 = df181.ix[df181['REMI'] == remi]
        for y in years:
            ydf = pdf181.ix[pdf181['ANNO_COMPETENZA'] == y]
            if ydf.shape[0] > 0:
                for m in months:
                    mydf = ydf.ix[ydf['MESE_COMP'] == m]
                    if mydf.shape[0] > 0:
                        mcount += 1
                        m_cons = mydf['CONSUMO_SMC'].sum() 
                        if mydf['CONSUMO_SMC'].sum() < 0:                        
                            vaf += 0
                        else:
                            vaf += m_cons
        fm12 = vaf
        sii = 0
        if len(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_REMI'] == remi].values.tolist()) > 0:
            sii = float(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_REMI'] == remi].values.tolist()[0])
        VAF = 0
        if mcount == 12:
            VAF = vaf
        res[remi] = [remi, pp, cons_contr, cons_distr, sii, VAF, vaf, DaConferire([cons_contr, cons_distr, sii, VAF, vaf])]


resdf = pd.DataFrame.from_dict(res, orient = 'index')
resdf.columns = [['REMI', 'PROFILO_PRELIEVO', 'CONSUMO_CONTRATTUALE', 'CONSUMO_DISTRIBUTORE', 'SII', 
                  'VOLUME ANNUO FATTURATO', 'FATTURATO MIN 12', 'DA CONFERIRE']]
                  
resdf.to_excel('Trasferimenti.xlsx')