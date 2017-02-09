# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 09:44:34 2017

@author: d_floriello

GAS Consumption

"""

from __future__ import division
import pandas as pd
import numpy as np
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
import unidecode

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
def UpdateZona(vec, j, val):
    for k in range(j, 13, 1):
        vec[k] += val
    return vec
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

resdf = pd.read_excel('Trasferimenti.xlsx', converters = {'REMI': str})
                  
####### aggregazione capacità per trasportatore
trasp = pd.read_excel('Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\DB Trasportatori.xlsx', skiprows = [0,1,2,3,4,5])
trasp = trasp[trasp.columns[-2:]]
trasp = trasp.dropna()
trasp.columns = [['REMI', 'AREA']]
trasp = trasp.ix[trasp['AREA'] != '0']

### SNAM
directory = 'Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\SNAM'
listfiles = [f for f in listdir(directory) if isfile(join(directory, f))]                 
base = [lf for lf in listfiles if 'CONF' in lf]
others = list(set(listfiles).difference(set(base)))
    
snamb = pd.read_excel(directory + '/' + base[0], sheetname = 'Punti di Riconsegna', converters = {'Codice Punto': str}) 
snama = pd.read_excel(directory + '/' + base[0], sheetname = 'Punti di uscita')   
snamb.columns = [unidecode.unidecode(x) for x in snamb.columns.tolist()]
snama.columns = [unidecode.unidecode(x) for x in snama.columns.tolist()]
remi_snam = list(set(snamb['Codice Punto'].values.tolist()))

cen = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_CEN')],12)
mer = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_MER')],12)
noc = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_NOC')],12)
nor = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_NOR')],12)
soc = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_SOC')],12)
sor = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_SOR')],12)

CGsnam = OrderedDict()
for rs in remi_snam:
    index_rs = remi_snam.index(rs)
    atrs = snamb.ix['Codice Punto' == rs]
    atr = []
    atr.append(rs)
    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rs].values.tolist()[0])
    mcg = np.repeat(atrs['Capacita Richiesta [Sm3/g]'].values.tolist()[0], 12)
    CGsnam[index_rs] = atr.extend(mcg)

cgsnam = pd.DataFrame.from_dict(CGsnam, orient = 'index')
cgsnam.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

for of in others:
    if 'TRAS' in of:
        df = pd.read_excel(directory + '/' + of, sheetname = 'Esito', converters = {'PdR Aggregato': str})
        df.columns = [unidecode.unidecode(x) for x in df.columns.tolist()]
        for i in range(df.shape[0]):
            rr = df['PdR Aggregato'].ix[i].tolist()[0]
            za = df['Codice Area di prelievo'].ix[i].tolist()[0]
            m = df['Data Inizio'].ix[i].tolist()[0][3:5]
            if za == 'M_RN_CEN':
                cen = UpdateZona(cen, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i].tolist()[0])
            elif za == 'M_RN_MER':
                mer = UpdateZona(mer, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i].tolist()[0])
            elif za == 'M_RN_NOC':
                noc = UpdateZona(noc, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i].tolist()[0])
            elif za == 'M_RN_NOR':
                nor = UpdateZona(nor, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i].tolist()[0])
            elif za == 'M_RN_SOC':
                soc = UpdateZona(soc, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i].tolist()[0])
            elif za == 'M_RN_SOR':
                sor = UpdateZona(sor, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i].tolist()[0])
            else:
                print 'NO ZONE FOUND!!!'
            if rr != '':                
                cgsnam[m].ix[cgsnam['REMI'] == rr] += df['Capacita Trasferita'].ix[i].tolist()[0] + df['Cap Addiz RR Conf/Rein RR Conf'].ix[i].tolist()[0]
                ic = cgsnam.columns.tolist().index(m)
                for j in range(ic, cgsnam.shape[1],1):
                    cgsnam[j].ix[cgsnam['REMI'] == rr] = cgsnam[m].ix[cgsnam['REMI'] == rr]

    elif 'INCR' in of:
        dfr = pd.read_excel(directory + '/' + of, sheetname = 'Punti di Riconsegna', converters = {'Codice Punto': str})
        dfr.columns = [unidecode.unidecode(x) for x in dfr.columns.tolist()]
        dfa = pd.read_excel(directory + '/' + of, sheetname = 'Punti di uscita', converters = {'Codice Punto': str})
        dfa.columns = [unidecode.unidecode(x) for x in dfa.columns.tolist()]
        for i in range(df.shape[0]):
            rr = dfr['Codice Punto'].ix[i].tolist()[0]
            za = dfa['Codice Punto'].ix[i].tolist()[0]
            m = df['Termini Temporali Da'].ix[i].tolist()[0][3:5]
            if za == 'M_RN_CEN':
                cen = UpdateZona(cen, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0])
            elif za == 'M_RN_MER':
                mer = UpdateZona(mer, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0])
            elif za == 'M_RN_NOC':
                noc = UpdateZona(noc, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0])
            elif za == 'M_RN_NOR':
                nor = UpdateZona(nor, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0])
            elif za == 'M_RN_SOC':
                soc = UpdateZona(soc, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0])
            elif za == 'M_RN_SOR':
                sor = UpdateZona(sor, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0])
            else:
                print 'NO ZONE FOUND!!!'
            if rr != '':                
                cgsnam[m].ix[cgsnam['REMI'] == rr] += dfr['Capacita Sottoscritta [Sm3/g]'].ix[i].tolist()[0] 
                ic = cgsnam.columns.tolist().index(m)
                for j in range(ic, cgsnam.shape[1],1):
                    cgsnam[j].ix[cgsnam['REMI'] == rr] = cgsnam[m].ix[cgsnam['REMI'] == rr]
        
### RETRAGAS
directory = 'Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\RETRAGAS'
listfiles = [f for f in listdir(directory) if isfile(join(directory, f))]                 
base = [lf for lf in listfiles if 'Conferimento' in lf]
others = list(set(listfiles).difference(set(base)))
    
rgb = pd.read_excel(directory + '/' + base[0], converters = {'PdrLogico': str}, skiprows = [0,1,2,3,4,5,6, 8]) 
remi_rg = list(set(rgb['PdrLogico'].values.tolist()))
    
rg = OrderedDict()
for rg in remi_rg:
    index_rs = remi_rg.index(rg)
    atrs = rgb.ix['PdrLogico' == rs]
    atr = []
    atr.append(rg)
    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rg].values.tolist()[0])
    cap = 0
    if atrs['Esito'].values.tolist()[0] > 0:
        cap = atrs['capacita'].values.tolist()[0]
    mcg = np.repeat(cap, 12)
    rg[index_rs] = atr.extend(mcg)

rg = pd.DataFrame.from_dict(rg, orient = 'index')
rg.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

for of in others:
#    if 'RCT' in of:
#        df = pd.read_excel(directory + '/' + of, converters = {'Codice Logico di riconsegna': str}, skiprows = [0,1,2,3,4,5,6,8])
#        df.columns = [unidecode.unidecode(x) for x in df.columns.tolist()]
#        for i in range(df.shape[0]):
#            rr = df['Codice Logico di riconsegna'].ix[i].tolist()[0]
#            m = df['Inizio Contratto'].ix[i].tolist()[0][3:5]
#            if rr != '':
#                rg[m].ix[rg['REMI'] == rr] += df['Capacita di trasporto richiesta (sm3/g)'].ix[i]
#                ic = rg.columns.tolist().index(m)
#                for j in range(ic, rg.shape[1],1):
#                    rg[j].ix[rg['REMI'] == rr] = rg[m].ix[rg['REMI'] == rr]
    if 'TRAS' in of or 'INCR' in of or 'RCT' in of:
        m = of[2:4]
        df = pd.read_excel(directory + '/' + of, converters = {'PdrLogico': str}, skiprows = [0,1,2,3,4,5,6, 8])
        df.columns = [unidecode.unidecode(x) for x in df.columns.tolist()]
        for i in range(df.shape[0]):
            rr = df['PdrLogico'].ix[i].tolist()[0]
            if rr != '':
                cap = 0
                if 'TRAS' in of or 'INCR' in of:
                    if df['Esito'].ix[i].tolist()[0] > 0:
                        cap = df['capacita'].ix[i]
                rg[m].ix[rg['REMI'] == rr] += cap
                ic = rg.columns.tolist().index(m)
                for j in range(ic, rg.shape[1],1):
                    rg[j].ix[rg['REMI'] == rr] = rg[m].ix[rg['REMI'] == rr]
    else:
        print "Error in the file's name"


### SGI
directory = 'Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\SGI'
listfiles = [f for f in listdir(directory) if isfile(join(directory, f))]                 
base = [lf for lf in listfiles if 'CONF' in lf]
others = list(set(listfiles).difference(set(base)))
    
sgi = pd.read_excel(directory + '/' + base[0], converters = {'Punto di Riconsegna': str}) 
sgi.columns = [unidecode.unidecode(x) for x in sgi.columns.tolist()]

remi_sgi = list(set(sgi['Punto di Riconsegna'].values.tolist()))
    
sg = OrderedDict()
for rg in remi_sgi:
    index_rs = remi_sgi.index(rg)
    atrs = sgi.ix['Punto di Riconsgna' == rs]
    atr = []
    atr.append(rg)
    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rg].values.tolist()[0])
    mcg = np.repeat(sgi['Capacita impegnata (Sm3/g)'].ix['Punto di Riconsgna' == rs], 12)
    rg[index_rs] = atr.extend(mcg)
    
sg = pd.DataFrame.from_dict(sg, orient = 'index')
sg.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]




################################################    
prof = pd.read_excel('C:/Users/d_floriello/Documents/Profili standard di prelievo 2016-17.xlsx', sheetname = '% prof', 
                     skiprows = [0,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473])    
                     

aggremi = OrderedDict()
remis = list(set(resdf['REMI'].values.tolist()))

for re in remis:
    remi_index = remis.index(re)
    atremi = resdf.ix[resdf['REMI'] == re]
    atremi = atremi.reset_index(drop=True)
    tot_cap = 0
    for i in range(atremi.shape[0]):
        tot_cap += atremi['DA CONFERIRE'].ix[i] * prof[atremi['PROFILO_PRELIEVO'].ix[i]].max()                  
                  
                  
                  