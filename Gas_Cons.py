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
import datetime

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
    for k in range(j, 12, 1):
        vec[k] += val
    return vec
####################################################################################################
def cleanDF(df):
    lr = list(set(df['REMI'].values.tolist()))
    surv = []
    for l in lr:
        print l
        ldf = df.ix[df['REMI'] == l]
        if ldf.shape[0] == 1:
            surv.append(l)
        else:
            ld = []
            #ldf = ldf.reset_index(drop = True)
            for i in range(ldf.shape[0]):
                ii = ldf.index.tolist()[i]
                ld.append(np.sum(np.abs(np.diff(ldf[ldf.columns[2:]].ix[ii].values.tolist()))))
            surv.append(ldf.index.tolist()[ld.index(np.max(ld))])
    return df.ix[surv]
####################################################################################################
def RCleaner(df):
    df1 = df.ix[df['SHIPPER'] == '0001808491-AXOPOWER SRL']
    df2 = df1.ix[df['DESCRIZIONE_PRODOTTO'] != 'Solo Superi e Quota Fissa']
    df3 = df2.ix[df2['D_VALIDO_AL'] >= datetime.datetime(2016, 10, 1)]
    return df3.reset_index(drop = True)
####################################################################################################
def ActiveAtMonth(start, end):
    active = np.repeat(0,12)
    if start <= datetime.datetime(2016,10,1) and end >= datetime.datetime(2017, 9, 30):
        active = np.repeat(1, 12)
        return active
    elif start <= datetime.datetime(2016,10,1) and end <= datetime.datetime(2017, 9, 30):
        fm = end.month
        if fm >= 10:
            active[:fm-9] = 1
        else:
            active[:fm+3] = 1
        return active
    elif start >= datetime.datetime(2016,10,1) and end >= datetime.datetime(2017, 9, 30):
        im = start.month   
        if im >= 10:
            active[im-10:] = 1
        else:
            active[im+2:] = 1
        return active
    elif start >= datetime.datetime(2016,10,1) and end <= datetime.datetime(2017, 9, 30):
        im = start.month
        fm = end.month
        if im == fm:
            if im >= 10:
                active[im-10] = 1
            else:
                active[im+2] = 1
        else:
            if im >= 10 and fm >= 10:
                active[im-10:fm-9] = 1
            elif im >= 10 and fm <= 10:
                active[im-10:fm+3] = 1
            elif im <= 10 and fm <= 10:
                active[im+2:fm+3] = 1
            else:
                print 'impossible dates:'
                print start 
                print end
        return active
####################################################################################################
def Regulator(vec, k):
    res = np.repeat(0,12)
    if k >= 10:
        res[:k-9] = vec[:k-9]/100
    else:
        res[:k+3] = vec[:k+3]/100
    return res
####################################################################################################

doc1 = 'Z:/AREA ENERGY MANAGEMENT GAS/Transizione shipper/AT 2016-2017/20170201 Report Fatturato Gas_Dicembre.xlsx'
#doc2 = 'C:/Users/d_floriello/Downloads/170206-101449-218.xls'
doc2 = 'C:/Users/d_floriello/Documents/Report 2018.xls'
doc3 = 'Z:/AREA ENERGY MANAGEMENT GAS/Aggiornamento Anagrafico/1702/Anagrafica TIS EVOLUTION.xlsm'


df181 = pd.read_excel(doc1, sheetname = 'Report fatturato GAS', skiprows = [0,1], converters={'PDR': str,'REMI': str,
                      'COD_CLIENTE': str})
df218 = pd.read_excel(doc2, sheetname = 'Simil Template_Globale', skiprows = [0,1], converters={'FORNITURA_POD': str, 
                      'CLIENTE_CODICE': str, 'COD_REMI': str})
dfA = pd.read_excel(doc3, sheetname = 'Importazione', converters={'COD_PDR': str, 'COD_REMI': str})

years = [2016, 2017]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

df218 = RCleaner(df218)

cod = list(set(df218['CLIENTE_CODICE']))

res = OrderedDict()
for c in cod:
    cdf218 = df218.ix[df218['CLIENTE_CODICE'] == c]
    pdrs = list(set(cdf218['FORNITURA_POD'].values.tolist()))
    remi = cdf218['COD_REMI'].values.tolist()[-1]
    cons_contr = cdf218['CONSUMO_CONTR_ANNUO'].values.tolist()[-1]
    cons_distr = cdf218['CONSUMO_DISTRIBUTORE'].values.tolist()[-1]
    if len(pdrs) > 0 and str(pdrs[0]) != 'nan':
        for p in pdrs:
            mcount = 0
            vaf = 0
            pdf181 = df181.ix[df181['PDR'] == p]
            PDR = cdf218['PROFILO_PRELIEVO'].ix[cdf218['FORNITURA_POD'] == p].values.tolist()
            if len(PDR) == 1:
                pp = PDR[0]
            else:
                pp = PDR[-1]
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

resdf.to_excel('Trasferimenti_clean.xlsx')

resdf = pd.read_excel('Trasferimenti_clean.xlsx', converters = {'REMI': str})
                  
####### aggregazione capacità per trasportatore
trasp = pd.read_excel('Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\DB Trasportatori.xlsx', skiprows = [0,1,2,3,4,5])
trasp = trasp[trasp.columns[-2:]]
trasp = trasp.dropna()
trasp.columns = [['REMI', 'AREA']]
trasp = trasp.ix[trasp['AREA'] != '0']

tot_remi = 0
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

tot_remi += len(remi_snam)

cen = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_CEN')],12)
mer = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_MER')],12)
noc = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_NOC')],12)
nor = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_NOR')],12)
soc = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_SOC')],12)
sor = np.repeat(snama['Capacita Richiesta [Sm3/g]'].ix[snama['Codice Punto'].tolist().index('M_RN_SOR')],12)

CGsnam = OrderedDict()
for rs in remi_snam:
    index_rs = remi_snam.index(rs)
    atrs = snamb.ix[snamb['Codice Punto'] == rs]
    atr = []
    atr.append(rs)
    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rs].values.tolist()[0])
    mcg = np.repeat(atrs['Capacita Sottoscritta [Sm3/g]'].values.tolist()[0], 12)
    atr.extend(mcg)
    CGsnam[index_rs] = atr

cgsnam = pd.DataFrame.from_dict(CGsnam, orient = 'index')
cgsnam.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

### to check:
#cgsnam2 = cgsnam
#cgsnam2.to_excel('cgsnam2.xlsx')
###

NRemi = pd.DataFrame()
g_i = 0
for of in others:
    newremi = OrderedDict()    
    print of
    if 'TRAS' in of:
        df = pd.read_excel(directory + '/' + of, sheetname = 'Esito', converters = {'PdR Aggregato': str})
        df.columns = [unidecode.unidecode(x) for x in df.columns.tolist()]
        for i in range(df.shape[0]):
            g_i += 1
            rr = df['PdR Aggregato'].ix[i]
            za = df['Codice Area di prelievo'].ix[i]
            m = str(int(df['Data Inizio'].ix[i][3:5]))
            if za == 'M_RN_CEN':
                cen = UpdateZona(cen, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i])
            elif za == 'M_RN_MER':
                mer = UpdateZona(mer, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i])
            elif za == 'M_RN_NOC':
                noc = UpdateZona(noc, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i])
            elif za == 'M_RN_NOR':
                nor = UpdateZona(nor, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i])
            elif za == 'M_RN_SOC':
                soc = UpdateZona(soc, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i])
            elif za == 'M_RN_SOR':
                sor = UpdateZona(sor, int(m), df['Cap Addiz RN Conf/Cap Rila RN'].ix[i])
            else:
                print 'NO ZONE FOUND!!!'
            if str(rr) != 'nan' and rr != '':     
                try:
                    ir = cgsnam['REMI'].values.tolist().index(rr)
                    up = cgsnam[m].ix[ir] + df['Capacita Trasferita'].ix[i] + df['Cap Addiz RR Conf/Rein RR Conf'].ix[i]
                    ic = cgsnam.columns.tolist().index(m)
                    cgsnam.set_value(ir, m, up)
                    for j in range(ic, cgsnam.shape[1],1):
                        cgsnam.set_value(ir, cgsnam.columns[j], up)
                except:
                    index_rs = df['PdR Aggregato'].values.tolist().index(rr)
                    print g_i
                    atr = []
                    atr.append(rr)
                    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rr].values.tolist()[0])
                    up = df['Capacita Trasferita'].ix[i] + df['Cap Addiz RR Conf/Rein RR Conf'].ix[i]
                    mcg = np.repeat(0, 12)
                    if m in ['10','11','12']:
                        mcg[(int(m)-10):] = up
                    else:
                        mcg[int(m)+2:] = up
                    atr.extend(mcg)
                    newremi[g_i] = atr
        newremi = pd.DataFrame.from_dict(newremi, orient = 'index')
        newremi.reset_index(drop = True)
        newremi.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        NRemi = NRemi.append(newremi)

    elif 'INCR' in of:
        dfr = pd.read_excel(directory + '/' + of, sheetname = 'Punti di Riconsegna', converters = {'Codice Punto': str})
        dfr.columns = [unidecode.unidecode(x) for x in dfr.columns.tolist()]
        dfa = pd.read_excel(directory + '/' + of, sheetname = 'Punti di uscita', converters = {'Codice Punto': str})
        dfa.columns = [unidecode.unidecode(x) for x in dfa.columns.tolist()]
        for i in range(dfa.shape[0]):
            za = dfa['Codice Punto'].ix[i]
            m = str(int(dfa['Termini Temporali Da'].ix[i][3:5]))
            if za == 'M_RN_CEN':
                cen = UpdateZona(cen, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i])
            elif za == 'M_RN_MER':
                mer = UpdateZona(mer, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i])
            elif za == 'M_RN_NOC':
                noc = UpdateZona(noc, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i])
            elif za == 'M_RN_NOR':
                nor = UpdateZona(nor, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i])
            elif za == 'M_RN_SOC':
                soc = UpdateZona(soc, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i])
            elif za == 'M_RN_SOR':
                sor = UpdateZona(sor, int(m), dfa['Capacita Sottoscritta [Sm3/g]'].ix[i])
            else:
                print 'NO ZONE FOUND!!!'
        for i in range(dfr.shape[0]):
            g_i += 1
            rr = dfr['Codice Punto'].ix[i]
            print rr
            m = str(int(dfr['Termini Temporali Da'].ix[i][3:5]))
            if rr != '' and str(rr) != 'nan':      
                try:
                    ir = cgsnam['REMI'].values.tolist().index(rr)
                    up = int(cgsnam[m].ix[cgsnam['REMI'] == rr].values.tolist()[0]) + dfr['Capacita Sottoscritta [Sm3/g]'].ix[i]
                    ic = cgsnam.columns.tolist().index(m)
                    cgsnam.set_value(ir, m, up)
                    for j in range(ic, cgsnam.shape[1],1):
                        cgsnam.set_value(ir, cgsnam.columns[j], up)
                except:
                    index_rs = dfr['Codice Punto'].values.tolist().index(rr)
                    print g_i
                    atr = []
                    atr.append(rr)
                    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rr].values.tolist()[0])
                    up = dfr['Capacita Sottoscritta [Sm3/g]'].ix[i]
                    mcg = np.repeat(0, 12)
                    if m in ['10','11','12']:
                        mcg[(int(m)-10):] = up
                    else:
                        mcg[int(m)+2:] = up
                    atr.extend(mcg)
                    newremi[g_i] = atr
        newremi = pd.DataFrame.from_dict(newremi, orient = 'index')
        newremi.reset_index(drop = True)
        newremi.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        NRemi = NRemi.append(newremi)

#cgsnam = cgsnam.ix[cgsnam[cgsnam.columns[2:]].sum(axis= 1) > 0]

NR = NRemi.groupby('REMI')
NR = NR.agg(sum)

tot_remi += len(list(set(NRemi['REMI'].values.tolist())))

#if len(newremi.keys()) > 0:
#    newremi = pd.DataFrame.from_dict(newremi, orient = 'index')
#    newremi = newremi.reset_index(drop = True)
#    newremi.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
#print '#############################################################################################'
#print 'ci sono {} nuovi REMI da SNAM!'.format(cleanDF(NRemi).dropna().shape[0])
#print '#############################################################################################'


#cgsnam = cgsnam.append(cleanDF(NRemi).dropna(), ignore_index = True)    

#### another check:
#cgsnam.sum(axis = 1)
#cgsnam2.sum(axis = 1)
#cgsnam.to_excel('cgsnam.xlsx')
#### check on newremi:
#NRemi2 = NRemi
#nr = cleanDF(NRemi).dropna()
#nr.to_excel('nr.xlsx')

### RETRAGAS
directory = 'Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\RETRAGAS'
listfiles = [f for f in listdir(directory) if isfile(join(directory, f))]                 
base = [lf for lf in listfiles if 'Conferimenti' in lf]
others = list(set(listfiles).difference(set(base)))
    
rgb = pd.read_excel(directory + '/' + base[0], converters = {'PdrLogico': str}, skiprows = [0,1,2,3,4,5,6,8]) 
remi_rg = list(set(rgb['PdrLogico'].values.tolist()))
    
tot_remi += len(remi_rg)    
    
Rg = OrderedDict()
for rg in remi_rg:
    index_rs = remi_rg.index(rg)
    atrs = rgb.ix[rgb['PdrLogico'] == rg]
    atr = []
    atr.append(rg)
    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rg].values.tolist()[0])
    cap = 0
    if atrs['Esito'].values.tolist()[0] > 0:
        cap = int(atrs['capacita'].values.tolist()[0])
    mcg = np.repeat(cap, 12)
    atr.extend(mcg)
    Rg[index_rs] = atr

Rg = pd.DataFrame.from_dict(Rg, orient = 'index')
Rg.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

###### check retragas:
# Rg2 = Rg

g_i = 0
NRemi = pd.DataFrame()
for of in others:
    print of
    newremi = OrderedDict()
    if 'TRAS' in of or 'INCR' in of or 'RCT' in of:
        m = str(int(of[2:4]))
        df = pd.read_excel(directory + '/' + of, converters = {'PdrLogico': str}, skiprows = [0,1,2,3,4,5,6, 8])
        df.columns = [unidecode.unidecode(x) for x in df.columns.tolist()]
        for i in range(df.shape[0]):
            g_i += 1
            rr = df['PdrLogico'].ix[i]
            print rr
            if rr != '' and str(rr) != 'nan':
                cap = 0
                ic = Rg.columns.tolist().index(m)
                try:
                    ir = Rg['REMI'].values.tolist().index(rr)
                    if 'INCR' in of:
                        if df['Esito'].ix[i] > 0:
                            cap = df['capacita'].ix[i] + Rg[m].ix[ir]
                            Rg.set_value(ir, m, cap)
                    else:
                        cap = df['capacita'].ix[i] + Rg[m].ix[ir]
                        Rg.set_value(ir, m, cap)
                    for j in range(ic, Rg.shape[1],1):
                        Rg.set_value(ir, Rg.columns[j], cap)
                except:
                    print g_i
                    atr = []
                    atr.append(rr)
                    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rr].values.tolist()[0])
                    up = df['capacita'].ix[i]
                    mcg = np.repeat(0, 12)
                    if m in ['10','11','12']:
                        mcg[(int(m)-10):] = up
                    else:
                        mcg[int(m)+2:] = up
                    atr.extend(mcg)
                    newremi[g_i] = atr
        newremi = pd.DataFrame.from_dict(newremi, orient = 'index')
        if newremi.shape[0] > 0:
            newremi.reset_index(drop = True)
            newremi.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            Rg = Rg.append(newremi, ignore_index = True)
    else:
        print "Error in the file's name"

tot_remi += len(list(set(NRemi['REMI'].values.tolist())))


print '#############################################################################################'
print 'ci sono {} nuovi REMI da RETRAGAS!'.format(cleanDF(NRemi).shape[0])
print '#############################################################################################'


#Rg = Rg.append(NRemi, ignore_index = True)

### SGI
directory = 'Z:\AREA ENERGY MANAGEMENT GAS\ESITI TRASPORTATORI\SGI'
listfiles = [f for f in listdir(directory) if isfile(join(directory, f))]                 
base = [lf for lf in listfiles if 'CONF' in lf]
others = list(set(listfiles).difference(set(base)))
    
sgi = pd.read_excel(directory + '/' + base[0], converters = {'Punto di Riconsegna': str}) 
sgi.columns = [unidecode.unidecode(x) for x in sgi.columns.tolist()]

remi_sgi = list(set(sgi['Punto di Riconsegna'].values.tolist()))
    
tot_remi += len(remi_sgi)    
    
sg = OrderedDict()
for rg in remi_sgi:
    index_rs = remi_sgi.index(rg)
    atrs = sgi.ix[sgi['Punto di Riconsegna'] == rg]
    atr = []
    atr.append(rg)
    atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rg].values.tolist()[0])
    mcg = np.repeat(sgi['Capacita impegnata (Sm3/g)'].ix[sgi['Punto di Riconsegna'] == rg], 12)
    atr.extend(mcg)
    sg[index_rs] = atr
    
sg = pd.DataFrame.from_dict(sg, orient = 'index')
sg.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

### check SGI:
# sg2 = sg

g_i = 0
NRemi = pd.DataFrame()
for of in others:
    print of
    newremi = OrderedDict()
    df = pd.read_excel(directory + '/' + of, converters = {'PUNTO DI RICONSEGNA': str})
    df.columns = [unidecode.unidecode(x) for x in df.columns.tolist()]
    for i in range(df.shape[0]):
        print i
        g_i += 1
        rr = df['PUNTO DI RICONSEGNA'].ix[i]
        m = str(df["DATA DI INIZIO VALIDITA' DELLA CAPACITA' RICHIESTA"].ix[i].month)
        if rr != '' and str(rr) != 'nan':
            ic = sg.columns.tolist().index(m)
            try:
                ir = sg['REMI'].values.tolist().index(rr)
                up = int(sg[m].ix[sg['REMI'] == rr].values.tolist()[0]) + df["CAPACITA'\nOTTENUTA\n(Sm3/g)"].ix[i] 
                sg.set_value(ir, ic, up)
                for j in range(ic, sg.shape[1],1):
                    sg.set_value(ir, sg.columns[j], up)
            except:
                atr = []
                atr.append(rr)
                atr.append('M_RN_' + trasp['AREA'].ix[trasp['REMI'] == rr].values.tolist()[0])
                up = df["CAPACITA'\nOTTENUTA\n(Sm3/g)"].ix[i]
                mcg = np.repeat(0, 12)
                if m in ['10','11','12']:
                    mcg[(int(m)-10):] = up
                else:
                    mcg[int(m)+2:] = up
                atr.extend(mcg)
                newremi[g_i] = atr
        newremi = pd.DataFrame.from_dict(newremi, orient = 'index')
        if newremi.shape[0] > 0:
            newremi.reset_index(drop = True)
            newremi.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            NRemi = NRemi.append(newremi, ignore_index = True)

tot_remi += len(list(set(NRemi['REMI'].values.tolist())))


print '#############################################################################################'
print 'ci sono {} nuovi REMI da SGI!'.format(cleanDF(NRemi).shape[0])
print '#############################################################################################'

print '#############################################################################################'
print '#############################################################################################'
print '#############################################################################################'

print 'tot remi = {}'.format(tot_remi)

print '#############################################################################################'
print '#############################################################################################'
print '#############################################################################################'


sg = sg[sg.columns[:14]]
sg = sg.append(NRemi, ignore_index = True)

########    

cg1 = cgsnam.append(Rg, ignore_index = True)
CGM = cg1.append(sg, ignore_index= True)

grouped = CGM.groupby(['REMI', 'AREA'])
cgm2 = grouped.agg(sum)

cgm2.to_excel('agg_cgm.xlsx')
################################################ 
### Estimated requested capacity    
prof = pd.read_excel('C:/Users/d_floriello/Documents/Profili standard di prelievo 2016-17.xlsx', sheetname = '% prof', 
                     skiprows = [0,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473])    
                     

pdrprofile = OrderedDict()
remis = list(set(resdf['REMI'].values.tolist()))

for re in remis:
    #print re
    atremi = resdf.ix[resdf['REMI'] == re]
    pdrl = list(set(atremi.index.tolist()))
    for p in pdrl:
        atpdr = df218.ix[df218['FORNITURA_POD'] == p].reset_index(drop = True)
        for i in range(atpdr.shape[0]):
     #       print i
            di = atpdr['D_VALIDO_DAL'].ix[i]
            df = atpdr['D_VALIDO_AL'].ix[i]
            tot_cap = (atremi['DA CONFERIRE'].ix[p]) * (ActiveAtMonth(di, df)) * (prof[atpdr['PROFILO_PRELIEVO'].ix[i]].max())
            pre = [re, trasp['AREA'].ix[trasp['REMI'] == re].values.tolist()[0]]
            pre.extend(tot_cap.tolist())
            pdrprofile[str(p) + '_' + str(i)] = pre
    
    
pdrprofile = pd.DataFrame.from_dict(pdrprofile, orient = 'index')
#pdrprofile = pdrprofile.reset_index(drop = True)
pdrprofile.columns = [['REMI', 'AREA', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
 
gbr = pdrprofile.groupby('REMI')
GBR = gbr.agg(sum)
GBR.to_excel('stima capacita richiesta.xlsx')                 
### Estimation of residual capacity

mtoday = datetime.datetime.now().month
diff = OrderedDict()
cap_remi = [cgm2.index.tolist()[i][0] for i in range(len(cgm2.index.tolist()))]
missing = []
for i in GBR.index.tolist():
    remi = i
    try:
        cap_remi.index(remi)
        dv = cgm2.ix[remi].values.ravel() - 1.2 * Regulator(GBR.ix[i].values.ravel(), mtoday)
        ld = [remi]
        ld.extend(dv.tolist())
        diff[i] = ld
    except:
        missing.append(remi)
print 'mancano {} REMI'.format(len(missing))        
        
Diff = pd.DataFrame.from_dict(diff, orient = 'index')
Diff.columns = [['REMI', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

Diff.to_excel('capacita residue.xlsx')