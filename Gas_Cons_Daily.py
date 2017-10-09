# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:21:30 2017

@author: d_floriello

Daily peak of gas consumption
"""

from __future__ import division
import pandas as pd
import numpy as np
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
import unidecode
import datetime
import time

####################################################################################################
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
        print(l)
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
def RCleaner(df, year):
    df1 = df.ix[df['SHIPPER'] == '0001808491-AXOPOWER SRL']
    df2 = df1.ix[df['DESCRIZIONE_PRODOTTO'] != 'Solo Superi e Quota Fissa']
    df3 = df2.ix[df2['D_VALIDO_AL'] >= datetime.datetime(year, 10, 1)]
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
                print('impossible dates:')
                print(start) 
                print(end)
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
def GenerateCapacity(C, m):
    cvec = np.repeat(0, 12)
    if m >= 10:
        cvec[m-10:] = C
    else:
        cvec[m+2:] = C
    return cvec
####################################################################################################
def GetStartDate(dt):
    if dt >= datetime.datetime(2016,10,1):
        return dt
    else:
        return datetime.datetime(2016,10,1)
####################################################################################################
def GetEndDate(dt):
    if dt <= datetime.datetime(2017,9,30):
        return dt
    else:
        return datetime.datetime(2017,9,30)
####################################################################################################
def WYEstimation(cons, prof, setmonth, pp):
    used_perc = 0
    for m in setmonth:
        used_perc += prof[pp].ix[prof.index.month == m].sum()
    if used_perc > 1e-3:
        return cons/(used_perc/100)
    else:
        return 0
####################################################################################################
def DaConferire(l, prof, setmonth, pp):
    ### l = [cons_contr, cons_distr, sii, VAF, vaf]
    if l[3] > 0:
        return l[3]
    elif l[3] == 0 and l[2] > 0 and l[4] >= 0:
        if l[2] >= l[4] or l[4] == 0: 
            return l[2]
        elif l[4] > l[2]:
            y = WYEstimation(l[4], prof, setmonth, pp)
            if y > 0:
                return y
            else:
                if l[2] > 0:
                    return l[2]
    elif l[3] == 0 and l[2] == 0 and l[1]> 0:
        return l[1]
    else:
        return l[0]
####################################################################################################
def GetBestConsumption(df218, df181, dfA, from_month):
    
    years = [2016, 2017]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    current_month = from_month - 2
    tym = list(set(months).intersection(set(np.arange(1,current_month + 1,1).tolist())))
    lym = list(set(months).difference(set(np.arange(1,current_month + 1,1).tolist())))
        
    cod = list(set(df218['CLIENTE_CODICE']))

    res = OrderedDict()
    for c in cod:
        cdf218 = df218.ix[df218['CLIENTE_CODICE'] == c]
        pdrs = list(set(cdf218['FORNITURA_POD'].values.tolist()))
        remi = cdf218['COD_REMI'].values.tolist()[-1]
        if len(pdrs) > 0 and str(pdrs[0]) != 'nan':
            for p in pdrs:
                setm = []
                mcount = 0
                vaf = 0
                cons_contr = cdf218['CONSUMO_CONTR_ANNUO'].ix[cdf218['FORNITURA_POD'] == p].values.tolist()[-1]
                cons_distr = cdf218['CONSUMO_DISTRIBUTORE'].ix[cdf218['FORNITURA_POD'] == p].values.tolist()[-1]            
                pdf181 = df181.ix[df181['PDR'] == p]
                PDR = cdf218['PROFILO_PRELIEVO'].ix[cdf218['FORNITURA_POD'] == p].values.tolist()
                if not isinstance(PDR, list):
                    PDR = [PDR]
                if len(PDR) == 0:
                    pp = 'T2E1'
                elif (len(PDR) == 1) and (not str(PDR[0]) == 'nan'):
                    pp = PDR[0]
                else:
                    if not str(PDR[-1]) == 'nan':
                        pp = PDR[-1]
                    else:
                        pp = 'T2E1'
                for y in years:
                    ydf = pdf181.ix[pdf181['ANNO_COMPETENZA'] == y]
                    if y == years[0]:
                        MM = lym
                    else:
                        MM = tym
                    if ydf.shape[0] > 0:
                        for m in MM:
                            mydf = ydf.ix[ydf['MESE_COMP'] == m]
                            if mydf.shape[0] > 0:
                                setm.append(m)                            
                                mcount += 1
                                m_cons = mydf['CONSUMO_SMC'].sum() 
                                if mydf['CONSUMO_SMC'].sum() < 0:                        
                                    vaf += 0
                                else:
                                    vaf += m_cons
                #fm12 = vaf
                sii = 0
                if len(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_PDR'] == p].values.tolist()) > 0:
                    sii = float(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_PDR'] == p].values.tolist()[0])
                pp = dfA['COD_PROF_PREL_STD'].ix[dfA['COD_PDR'] == p].values.tolist()[0]
                VAF = 0
                if mcount == 12:
                    VAF = vaf
                res[str(p)] = [str(p), remi, pp, cons_contr, cons_distr, sii, VAF, vaf, max([DaConferire([cons_contr, cons_distr, sii, VAF, vaf], prof, setm, pp),5]),pp]
        else:
            setm = []
            mcount = 0
            vaf = 0
            cons_contr = cdf218['CONSUMO_CONTR_ANNUO'].ix[cdf218['COD_REMI'] == remi].values.tolist()[-1]
            cons_distr = cdf218['CONSUMO_DISTRIBUTORE'].ix[cdf218['COD_REMI'] == remi].values.tolist()[-1]            
            PDR = cdf218['PROFILO_PRELIEVO'].ix[cdf218['COD_REMI'] == remi].values.tolist()[-1]
            if not isinstance(PDR, list):
                PDR = [PDR]
            if len(PDR) == 0:
                pp = 'T2E1'
            elif (len(PDR) == 1) and (not str(PDR[0]) == 'nan'):
                pp = PDR[0]
            else:
                if not str(PDR[-1]) == 'nan':
                    pp = PDR[-1]
                else:
                    pp = 'T2E1'
            pdf181 = df181.ix[df181['REMI'] == remi]
            for y in years:
                ydf = pdf181.ix[pdf181['ANNO_COMPETENZA'] == y]
                if y == years[0]:
                    MM = lym
                else:
                    MM = tym
                if ydf.shape[0] > 0:
                    for m in MM:
                        mydf = ydf.ix[ydf['MESE_COMP'] == m]
                        if mydf.shape[0] > 0:
                            setm.append(m)
                            mcount += 1
                            m_cons = mydf['CONSUMO_SMC'].sum() 
                            if mydf['CONSUMO_SMC'].sum() < 0:                        
                                vaf += 0
                            else:
                                vaf += m_cons
            #fm12 = vaf
            sii = 0
            if len(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_REMI'] == remi].values.tolist()) > 0:
                sii = float(dfA['PRELIEVO_ANNUO_PREV'].ix[dfA['COD_REMI'] == remi].values.tolist()[0])
            pp = dfA['COD_PROF_PREL_STD'].ix[dfA['COD_PDR'] == p].values.tolist()[0]
            VAF = 0
            if mcount == 12:
                VAF = vaf
            res[remi] = [remi, remi, pp, cons_contr, cons_distr, sii, VAF, vaf, max([DaConferire([cons_contr, cons_distr, sii, VAF, vaf], prof, setm, pp),5]), pp]
    

    resdf = pd.DataFrame.from_dict(res, orient = 'index')
    resdf.columns = [['PDR','REMI', 'PROFILO_PRELIEVO', 'CONSUMO_CONTRATTUALE', 'CONSUMO_DISTRIBUTORE', 'SII', 
                  'VOLUME ANNUO FATTURATO', 'FATTURATO MIN 12', 'DA CONFERIRE', 'PROFILO_PRELIEVO']]

####################################################################################################
def ATCleaner(dfprog, year):
    ### @PARAM: year is the first half year of the thermal year
    diz = OrderedDict()
    for i in range(dfprog.shape[0]):
        pdr = str(dfprog['PDR'].ix[i])
        remi = str(dfprog['REMi'].ix[i])
        pp = str(dfprog['PROFILO PRELIEVO'].ix[i])
        cap = dfprog['CONSUMO ANNUO'].ix[i]
        l = [pdr, remi]        

        di = dfprog['INIZIO FORNITURA'].ix[i] if (not isinstance(dfprog['INIZIO FORNITURA'].ix[i], str)) else dfprog['AXOPOWER SHIPPER'].ix[i]
        if di.date() < datetime.date(year,10,1):
            l.append(datetime.date(year,10,1))
        else:
            l.append(di.date())        

        df = dfprog['FINE FORNITURA'].ix[i].date()
        if df > datetime.date(year + 1,9,30):
            l.append(datetime.date(year + 1,9,30))
        else:
            l.append(df)
        
        l.append(pp)
        l.append(cap)
        diz[pdr] = l
    
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['PDR', 'REMI', 'INIZIO FORNITURA', 'FINE FORNITURA', 'PROFILO PRELIEVO', 'CONSUMO ANNUO']]
    
    return diz
####################################################################################################
def isActiveDailyAT(pdr, dfprog, year):
    dfp = dfprog.ix[dfprog['PDR'] == pdr]
    info = dfp['INIZIO FORNITURA'].values[0]
    fifo = dfp['FINE FORNITURA'].values[0]
    active_dates = pd.date_range(info, fifo, freq = 'D')
    active = np.repeat(1.0, active_dates.size)
    if info > datetime.date(year,10,1):
        before = pd.date_range(str(year) + '-10-01', info - datetime.timedelta(days = 1), freq = 'D')
        a_before = np.repeat(0.0,before.size)
        active = np.concatenate((a_before, active))
    if fifo < datetime.date(year + 1,9,30):
        after = pd.date_range(fifo + datetime.timedelta(days = 1), str(year + 1) + '-09-30', freq = 'D')
        a_after = np.repeat(0.0,after.size)
        active = np.concatenate((active, a_after))
    return active
####################################################################################################
def DailyConsumptionAT(pm, active, cons, pp, year):
    ypm = pm.ix[pm.index > datetime.date(year,9,30)]
    ypm = ypm.ix[ypm.index < datetime.date(year + 1,10,1)]    
    ypp = ypm[pp].values.ravel()/100
    dc = ypp * active * cons
    return dc
####################################################################################################
def GetDailyConsumptionAT(dfc, pm, year):
#### @PARAM: dfc is dfprog ALREADY cleaned with AT1617Cleaner
    diz = OrderedDict()
    for i in range(dfc.shape[0]):
        pdr = dfc['PDR'].ix[i]
        remi = dfc['REMI'].ix[i]
        cons = dfc['CONSUMO ANNUO'].ix[i]
        pp = dfc['PROFILO PRELIEVO'].ix[i]
        c = [pdr, remi, pp]
        active = isActiveDailyAT(pdr, dfc, year)
        dcat = DailyConsumptionAT(pm, active, cons, pp, year)
        c.extend(dcat.tolist())
        diz[pdr] = c
    
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    cols = ['PDR', 'REMI', 'PROFILO PRELIEVO']
    dts = pd.date_range(str(year) + '-10-01', str(year + 1) + '-09-30', freq = 'D')
    cols.extend(dts.tolist())
    
    diz.columns = cols
    return diz
###################################################################################################

doc1 = "Z:/AREA ENERGY MANAGEMENT GAS/Transizione shipper/AT 2017-2018/20171003 Report Fatturato Gas_Agosto.xlsx"
#doc2 = 'C:/Users/d_floriello/Downloads/170206-101449-218.xls'
doc2 = "Z:/AREA ENERGY MANAGEMENT GAS/ESITI TRASPORTATORI/17-18 Anagrafica Clienti.xlsx"
doc3 = "Z:/AREA ENERGY MANAGEMENT GAS/Aggiornamento Anagrafico Gas/1710/Anagrafica TIS EVOLUTION.xlsm"


df181 = pd.read_excel(doc1, sheetname = 'Fatturato Gas', skiprows = [0], converters={'PDR': str,'REMI': str,
                      'COD_CLIENTE': str})
df218 = pd.read_excel(doc2, sheetname = 'Simil Template_Globale', skiprows = [0,1], converters={'FORNITURA_POD': str, 
                      'CLIENTE_CODICE': str, 'COD_REMI': str})
dfA = pd.read_excel(doc3, sheetname = 'Importazione', converters={'COD_PDR': str, 'COD_REMI': str})


prof = pd.read_excel('C:/Users/d_floriello/Documents/Profili standard di prelievo 2017-18.xls.xlsx', sheetname = '% prof', 
                     skiprows = [0,2])    
sd = np.array(list(map(lambda date: date.date(), pd.date_range(start = '2017-10-01', end = '2018-09-30', freq = 'D'))))
prof = prof.set_index(sd)


prof_old = pd.read_excel('C:/Users/d_floriello/Documents/Profili_AT16-17.xlsx', skiprows = [1])    
sd = np.array(list(map(lambda date: date.date(), pd.date_range(start = '2016-01-01', end = '2017-12-31', freq = 'D'))))
prof_old = prof_old.set_index(sd)



df218 = RCleaner(df218, 2016)

resdf = GetBestConsumption(df218, df181, dfA)
#### check that every pdr has a value in "da conferire": ###########################################
if resdf['DA CONFERIRE'].ix[resdf['DA CONFERIRE'] == 0].values.size > 0:
    print('ATTENZIONE: alcuni PDR non hanno valore da conferire!!')
    print(resdf['PDR'].ix[resdf['DA CONFERIRE'] == 0].values)
else:
    print('tutti i PDR hanno valori da conferire')
####################################################################################################
### capacity constraint (=> 5) on REMI, not PDR
####################################################################################################

#### Daily consumption AT 2016-2017

prog = 'Z:/AREA ENERGY MANAGEMENT GAS/Programmazione/Programmazione/AT 2016-2017 DB Programmazione_CG.xlsx'

dfprog = pd.read_excel(prog, sheetname = 'Portafoglio Axopower', converters={'PDR': str,'REMi': str})
dfprog = dfprog[dfprog.columns[:12]]
dfprog = dfprog.ix[:3022]


dfc = ATCleaner(dfprog, 2016)                  
df1617 = GetDailyConsumptionAT(dfc, prof_old, 2016)

df1617.to_excel('consumi_giornalieri_AT1617.xlsx')

#### Daily consumption AT 2017-2018

prog = 'Z:/AREA ENERGY MANAGEMENT GAS/Programmazione/Programmazione/AT 2017-2018 DB Programmazione_CG.xlsx'

dfprog = pd.read_excel(prog, sheetname = 'Anagrafica EE+GAS_Globale', converters={'PDR': str,'REMi': str})
dfprog = dfprog[dfprog.columns[:12]]
dfprog = dfprog.ix[:2262]


dfc = ATCleaner(dfprog, 2017)                  
df1718 = GetDailyConsumptionAT(dfc, prof, 2017)

df1718.to_excel('consumi_giornalieri_AT1718.xlsx')

