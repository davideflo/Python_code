# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:23:45 2016

@author: d_floriello

PDF-Extractor
"""

from __future__ import division
import PyPDF2
import re
from collections import OrderedDict
import datetime
import calendar
import pandas as pd
import math

####################################################################################################
def FilterAttiva(tt, da):
    res = []
    for d in da:
        iniziolinea = tt[d-2:d+3]
        if iniziolinea == 'ReAtt':
            pass
        else:
            res.append(d)
        return res
####################################################################################################
def Estrai_Linea_Att(tt, string, da):
    inter = tt[da: da+70]
    st = inter.find('kWh')    
    inter = inter[:st+3]
    en = ''
    counter_dot = 0
    counter_comma = 0
    for i in range(st-2, 0, -1):
        en += inter[i]
        if inter[i] == ',':
            counter_comma +=1
        elif inter[i] == '.':
            counter_dot += 1
        if counter_comma == 2:
            break
    en = en[:(len(en)-4)]
    en = en[::-1]
    en = en[:len(en)-4].replace('.','') + en[len(en)-4:].replace(',','.')
    enf = math.ceil(float(en))
    data_inizio = inter[6:16]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    return (data_inizio, data_fine, enf)
####################################################################################################
def Estrai_Attiva(tt, d):
    rea = []
    inter = tt[d: d+70]
    st = inter.find('kWh')
    inter = inter[:st+3]
    data_inizio = inter[6:16]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    dti = datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),int(str(data_inizio)[:2]))
    dtf = datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),int(str(data_fine)[:2]))
    delta = (dtf - dti).days
    days_first_month = (datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),
                        calendar.monthrange(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]))[1]) - dti).days
    days_last_month = delta - days_first_month
    en = ''
    counter_dot = 0
    counter_comma = 0
    for i in range(st-2, 0, -1):
        en += inter[i]
        if inter[i] == ',':
            counter_comma +=1
        elif inter[i] == '.':
            counter_dot += 1
        if counter_comma == 2:
            break
    en = en[:(len(en)-4)]
    en = en[::-1]
    en = en[:len(en)-4].replace('.','') + en[len(en)-4:].replace(',','.')
    enf = float(en)   
    data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]))[1]))    
    datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:4]
    data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),1))    
    datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:4]   
    rea.append((data_inizio, datafine1, math.ceil((enf/delta)*days_first_month)))
    rea.append((datainizio2, data_fine, math.ceil((enf/delta)*days_last_month)))
    return rea
####################################################################################################
def Estrai_Multiple_Att(tt, string):
    res = []
    if string in ['Att-f1', 'Att-f2', 'Att-f3']:
        da = [m.start() for m in re.finditer(string, tt)]
        da = FilterAttiva(tt, da)
        if len(da) <= 0:
            return ''
        elif len(da) == 1:
            return Estrai_Linea_Att(tt, string, da[0])
        else:
#            for d in range(0,len(da),2):
             for d in range(len(da)):
                print d
                res.append(Estrai_Linea_Att(tt, string, da[d]))
        return res
    elif string == 'Att-f0':
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) == 1:
            return Estrai_Linea_Att(tt, string, da[0])
        elif len(da) > 1:
            for d in range(0,len(da),2):
                res.append(Estrai_Linea_Att(tt, string, da[d]))
            return res
        else:
            da = [m.start() for m in re.finditer('Attiva', tt)]
            if len(da) > 0:
                for d in da:
                    res.append(Estrai_Attiva(tt,d))
                return res
            else:
                return ''
####################################################################################################        
def Estrai_Linea_ReAtt(tt, string, da):
    inter = tt[da: da+80]
    st = inter.find('kVArh')
    inter = inter[:st+5]    
    en = ''
    counter_dot = 0
    counter_comma = 0
    for i in range(st-2, 0, -1):
        en += inter[i]
        if inter[i] == ',':
            counter_comma +=1
        elif inter[i] == '.':
            counter_dot += 1
        if counter_comma == 2:
            break
    en = en[:(len(en)-4)]
    en = en[::-1]
    en = en[:len(en)-4].replace('.','') + en[len(en)-4:].replace(',','.')
    enf = math.ceil(float(en)) 
    data_inizio = inter[6:16]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    return (data_inizio, data_fine, enf)
####################################################################################################
def Estrai_Reattiva(tt, d):
    rea = []
    inter = tt[d: d+70]
    data_inizio = inter[8:18]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    dti = datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),int(str(data_inizio)[:2]))
    dtf = datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),int(str(data_fine)[:2]))
    delta = (dtf - dti).days
    days_first_month = (datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),
                        calendar.monthrange(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]))[1]) - dti).days
    days_last_month = delta - days_first_month
    en = ''
    st = inter.find('kVArh')
    counter_dot = 0
    counter_comma = 0
    for i in range(st-2, 0, -1):
        en += inter[i]
        if inter[i] == ',':
            counter_comma +=1
        elif inter[i] == '.':
            counter_dot += 1
        if counter_comma == 2:
            break
    en = en[:(len(en)-4)]
    en = en[::-1]
    en = en[:len(en)-4].replace('.','') + en[len(en)-4:].replace(',','.')
    enf = float(en)   
    data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]))[1]))    
    datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:4]
    data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),1))    
    datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:4]   
    rea.append((data_inizio, datafine1, math.ceil((enf/delta)*days_first_month)))
    rea.append((datainizio2, data_fine, math.ceil((enf/delta)*days_last_month)))
    return rea
####################################################################################################
def Estrai_Multiple_ReAtt(tt, string):
    res = []
    if string in ['ReAtt-f1', 'ReAtt-f2', 'ReAtt-f3']:
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) <= 0:
            return ''
        elif len(da) == 1:
            return Estrai_Linea_ReAtt(tt, string, da[0])
        else:
            for d in da:
                print d                
                res.append(Estrai_Linea_ReAtt(tt, string, d))
            return res
    elif string == 'ReAtt-f0':
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) == 1:
            return Estrai_Linea_ReAtt(tt, string, da[0])
        elif len(da) > 1:
            for d in da:
                res.append(Estrai_Linea_ReAtt(tt, string, d))
            return res
        else:
            da = [m.start() for m in re.finditer('Reattiva', tt)]
            if len(da) > 0:
                for d in da:
                    res.append(Estrai_Reattiva(tt,d))
                    return res
            else:
                return ''
####################################################################################################        
def Estrai_Linea_CdP(tt):
    pot = []
    lines =  [m.start() for m in re.finditer('corrispettivo di potenza', tt)]
    if len(lines) == 0:
        return ' '
    else:
        for line in lines:
            inter = tt[line:line+61]
            CdP = ''
            counter_dot = 0
            counter_comma = 0
            for i in range(len(inter)-1, 0, -1):
                CdP += inter[i]
                if inter[i] == ',':
                    counter_comma += 1
                elif inter[i] == '.':
                    counter_dot += 1
                if counter_comma == 3:
                    break
            commas = [m.start() for m in re.finditer(',', CdP)]
            second = commas[1]
            third = commas[2]
            CdP = CdP[(second-3):(third-8)]
            CdP = CdP[::-1]
            CdP = CdP.replace('.','')
            CdP = CdP.replace(',', '.')
            pot.append(math.ceil(float(CdP)))
    return pot
####################################################################################################    
def Transform(rea):
    if len(rea) > 0:
        return rea[2]
    else:
        return ''
####################################################################################################
def Transform_pot(pot,t):
    if len(pot) > 1:
        return pot[t]
    else:
        return pot[0]
####################################################################################################        
#prodotto = 'C:/Users/d_floriello/Documents/fattura_unareti.pdf'
#prodotto = 'C:/Users/d_floriello/Documents/201690120000189_Dettaglio.pdf'

#prodotto = 'C:/Users/d_floriello/Documents/201690110003764_Dettaglio.pdf'

prodotto = 'C:/Users/d_floriello/Documents/201690120001617_Dettaglio.pdf'
prodotto = 'C:/Users/d_floriello/Documents/201690110007780_Dettaglio.pdf'


pdfFileObj = open(prodotto, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
numpages = pdfReader.numPages

### numero fattura
pagina_iniziale = pdfReader.getPage(0).extractText()
nf = pagina_iniziale.find('NUMERO FATTURA')
numero_fattura = pagina_iniziale[nf+16:nf+31]

em = pagina_iniziale.find('emessa il')
emessa = pagina_iniziale[em+10:em+20]

###### put all in a single string:
text = 'INIZIO '
for i in range(1, numpages, 1):
    pageObj = pdfReader.getPage(i)
    text += pageObj.extractText()

pods = [m.start() for m in re.finditer('codice POD:', text)]


list_pod = []
not_processed = OrderedDict()
diz = OrderedDict()
for x in range(len(pods)):    
    print x
    if x < len(pods)-1:
        capitolo = text[pods[x]:pods[x+1]]
    else:
        capitolo = text[pods[x]:]
    ixpod = capitolo.find('codice POD')
    POD = capitolo[ixpod+12:ixpod+26]
    list_pod.append(POD)
    pot = Estrai_Linea_CdP(capitolo)    
    a0 = Estrai_Multiple_Att(capitolo, 'Att-f0')
    rea0 = Estrai_Multiple_ReAtt(capitolo, 'ReAtt-f0')
    a1 = Estrai_Multiple_Att(capitolo, 'Att-f1')
    rea1 = Estrai_Multiple_ReAtt(capitolo, 'ReAtt-f1')
    a2 = Estrai_Multiple_Att(capitolo, 'Att-f2')
    rea2 = Estrai_Multiple_ReAtt(capitolo, 'ReAtt-f2')
    a3 = Estrai_Multiple_Att(capitolo, 'Att-f3')
    rea3 = Estrai_Multiple_ReAtt(capitolo, 'ReAtt-f3')
    try:    
        if a0 != '':
            for a in range(len(a0)):
                tup = a0[a]
                for t in range(len(tup)):
                    index = numero_fattura + '_' + str(x) + '_' + str(t)
                    diz[index] = [emessa,POD, tup[t][0], tup[t][1], tup[t][2], '', '', '', rea0[a][t][2], '', '', '', Transform_pot(pot,t)]
        else:
            ### 1)a. is a tuple => there is only one line
            if isinstance(a1, tuple):
                index = numero_fattura + '_' + str(x)
                diz[index] = [emessa,POD, a1[0], a1[1], '', a1[2], a2[2], a3[2], '', Transform(rea1),  Transform(rea2),  Transform(rea3), pot[0]]
            ### 2) else: => more lines
            else:
                for a in range(len(a1)):
                    index = numero_fattura + '_' + str(x) + '_' + str(a)
                    if isinstance(rea1[a], tuple) and isinstance(rea2[a], tuple) and isinstance(rea3[a], tuple):
                        re1 = rea1[a][2]
                        re2 = rea2[a][2]
                        re3 = rea3[a][2]
                        diz[index] = [emessa,POD, a1[a][0], a1[a][1], '', a1[a][2], a2[a][2], a3[a][2], '', re1,  re2,  re3, pot[a]]
                    else:
                        diz[index] = [emessa,POD, a1[a][0], a1[a][1], '', a1[a][2], a2[a][2], a3[a][2], '', rea1[2],  rea2[2],  rea3[2], pot[a]]
    except:
        not_processed[numero_fattura + '_' + str(x)] = POD
        print '{} NOT PROCESSED'.format(POD)
pp = len(not_processed.keys())/len(list_pod)
print 'Percentage not processed: {}'.format(pp)
missed_pg = []
if pp > 0:
    for np in not_processed:
        for p in range(numpages):
            if np in pdfReader.getPage(p).extractText():
                missed_pg.append(p+1)

Diz = pd.DataFrame.from_dict(diz, orient = 'index')          
Diz.columns = [['data emissione','pod','lettura rilevata il','lettura rilevata il','Att-f0','Att-f1','Att-f2','Att-f3','ReAtt-f0',
               'ReAtt-f1','ReAtt-f2','ReAtt-f3','potenza']]
Diz.to_excel(numero_fattura+'_A2A.xlsx')        

NP = pd.DataFrame.from_dict(not_processed, orient = 'index')

if NP.shape[0] > 0:                
    NP.columns = [['POD']]
    NP.to_excel(numero_fattura+'_manuale_A2A.xlsx')
else:
    print 'Tutto finito'