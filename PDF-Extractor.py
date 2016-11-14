# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:23:45 2016

@author: d_floriello

PDF-Extractor
"""

import PyPDF2
import re
from collections import OrderedDict
import datetime
import calendar
    

####################################################################################################
def Estrai_Linea_Att(tt, string, da):
    inter = tt[da: da+56]
    en = ''
    st = inter.find('kWh')
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
    data_inizio = inter[6:16]
    data_fine = inter[17:27]
    return (data_inizio, data_fine, en)
####################################################################################################
def Estrai_Attiva(tt, d):
    rea = []
    inter = tt[d: d+58]
    data_inizio = inter[6:16]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    dti = datetime.date(int(str(data_inizio)[6:])),int(str(data_inizio)[3:5]),int(str(data_inizio)[:2])
    dtf = datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),int(str(data_fine)[:2]))
    delta = (dtf - dti).days
    days_first_month = (datetime.date(int(str(data_inizio)[6:])),int(str(data_inizio)[3:5]),
                        calendar.monthrange(int(str(data_inizio)[6:]))[1], int(str(data_inizio)[3:5]) - dti).days
    days_last_month = delta - days_first_month
    en = ''
    st = inter.find('kWh')
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
    en = en[:len(en)-5].replace('.',',') + en[len(en)-4].replace(',','.')
    enf = float(en)   
    data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]))[1]))    
    datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:5]
    data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),1))    
    datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:5]   
    rea.append((data_inizio, datafine1, (enf/delta)*days_first_month),(datainizio2, data_fine, (enf/delta)*days_last_month))
####################################################################################################
def Estrai_Multiple_Att(tt, string):
    res = []
    if string in ['Att-f1', 'Att-f2', 'Att-f3']:
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) <= 0:
            return ' '
        elif len(da) == 1:
            return Estrai_Linea_Att(tt, string, da[0])
        else:
            for d in da:
                res.append(Estrai_Linea_Att(tt, string, d))
    elif string == 'Att-f0':
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) == 1:
            return Estrai_Linea_Att(tt, string, da[0])
        elif len(da) > 1:
            for d in da:
                res.append(Estrai_Linea_Att(tt, string, d))
        else:
            da = [m.start() for m in re.finditer('Attiva', tt)]
            if len(da) > 0:
                for d in d:
                    res.append(Estrai_Attiva(tt,d))
            else:
                return ' '
####################################################################################################        
def Estrai_Linea_DATICONTATORE_ReAtt(tt, string):
    da = tt.find(string)
    if da == -1:
        return ' '
    inter = tt[da: da+60]
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
    return en
####################################################################################################
def Estrai_linea_ReAtt0(tt):
    da = tt.find('ReAtt-f0')
    if da == -1:
        da2 = tt.find('Reattiva')
        if da2 == -1:
            return ' '
        else:
            return Estrai_Linea_DATICONTATORE_ReAtt(tt, 'Reattiva', True)
    else:
        return Estrai_Linea_DATICONTATORE_ReAtt(tt, 'ReAtt-f0', True)
####################################################################################################
def Estrai_Linea_ReAtt(tt, string, da):
    inter = tt[da: da+60]
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
    data_inizio = inter[6:16]
    data_fine = inter[17:27]
    return (data_inizio, data_fine, en)
####################################################################################################
def Estrai_Reattiva(tt, d):
    rea = []
    inter = tt[d: d+56]
    data_inizio = inter[8:18]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    dti = datetime.date(int(str(data_inizio)[6:])),int(str(data_inizio)[3:5]),int(str(data_inizio)[:2])
    dtf = datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),int(str(data_fine)[:2]))
    delta = (dtf - dti).days
    days_first_month = (datetime.date(int(str(data_inizio)[6:])),int(str(data_inizio)[3:5]),
                        calendar.monthrange(int(str(data_inizio)[6:]))[1], int(str(data_inizio)[3:5]) - dti).days
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
    en = en[:len(en)-5].replace('.',',') + en[len(en)-4].replace(',','.')
    enf = float(en)   
    data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]))[1]))    
    datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:5]
    data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),1))    
    datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:5]   
    rea.append((data_inizio, datafine1, (enf/delta)*days_first_month),(datainizio2, data_fine, (enf/delta)*days_last_month))
####################################################################################################
def Estrai_Multiple_ReAtt(tt, string):
    res = []
    if string in ['ReAtt-f1', 'ReAtt-f2', 'ReAtt-f3']:
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) <= 0:
            return ' '
        elif len(da) == 1:
            return Estrai_Linea_ReAtt(tt, string, da[0])
        else:
            for d in da:
                res.append(Estrai_Linea_Att(tt, string, d))
    elif string == 'ReAtt-f0':
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) == 1:
            return Estrai_Linea_ReAtt(tt, string, da[0])
        elif len(da) > 1:
            for d in da:
                res.append(Estrai_Linea_ReAtt(tt, string, d))
        else:
            da = [m.start() for m in re.finditer('Reattiva', tt)]
            if len(da) > 0:
                for d in d:
                    res.append(Estrai_Reattiva(tt,d))
            else:
                return ' '
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
            pot.append(CdP)
    return pot
####################################################################################################    
    
prodotto = 'C:/Users/d_floriello/Documents/fattura_unareti.pdf'

pdfFileObj = open(prodotto, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
numpages = pdfReader.numPages

### numero fattura
pagina_iniziale = pdfReader.getPage(0).extractText()
nf = pagina_iniziale.find('NUMERO FATTURA')
numero_fattura = pagina_iniziale[nf+16:nf+31]

###### put all in a single string:
text = 'INIZIO '
for i in range(1, numpages, 1):
    pageObj = pdfReader.getPage(i)
    text += pageObj.extractText()

pods = [m.start() for m in re.finditer('codice POD:', text)]

#'n.fattura'+','+'pod'+','+'lettura rilevata il'+','+'lettura rilevata il'+','+'Att-f0'+','+'Att-f1'+','
#+'Att-f2'+','+'Att-f3'+','+'ReAtt-f0'+','+'ReAtt-f1'+','+'ReAtt-f2'+','+'ReAtt-f3'
list_pod = []
diz = OrderedDict()
for x in range(len(pods) - 1):    
    print x
    TBI = []
    capitolo = text[pods[x]:pods[x+1]]
    ixpod = capitolo.find('codice POD')
    POD = capitolo[ixpod+12:ixpod+26]
    list_pod.append(POD)
    index = numero_fattura + '_' + str(x)    
    TBI.append(POD)
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f0', True)
    rea0 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f0')
    if a0 == ' ':        
        TBI.append(a0)        
        a1 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1', True)
        TBI.append(a1[0])
        TBI.append(a1[1])
        TBI.append(a1[2])
        a2 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f2')
        TBI.append(a2)
        a3 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f3')
        TBI.append(a3)
        TBI.append(rea0)        
        rea1 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f1')
        TBI.append(rea1)
        rea2 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f2')
        TBI.append(rea2)
        rea3 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f3')
        TBI.append(rea3)
    else:
        TBI.append(a0[0])        
        TBI.append(a0[1])        
        TBI.append(a0[2])        
        a1 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1', True)
        TBI.append(a1[0])
        a2 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f2')
        TBI.append(a2)
        a3 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f3')
        TBI.append(a3)
        TBI.append(rea0)        
        rea1 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f1')
        TBI.append(rea1)
        rea2 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f2')
        TBI.append(rea2)
        rea3 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo, 'ReAtt-f3')
        TBI.append(rea3)
    cdp = Estrai_Linea_CdP(capitolo)
    TBI.append(cdp)
    diz[index] = TBI
TBI = []
capitolo_last = text[pods[len(pods)-1]:]
ixpod = capitolo_last.find('codice POD')
POD = capitolo_last[ixpod+12:ixpod+26]
list_pod.append(POD)
index = numero_fattura + '_' + str(len(pods))    
TBI.append(POD)
a0 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f0', True)
rea0 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f0')
if a0 == ' ':        
    TBI.append(a0)        
    a1 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f1', True)
    TBI.append(a1[0])
    TBI.append(a1[1])
    TBI.append(a1[2])
    a2 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f2')
    TBI.append(a2)
    a3 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f3')
    TBI.append(a3)
    TBI.append(rea0)        
    rea1 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f1')
    TBI.append(rea1)
    rea2 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f2')
    TBI.append(rea2)
    rea3 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f3')
    TBI.append(rea3)
else:
    TBI.append(a0[0])        
    TBI.append(a0[1])        
    TBI.append(a0[2])        
    a1 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f1', True)
    TBI.append(a1[0])
    a2 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f2')
    TBI.append(a2)
    a3 = Estrai_Linea_DATICONTATORE(capitolo_last, 'Att-f3')
    TBI.append(a3)
    TBI.append(rea0)        
    rea1 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f1')
    TBI.apend(rea1)
    rea2 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f2')
    TBI.append(rea2)
    rea3 = Estrai_Linea_DATICONTATORE_ReAtt(capitolo_last, 'ReAtt-f3')
    TBI.append(rea3)
cdp = Estrai_Linea_CdP(capitolo_last)
TBI.append(cdp)
diz[index] = TBI
        
        