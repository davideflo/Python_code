# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:51:26 2017

@author: d_floriello

functions for A2A 
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
def diff_month(d1, d2):
    return (d1.year - d2.year)*12 + d1.month - d2.month
####################################################################################################
def IsLastDay(dtt):
    DAY = dtt.day
    MONTH = dtt.month
    YEAR = dtt.year
    cal = calendar.monthrange(YEAR, MONTH)[1]
    if DAY == cal:
        return True
    else:
        return False
####################################################################################################
def listmonth(dti, dtf):
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    resm = []
    monthdiff = diff_month(dtf, dti)
    ii = 0    
    if IsLastDay(dti):
        ii = months.index((dti + datetime.timedelta(days = 1)).month) 
    else:
        ii = months.index(dti.month)
    for j in range(monthdiff):
        ind = (ii+j)%12
        resm.append(months[ind])
    return resm
####################################################################################################
def listmonth2(dti, dtf):
    resm = []    
    if IsLastDay(dti):
        dti = dti + datetime.timedelta(days = 1)
    med = datetime.datetime(dti.year, dti.month, calendar.monthrange(dti.year, dti.month)[1])
    resm.append((dti, med))
    dff = datetime.datetime(dtf.year, dtf.month, calendar.monthrange(dtf.year, dtf.month)[1])
    while med != dff:
        medi = med + datetime.timedelta(days = 1)
        dfi = datetime.datetime(medi.year, medi.month, calendar.monthrange(medi.year, medi.month)[1])
        resm.append((medi, dfi))
        med = dfi
    return resm    
####################################################################################################
def FilterAttiva(tt, da):
    res = []
    for d in da:
        iniziolinea = tt[d-2:d+3]
        if iniziolinea == 'ReAtt':
            print 'Reatt --> discard'
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
    deltad = (dtf - dti).days
#    deltam = 0
#    if calendar.monthrange(dti.year, dti.month) == dti.day:
#        deltam = dtf.month - dti.month
#    else:
#        deltam = (dtf.month - dti.month) + 1
#    days_first_month = (datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),
#                        calendar.monthrange(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]))[1]) - dti).days
#    days_last_month = delta - days_first_month
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
    
    #condition2 = (dtf - dti).days <= min([calendar.monthrange(dtf.year, dtf.month)[1],calendar.monthrange(dti.year, dti.month)[1]])
    condition = (dti.month != dtf.month)    
    
    if condition:
        lm = listmonth2(dti, dtf)
        for l in range(len(lm)):
            dr = lm[l]
            print dr
            datainizio = str(dr[0].day) + '/' + str(dr[0].month) + '/' + str(dr[0].year)
            datafine = str(dr[1].day) + '/' + str(dr[1].month) + '/' + str(dr[1].year)
            valore = math.ceil((enf/deltad)*dr[1].day)
            rea.append((datainizio, datafine, valore))
#           
#            if dr == dti.month:
#                dff = datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]),dr)[1])
#                data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]),dr)[1]))    
#                datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:4]
#                delta = (dff - dti).days
#                if not IsLastDay(dff):
#                    rea.append((data_inizio, datafine1, math.ceil((enf/deltad)*delta)))
#            elif dti.month < dr <dtf.month:
#                data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),dr,calendar.monthrange(int(str(data_inizio)[6:]),dr)[1]))    
#                datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:4]
#                data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),dr,1))    
#                datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:4] 
#                delta = calendar.monthrange(int(str(data_inizio)[6:]),dr)[1]
#                rea.append((datainizio2, datafine1, math.ceil((enf/deltad)*delta)))
#            else:
#                data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),dr,1))    
#                datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:4] 
#                delta = (dtf - datetime.date(int(str(data_inizio2)[:4]),dr,1)).days + 1
#                rea.append((datainizio2, data_fine, math.ceil((enf/deltad)*delta)))
    
#    elif (not condition) and condition2:
#        rea.append((data_inizio, data_fine, enf))
    else:
        rea.append((data_inizio, data_fine, enf))
        
    return rea
####################################################################################################
def Estrai_Multiple_Att(tt, string):
    res = []
    if string in ['Att-f1', 'Att-f2', 'Att-f3']:
        da = [m.start() for m in re.finditer(string, tt)]
        da = FilterAttiva(tt, da)
        if da is None:
            return ''
        elif len(da) <= 0:
            return ''
        elif len(da) == 1:
#            return Estrai_Linea_Att(tt, string, da[0])
             return Estrai_Attiva(tt, da[0])
        else:
#            for d in range(0,len(da),2):
             for d in range(len(da)):
                print d
#                res.append(Estrai_Attiva(tt, da[d]))
                res.append(Estrai_Linea_Att(tt, string, da[d]))               
             return res
    elif string == 'Att-f0':
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) == 1:
#            return Estrai_Linea_Att(tt, string, da[0])
            return Estrai_Attiva(tt, da[0])
        elif len(da) > 1:
            for d in range(0,len(da),2):
#                res.append(Estrai_Linea_Att(tt, string, da[d]))
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
    data_inizio = inter[8:16]
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    return (data_inizio, data_fine, enf)
####################################################################################################
def Estrai_Reattiva(tt, d):
    rea = []
    inter = tt[d: d+70]
    data_inizio = inter[8:18]
    st = inter.find('kVArh')    
    bsl = [m.start() for m in re.finditer('/', inter)]
    data_fine = inter[(bsl[2]-2):(bsl[3]+5)]
    dti = datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),int(str(data_inizio)[:2]))
    dtf = datetime.date(int(str(data_fine)[6:]),int(str(data_fine)[3:5]),int(str(data_fine)[:2]))
    deltad = (dtf - dti).days
#    deltam = 0
#    if calendar.monthrange(dti.year, dti.month) == dti.day:
#        deltam = dtf.month - dti.month
#    else:
#        deltam = (dtf.month - dti.month) + 1
#    days_first_month = (datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),
#                        calendar.monthrange(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]))[1]) - dti).days
#    days_last_month = delta - days_first_month
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
    
    condition = (dti.month != dtf.month)    
    
    if condition:
        lm = listmonth2(dti, dtf)
        for l in range(len(lm)):
            dr = lm[l]
            print dr
            datainizio = str(dr[0].day) + '/' + str(dr[0].month) + '/' + str(dr[0].year)
            datafine = str(dr[1].day) + '/' + str(dr[1].month) + '/' + str(dr[1].year)
            valore = math.ceil((enf/deltad)*dr[1].day)
            rea.append((datainizio, datafine, valore))
#            if dr == dti.month:
#                dff = datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]),dr)[1])
#                data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),int(str(data_inizio)[3:5]),calendar.monthrange(int(str(data_inizio)[6:]),dr)[1]))    
#                datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:4]
#                delta = (dff - dti).days
#                if not IsLastDay(dff):
#                    rea.append((data_inizio, datafine1, math.ceil((enf/deltad)*delta)))
#            elif dti.month < dr <dtf.month:
#                data_fine1 = str(datetime.date(int(str(data_inizio)[6:]),dr,calendar.monthrange(int(str(data_inizio)[6:]),dr)[1]))    
#                datafine1 = data_fine1[8:10]+'/'+data_fine1[5:7]+'/'+data_fine1[:4]
#                data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),dr,1))    
#                datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:4] 
#                delta = calendar.monthrange(int(str(data_inizio)[6:]),dr)[1]
#                rea.append((datainizio2, datafine1, math.ceil((enf/deltad)*delta)))
#            else:
#                data_inizio2 = str(datetime.date(int(str(data_fine)[6:]),dr,1))    
#                datainizio2 = data_inizio2[8:10]+'/'+data_inizio2[5:7]+'/'+data_inizio2[:4] 
#                delta = (dtf - datetime.date(int(str(data_inizio2)[:4]),dr,1)).days + 1
#                rea.append((datainizio2, data_fine, math.ceil((enf/deltad)*delta)))
    
#    elif (not condition) and condition2:
#        rea.append((data_inizio, data_fine, enf))
    else:
        rea.append((data_inizio, data_fine, enf))
    
    return rea
####################################################################################################
def Estrai_Multiple_ReAtt(tt, string):
    res = []
    if string in ['ReAtt-f1', 'ReAtt-f2', 'ReAtt-f3']:
        da = [m.start() for m in re.finditer(string, tt)]
        if da is None:
            return ''
        elif len(da) <= 0:
            return ''
        elif len(da) == 1:
#            return Estrai_Linea_ReAtt(tt, string, da[0])
            return Estrai_Reattiva(tt, da[0])
        else:
            for d in da:
                print d                
#                res.append(Estrai_Linea_ReAtt(tt, string, d))
                res.append(Estrai_Reattiva(tt, d))
            return res
    elif string == 'ReAtt-f0':
        da = [m.start() for m in re.finditer(string, tt)]
        if len(da) == 1:
#            return Estrai_Linea_ReAtt(tt, string, da[0])
            return Estrai_Reattiva(tt, da[0])
        elif len(da) > 1:
            for d in da:
#                res.append(Estrai_Linea_ReAtt(tt, string, d))
                res.append(Estrai_Reattiva(tt, d))
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
    tt2 = tt.find('PERIODO')
    DA = tt[tt2:]
    periodi = [m.start() for m in re.finditer('dal', DA)]
    for k in range(len(periodi)):
        #print k
        if k < len(periodi)-1:
            DA2 = DA[periodi[k]:periodi[k+1]]
        else:
            DA2 = DA[periodi[k]:DA.find('Totale')]
        p = 0
        lines = [DA2.find('corrispettivo di potenza')]
        if len(lines) == 0 or lines[0] == -1:
            pot. append('')
        else:
            init = DA2[p+4:p+14]
            fine = DA2[p+18:p+28]
            d_init = datetime.date(int(str(init)[6:]),int(str(init)[3:5]),int(str(init)[0:2]))
            d_fine = datetime.date(int(str(fine)[6:]),int(str(fine)[3:5]),int(str(fine)[0:2]))
            month_diff = d_fine.month - d_init.month + 1
            #lines =  [m.start() for m in re.finditer('corrispettivo di potenza', tt)]
            for line in lines:
                inter = DA2[line:line+61]
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
                for m in range(month_diff):
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
def A2A_Executer(prodotto):
        
    #prodotto = 'C:/Users/d_floriello/Documents/201690120001617_Dettaglio.pdf'
    #prodotto = 'C:/Users/d_floriello/Documents/201690110007780_Dettaglio.pdf'
    #prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/1. UNARETI/201690120001827_Dettaglio.pdf'
    #unareti(prodotto)
    
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
        if len(rea1) > 0 and isinstance(rea1[0], list):
            rea1 = [item for sublist in rea1 for item in sublist]
        if len(rea2) > 0 and isinstance(rea2[0], list):
            rea2 = [item for sublist in rea2 for item in sublist]
        if len(rea3) > 0 and isinstance(rea3[0], list):
            rea3 = [item for sublist in rea3 for item in sublist]
        Rea1 = []
        Rea2 = []
        Rea3 = []
        Rea0 = []
        if len(rea1) == 0:
            for i in range(len(a1)):
                Rea1.append(('', '', ''))            
                rea1 = Rea1                
        if len(rea2) == 0:
            for i in range(len(a2)):
                Rea2.append(('', '', ''))     
                rea2 = Rea2                       
        if len(rea3) == 0:
            for i in range(len(a1)):
                Rea3.append(('', '', ''))
                rea3 = Rea3                            
        if len(rea0) == 0 and a0 != '':
            for i in range(len(a0)):
                Rea0.append(('', '', ''))                            
                rea0 = Rea0
        try:    
            if a0 != '':
                for a in range(len(a0)):
                    tup = a0[a]
                    for t in range(len(tup)):
                        index = numero_fattura + '_' + str(x) + '_' + str(t)
                        if int(tup[t][0][3:5]) == int(tup[t][1][3:5]):
                            raise ValueError
                        else:
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
            not_processed[POD] = 0 
            print '{} NOT PROCESSED'.format(POD)
    pp = len(not_processed.keys())/len(list_pod)
    print 'Percentage not processed: {}'.format(pp)
    #missed_pg = []
    if pp > 0:
        with open(prodotto, 'rb') as pdfFile:
            pdf = PyPDF2.PdfFileReader(pdfFile)
            for np in not_processed:
                for p in range(numpages):
                    #print p 
                    if np in pdf.getPage(p).extractText():
                        not_processed[np] = p+1
                        #missed_pg.append(p+1)
    
    Diz = pd.DataFrame.from_dict(diz, orient = 'index')          
    
    if Diz.shape[0] > 0:
        Diz.columns = [['data emissione','pod','lettura rilevata il','lettura rilevata il','Att-f0','Att-f1','Att-f2','Att-f3','ReAtt-f0',
                   'ReAtt-f1','ReAtt-f2','ReAtt-f3','potenza']]

        Diz.to_excel('C:/Users/d_floriello/fatture/'+numero_fattura+'_A2A.xlsx')        
    
    not_processed[''] = OrderedDict()
    NP = pd.DataFrame.from_dict(not_processed, orient = 'index')
    
    if NP.shape[0] > 0:                
        NP.columns = [['Pagina']]
        NP.to_excel('C:/Users/d_floriello/fatture/'+numero_fattura+'_manuale_A2A.xlsx')
    else:
        print 'Tutto finito'
    return 1
###############################################################################################################################
    

from os import listdir
from os.path import isfile, join

mypath = 'Z:/AREA BO&B/00000.File Distribuzione/1. UNARETI'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = onlyfiles[:len(onlyfiles)-1]

mypath2 = 'Z:/AREA BO&B/00000.File Distribuzione/1. UNARETI/'

prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/1. UNARETI/201790100016136_Dettaglio.pdf'
prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/1. UNARETI/201770100001052_Dettaglio.pdf'
A2A_Executer(prodotto)

for FILE in onlyfiles:
    A2A_Executer(mypath2+FILE)
    