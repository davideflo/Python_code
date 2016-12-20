# -*- coding: utf-8 -*-
"""

Lettore Dati Fatture Distributori 2.0 - AXOPOWER SPA

Created on Tue Nov 21 15:07:48 2016

@author: M.Protti - d_floriello
"""

from __future__ import division
import Tkinter, tkFileDialog, Tkconstants 
from Tkinter import * 
import os
import PyPDF2
import re
from collections import OrderedDict
import datetime
import calendar
import pandas as pd
import math
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

####################################################################################################
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
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = file(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 
####################################################################################################
def GetType(tt):
    tt2 = re.findall('(Fascia((?!Fascia).)*?Fascia)', tt, re.S)[0][0]
    if len([m.start() for m in re.finditer('CONSUMO REALE', tt2)]) > 1:
        return Attiva_Extracter(tt)
    else:
        return Attiva_Extracter2(tt)
####################################################################################################
def Attiva_Extracter(tt):
    Eff = ['F1', 'F2', 'F3']
    res = []
    if len([m.start() for m in re.finditer('ENERGIA ATTIVA', tt)]) > 3:
        raise ValueError
    for F in Eff:
        if F == 'F1':
            da = re.findall('(F1((?!F1).)*?F1)', tt, re.S)[0][0]
            da = re.sub('\F1$', '', da)
        elif F == 'F2':
            da = re.findall('(F2((?!F2).)*?F2)', tt, re.S)[0][0]
            da = re.sub('\F2$', '', da)
        elif F == 'F3':
            da = re.findall('(F3((?!F3).)*?F3)', tt, re.S)[0][0]
            da = re.sub('\F3$', '', da)
        tts = da.replace('.',',')
        if 'ENERGIA ATTIVA' in tts:
            bsl = [m.start() for m in re.finditer('/', tts)]
            dts = tts[(bsl[0]-2):(bsl[3]+5)]
            en = ''
            for j in range(len(tts)-1, 0, -1):
                en += tts[j]
                if tts[j] == ',':
                    break
            en = en[::-1]                        
            en = en[4:]
            enf = float(en)
            minus = dts.find('-')
            di = dts[:minus]
            df = dts[minus+1:]
            if F == 'F1':
                res.append(di)
                res.append(df)
                res.append(enf)
            else:
                res.append(enf)
    return res
####################################################################################################
def Attiva_Extracter2(tt):
    Eff = ['F1', 'F2', 'F3']
    res = []
    if len([m.start() for m in re.finditer('ENERGIA ATTIVA', tt)]) > 3:
        raise ValueError
    for F in Eff:
        if F == 'F1':
            da = [m.start() for m in re.finditer('F1', tt)]
            da2 = [m.start() for m in re.finditer('Fascia', tt)][1]
        elif F == 'F2':
            da = [m.start() for m in re.finditer('F2', tt)]
            da2 = [m.start() for m in re.finditer('Fascia', tt)][2]
        elif F == 'F3':
            da = [m.start() for m in re.finditer('F3', tt)]
            da2 = [m.start() for m in re.finditer('Quadro', tt)][0]
        tts = tt[da[0]:da2]
        if 'ENERGIA ATTIVA' in tts:
            bsl = [m.start() for m in re.finditer('/', tts)]
            dts = tts[(bsl[0]-2):(bsl[3]+5)]
            en = ''
            for j in range(len(tts)-1, 0, -1):
                en += tts[j]
                if tts[j] == ',':
                    break
            en = en[::-1]                        
            en = en[4:]
            enf = float(en)
            minus = dts.find('-')
            di = dts[:minus]
            df = dts[minus+1:]
            if F == 'F1':
                res.append(di)
                res.append(df)
                res.append(enf)
            else:
                res.append(enf)
    return res    
####################################################################################################
def Reattiva_Extracter(tt):
    Eff = ['F1', 'F2', 'F3']
    res = []
    for F in Eff:
        print F
        if F == 'F1':
            da = re.findall('(F1((?!F1).)*?Fascia)',tt,re.S)[0][0]
            da = re.sub('\ Fascia$', '', da)
            if len(da) == 0:
                da = re.findall('(F1((?!F1).)*?Quadro)',tt,re.S)[0][0]
                da = re.sub('\  Quadro$', '', da)
        elif F == 'F2:':
            da = re.findall('(F2((?!F2).)*?Fascia)',tt,re.S)[0][0]
            da = re.sub('\ Fascia$', '', da)
            if len(da) == 0:
                da = re.findall('(F2((?!F2).)*?Quadro)',tt,re.S)[0][0]
                da = re.sub('\  Quadro$', '', da)
        elif F == 'F3':
            da = re.findall('(F3((?!F3).)*?Quadro)',tt,re.S)[0][0]
            da = re.sub('\  Quadro$', '', da)
        tts = da.replace('.',',')
#        if 'ENERGIA REATTIVA' not in tts:
#            print 'Not ENERGIA REATTIVA'
#            raise ValueError                
        en = ''
        for j in range(len(tts)-1, 0, -1):
            en += tts[j]
            if tts[j] == ',':
                break
        en = en[::-1]                        
        en = en[4:]
        enf = float(en)
        res.append(enf)
    return res
####################################################################################################
def Potenza_Extractor(tt):
    Eff = ['F1', 'F2', 'F3']
    pot = []
    for F in Eff:
        points = [m.start() for m in re.finditer(F, tt)]
        for i in range(len(points)-1):
            da = tt[points[i]:points[i+1]]
            if 'POTENZA ELETTRICA' in da:
                tts = da.replace('.',',')
                en = ''
                for j in range(len(tts)-1, 0, -1):
                    en += tts[j]
                    if tts[j] == ',':
                        break
                en = en[::-1]                        
                en = en[4:]
                pot.append(float(en))
    if len(pot) > 0:
        return max(pot)
    else:
        return ''
####################################################################################################
def main(dire,nome):
    global prodotto,scedis
    text.insert(INSERT, "Fornitore: ",'bold') 
    text.insert(INSERT, scedis+"\n",'italic')
    text.insert(INSERT, "Directory: ",'bold') 
    text.insert(INSERT, dire+"\n",'italic')
    text.insert(INSERT, "File: ",'bold') 
    text.insert(INSERT, nome+"\n",'italic')
    prodotto = dire+nome
    if scedis == "Unareti":
        unareti(prodotto)
    elif scedis == "Ireti":
        ireti(prodotto)
    elif scedis == "SET":
        seti(prodotto)    
    text.insert(END, "Creato File Excel",'verdita')    
####################################################################################################             
def unareti(prodotto):
    text.insert(INSERT, "Procedura "+scedis+" iniziata\n",'verdita')    
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
    texto = 'INIZIO '
    for i in range(1, numpages, 1):
        pageObj = pdfReader.getPage(i)
        texto += pageObj.extractText()
    
    pods = [m.start() for m in re.finditer('codice POD:', texto)]
    
    
    list_pod = []
    not_processed = OrderedDict()
    diz = OrderedDict()
    for x in range(len(pods)):    
        print x
        if x < len(pods)-1:
            capitolo = texto[pods[x]:pods[x+1]]
        else:
            capitolo = texto[pods[x]:]
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
            text.insert(INSERT, "Non processata: ",'bold') 
            text.insert(INSERT, format(POD)+"\n",'italic')
            #print '{} NOT PROCESSED'.format(POD)
    pp = len(not_processed.keys())/len(list_pod)
    text.insert(INSERT, "POD processati: ",'bold') 
    text.insert(INSERT, str(x)+"\n",'italic')
    #text.insert(INSERT, "Percentuale non processata: ",'bold') 
    #text.insert(INSERT, format(round(pp*100,2))+"%\n",'italic')    
    text.insert(INSERT, "POD non automatizzati: ",'bold') 
    text.insert(INSERT, format(len(not_processed.keys()))+"\n",'italic')
    missed_pg = []
    if pp > 0:
        for np in not_processed:
            for p in range(numpages):
                if np in pdfReader.getPage(p).extractText():
                    missed_pg.append(p+1)
    
    Diz = pd.DataFrame.from_dict(diz, orient = 'index')          
    Diz.columns = [['data emissione','pod','lettura rilevata il','lettura rilevata il','Att-f0','Att-f1','Att-f2','Att-f3','ReAtt-f0',
               'ReAtt-f1','ReAtt-f2','ReAtt-f3','potenza']]
    Diz.to_excel(b+numero_fattura+'_A2A.xlsx')        

    NP = pd.DataFrame.from_dict(not_processed, orient = 'index')

    if NP.shape[0] > 0:                
        NP.columns = [['POD']]
        NP.to_excel(b+numero_fattura+'_manuale_A2A.xlsx')
    else:
        text.insert(INSERT, "Creato File Excel manuale\n",'rosso')   
####################################################################################################
def ireti(prodotto):
    text.insert(INSERT, "Procedura "+scedis+" iniziata\n",'verdita')
    texto = convert(prodotto)

    pods = [m.start() for m in re.finditer('Codice POD', texto)]

    fatt_n = texto[texto.find('Fattura n.')+10:texto.find('Fattura n.')+17]
    em = texto.find('Emessa il')
    EM = texto[em+9:em+25]
    
    
    list_pod = []
    not_proc = OrderedDict()
    diz = OrderedDict()
    for x in range(len(pods)):    
        print x
        if x < len(pods)-1:
            capitolo = texto[pods[x]:pods[x+1]]
        else:
            capitolo = texto[pods[len(pods)-1]:]
        codpod = capitolo[capitolo.find('Codice POD')+10:capitolo.find('Codice POD')+24]
        list_pod.append(codpod)
        index = codpod + '_' + str(x)
        try:
            ae = GetType(capitolo)
            ree = Reattiva_Extracter(capitolo)
            pe = Potenza_Extractor(capitolo)
            al = [fatt_n, codpod]
            al.extend(ae)
            al.extend(ree)
            al.extend([pe])
            diz[index] = al
        except:
            not_proc[index] = codpod
            #print '{} not processed'.format(codpod)
            
    text.insert(INSERT, "POD totali processati: ",'bold') 
    text.insert(INSERT, str(len(list_pod))+"\n",'italic')
    #print 'percentage not processed: {}%'.format(len(not_proc.keys())/len(list_pod)*100)
    #text.insert(INSERT, "Percentuale non automatizzata: ",'bold') 
    #text.insert(INSERT, format(round(len(not_proc.keys())/len(list_pod)*100,2))+"%\n",'italic')        
    text.insert(INSERT, "POD non automatizzata: ",'bold') 
    text.insert(INSERT, format(len(not_proc.keys()))+"%\n",'italic')
    #round(len(not_proc.keys())/len(list_pod),2)   
    
    Diz = pd.DataFrame.from_dict(diz, orient = 'index')
    Diz.columns = [['numero fattura', 'POD', 'data inizio', 'data fine', 'attiva F1', 'attiva F2', 'attiva F3',
                    'reattiva F1', 'reattiva F2', 'reattiva F3', 'potenza']] 
    Diz.to_excel(b+'IRETI_'+fatt_n.replace('/','-')+'.xlsx')
    
    NP = pd.DataFrame.from_dict(not_proc, orient = 'index')
    
    if NP.shape[0] > 0:
        NP.columns = [['POD']]             
        NP.to_excel(b+fatt_n.replace('/','-')+'_IRETI_manuale.xlsx')
        text.insert(INSERT, "Creato file Excel manuale\n",'rosso')
####################################################################################################             
def seti(prodotto):
    text.insert(INSERT, "Procedura "+scedis+" iniziata\n",'verdita')
    set1 = pd.read_table(prodotto, sep = ';')
    
    ix_pod = set1.ix[set1[set1.columns[0]] == 'POD'].index
    
    list_pod = []
    missing = []
    diz = OrderedDict()
    for x in range(len(ix_pod.tolist())):
        if x < len(ix_pod.tolist())-1:
            capitolo = set1.ix[ix_pod[x]:ix_pod[x+1]]
        else:
            capitolo = set1.ix[ix_pod[x]:]
        al = []
        pod = capitolo[capitolo.columns[1]].ix[ix_pod[x]]
        list_pod.append(pod)
        allegato = capitolo[capitolo.columns[1]].ix[ix_pod[x]+2]
        al.append([allegato])
        try:
            al.append(ExtractAttiva_Set(capitolo))
            al.append(ExtractReattiva_Set(capitolo))
            al.append(ExtractPotenza_Set(capitolo))
            diz[pod] = [item for sublist in al for item in sublist]
        except:            
            #print 'Errore nel pod {}'.format(pod)
            text.insert(INSERT, "Errore nel POD: ",'bold') 
            text.insert(INSERT, str(pod)+"\n",'italic')
            missing.append(pod)
            
    text.insert(INSERT, "POD totali processati: ",'bold') 
    text.insert(INSERT, str(len(list_pod))+"\n",'italic')        
    
    #print 'pod non processati {}'.format(len(missing))
    text.insert(INSERT, "POD non processati: ",'bold') 
    text.insert(INSERT, format(len(missing))+"\n",'italic')
    
    
    DF = pd.DataFrame.from_dict(diz, orient = 'index')
    DF.columns = [['Num allegato', 'data inizio', 'data fine', 'En Attiva F1', 'En Attiva F2', 'En Attiva F3',
                   'En Reattiva F1','En Reattiva F2','En Reattiva F3', 'Potenza']]
    
    DF.to_excel('fattura_SET.xlsx')            
        
####################################################################################################
def close_window (): 
    root.destroy()
####################################################################################################
def openDirectory():
    dirname = tkFileDialog.askdirectory(parent=root, initialdir='/home/', title='Selezionare la directory')
####################################################################################################    
def openFile():
    global b
    a = tkFileDialog.askopenfile(parent=root,initialdir='/home/',title='Selezionare il file', filetypes=[('document', '.pdf')])
    b = str(a.name)
    l = len(b)
    for x in range(l-1,0,-1):
        if b[x] == '/':
            b2 = b[x+1:l] 
            b = b[0:x+1]
            break
    main(b,b2)
####################################################################################################    
def openFile2():
    global b
    #a = tkFileDialog.askopenfile(parent=root,initialdir='/home/',title='Selezionare il file', filetypes=[('excel', '.xlsx')])
    a = tkFileDialog.askopenfile(parent=root,initialdir='/home/',title='Selezionare il file', filetypes=[('excel', '.csv')])    
    b = str(a.name)
    l = len(b)
    for x in range(l-1,0,-1):
        if b[x] == '/':
            b2 = b[x+1:l] 
            b = b[0:x+1]
            break
    main(b,b2)    
####################################################################################################    
def sel():
   global scedis
   pw = 'NB_PROTTI'
   if os.environ['COMPUTERNAME'] <> pw:
       root.destroy()
   if var.get() == 1:
       scedis = "Unareti"
   elif var.get() == 2:
       scedis = "Ireti"
   elif var.get() == 3:
       scedis = "SET" 
####################################################################################################       
def getDate(string):
    ii = string.find('-')
    return string[ii-4:ii+6]
####################################################################################################
def ExtractAttiva_Set(df):
    Eff = ['F1', 'F2', 'F3']
    res = []
    for F in Eff:
        string = 'quota variabile - ' + F
        df2 = df.ix[df[df.columns[2]] == string]
        if F == 'F1':
            res.append(getDate(str(df2[df2.columns[4]])))
            res.append(getDate(str(df2[df2.columns[5]])))
            res.append(round(df2[df2.columns[7]],0))
        else:
            res.append(round(df2[df2.columns[7]],0))
    return res
####################################################################################################
def ExtractReattiva_Set(df):
    Eff = ['F1', 'F2', 'F3']
    res = []
    for F in Eff:
        string = 'penalita cosfi 1^ fascia - ' + F
        df2 = df.ix[df[df.columns[2]] == string]
        if df2.size > 0:
            res.append(round(df2[df2.columns[7]].tolist()[0],0))
        else:
            res = ['','','']
    return res
####################################################################################################
def ExtractPotenza_Set(df):
    df2 = df.ix[df[df.columns[2]] == 'quota potenza']
    res = []
    if df2.size > 0:
        res.append(round(float(df2[df2.columns[7]].tolist()[0].replace(',','.')),0))
    else:
        res = ['']
    return res
####################################################################################################

            
root = Tk()
var = IntVar()
con = StringVar()
root.title('Lettore Dati Fatture Distributori 2.0 - AXOPOWER SPA')
label = Label( root, textvariable=con, relief=RAISED)
con.set("Scegli il Distributore")
label.pack()
#Options for buttons
button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}

R1 = Radiobutton(root, text="Unareti", variable=var, value=1,
                  command=sel)
R1.pack( anchor = W )

R2 = Radiobutton(root, text="Ireti", variable=var, value=2,
                  command=sel)
R2.pack( anchor = W )

R3 = Radiobutton(root, text="SET", variable=var, value=3,
                  command=sel)
R3.pack( anchor = W )
Button(root, text = 'Seleziona la directory', fg = 'black', command= openDirectory).pack(**button_opt)
Button(root, text = 'Seleziona il file PDF', fg = 'black', command= openFile).pack(**button_opt)       
Button(root, text = 'Seleziona il file Excel', fg = 'black', command= openFile2).pack(**button_opt)       
Button(root, text = "Fine", fg = 'black',  command = close_window).pack(**button_opt) 

text = Text(root, height=8, width=40)
text.pack()
text.tag_configure('italic', font=('Arial', 10, 'italic'))
text.tag_configure('bold', font=('Arial', 10, 'bold'))
text.tag_configure('verdita', font=('Arial', 10,'bold'), foreground="blue")
text.tag_configure('rosso', font=('Arial', 10,'bold'), foreground="red")
root.mainloop()

