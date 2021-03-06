# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:08:03 2017

@author: d_floriello

IRETI Executer
"""


from __future__ import division
#import PyPDF2
import re
from collections import OrderedDict
import pandas as pd
#import os
#from datetime import datetime
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import datetime
import calendar
import math
#from textract import process
#from tika import parser
#import pdfrw #import PdfReader, PdfWriter, PageMerge
#
#ipdf = pdfrw.PdfReader('C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf')
#text = parser.from_file('C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf')
#text = process('C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf')
#
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
            dtdi = datetime.datetime(int(di[6:]), int(di[3:5]), int(di[:2]))
            dtdf = datetime.datetime(int(df[6:]), int(df[3:5]), int(df[:2]))
            deltad = (dtdf - dtdi).days
            if dtdi.month == dtdf.month:
                res.append((di,df,enf))
#                    res.append(df)
#                    res.append(enf)
            else:
                for dr in range(dtdi.month, dtdf.month+1,1):
                    print dr
                    if dr == dtdi.month:
                        dff = datetime.date(dtdi.year,dtdi.month,calendar.monthrange(dtdi.year,dr)[1])
                        datafine1 = str(dff.day) +'/'+ str(dff.month) +'/'+ str(dff.year)
                        delta = (dff - dtdi).days
                        if not IsLastDay(dff):
#                            res.append(di)
#                            res.append(datafine1)
                            res.append((di, datafine1,math.ceil((enf/deltad)*delta)))
                            
                    elif dtdi.month < dr <dtdf.month:
                        dii = datetime.date(dtdi.year,dr,1)
                        dff = datetime.date(dtdi.year,dr,calendar.monthrange(dtdi.year,dr)[1])
                        datainizio2 = str(dii.day) +'/'+ str(dii.month) +'/'+ str(dii.year)
                        datafine1 = str(dff.day) +'/'+ str(dff.month) +'/'+ str(dff.year)
                        delta = calendar.monthrange(dff.year,dr)[1]
#                        res.append(datainizio2)
#                        res.append(datafine1)
                        res.append((datainizio2,datafine1,math.ceil((enf/deltad)*delta)))
                    else:
                        dii =  datetime.date(dtdf.year,dr,1)    
                        datainizio2 = str(dii.day) +'/'+ str(dii.month) +'/'+ str(dii.year)
                        delta = (dtdf - dii).days
#                        res.append(datainizio2)
#                        res.append(df)
                        res.append((datainizio2,df,math.ceil((enf/deltad)*delta)))
                                
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
#            if F == 'F1':
#                res.append(di)
#                res.append(df)
#                res.append(enf)
#            else:
#                res.append(enf)
            dtdi = datetime.datetime(int(di[6:]), int(di[3:5]), int(di[:2]))
            dtdf = datetime.datetime(int(df[6:]), int(df[3:5]), int(df[:2]))
            deltad = (dtdf - dtdi).days
            if dtdi.month == dtdf.month:
                res.append((di,df,enf))
#                    res.append(df)
#                    res.append(enf)
            else:
                for dr in range(dtdi.month, dtdf.month+1,1):
                    print dr
                    if dr == dtdi.month:
                        dff = datetime.date(dtdi.year,dtdi.month,calendar.monthrange(dtdi.year,dr)[1])
                        datafine1 = str(dff.day) +'/'+ str(dff.month) +'/'+ str(dff.year)
                        delta = (dff - dtdi).days
                        if not IsLastDay(dff):
#                            res.append(di)
#                            res.append(datafine1)
                            res.append((di, datafine1,math.ceil((enf/deltad)*delta)))
                            
                    elif dtdi.month < dr <dtdf.month:
                        dii = datetime.date(dtdi.year,dr,1)
                        dff = datetime.date(dtdi.year,dr,calendar.monthrange(dtdi.year,dr)[1])
                        datainizio2 = str(dii.day) +'/'+ str(dii.month) +'/'+ str(dii.year)
                        datafine1 = str(dff.day) +'/'+ str(dff.month) +'/'+ str(dff.year)
                        delta = calendar.monthrange(dff.year,dr)[1]
#                        res.append(datainizio2)
#                        res.append(datafine1)
                        res.append((datainizio2,datafine1,math.ceil((enf/deltad)*delta)))
                    else:
                        dii =  datetime.date(dtdf.year,dr,1)    
                        datainizio2 = str(dii.day) +'/'+ str(dii.month) +'/'+ str(dii.year)
                        delta = (dtdf - dii).days
#                        res.append(datainizio2)
#                        res.append(df)
                        res.append((datainizio2,df,math.ceil((enf/deltad)*delta)))
                                
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
        if 'ENERGIA REATTIVA' not in tts:
            print 'Not ENERGIA REATTIVA'
            res.append('')
        else:              
            en = ''
            for j in range(len(tts)-1, 0, -1):
                en += tts[j]
                if tts[j] == ',':
                    break
            en = en[::-1]                        
            en = en[4:]
            enf = float(en)
            minus = tts.find('-')
            di = tts[minus-10:minus]
            df = tts[minus+1:minus+11]
            dtdi = datetime.datetime(int(di[6:]), int(di[3:5]), int(di[:2]))
            dtdf = datetime.datetime(int(df[6:]), int(df[3:5]), int(df[:2]))
            deltad = (dtdf - dtdi).days
            if dtdi.month == dtdf.month:
                if F == 'F1':
                    res.append(di)
                    res.append(df)
                    res.append(enf)
                else:
                    res.append(enf)
            else:
                for dr in range(dtdi.month, dtdf.month+1,1):
                    print dr
                    if dr == dtdi.month:
                        dff = datetime.date(dtdi.year,dtdi.month,calendar.monthrange(dtdi.year,dr)[1])
    #                    datafine1 = str(dff.day) +'/'+ str(dff.month) +'/'+ str(dff.year)
                        delta = (dff - dtdi).days
                        res.append(math.ceil((enf/deltad)*delta))
                        
                    elif dtdi.month < dr <dtdf.month:
                        dii = datetime.date(dtdi.year,dr,1)
                        dff = datetime.date(dtdi.year,dr,calendar.monthrange(dtdi.year,dr)[1])
    #                    datainizio2 = str(dii.day) +'/'+ str(dii.month) +'/'+ str(dii.year)
    #                    datafine1 = str(dff.day) +'/'+ str(dff.month) +'/'+ str(dff.year)
                        delta = calendar.monthrange(dff.year,dr)[1]
                        res.append(math.ceil((enf/deltad)*delta))
                    else:
                        dii =  datetime.date(dtdf.year,dr,1)    
    #                    datainizio2 = str(dii.day) +'/'+ str(dii.month) +'/'+ str(dii.year)
                        delta = (dtdf - dii).days
                        res.append(math.ceil((enf/deltad)*delta))
                    
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

#prodotto = 'C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf'
#prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI/2727EM/Aem_STAMPABOLLETTE.2016120540_61150650_Allegato_24_1.001.pdf'
#prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI/2689EM/Aem_STAMPABOLLETTE.2016110560_61150650_Allegato_33_1.001.pdf'

def IRETI_Executer(prodotto):
    text = convert(prodotto)
    
    pods = [m.start() for m in re.finditer('Codice POD', text)]
    
    fatt_n = text[text.find('Fattura n.')+10:text.find('Fattura n.')+17]
    em = text.find('Emessa il')
    EM = text[em+9:em+26]
    
    
    list_pod = []
    not_proc = OrderedDict()
    diz = OrderedDict()
    for x in range(len(pods)):    
        print x
        if x < len(pods)-1:
            capitolo = text[pods[x]:pods[x+1]]
        else:
            capitolo = text[pods[len(pods)-1]:]
        codpod = capitolo[capitolo.find('Codice POD')+10:capitolo.find('Codice POD')+24]
        list_pod.append(codpod)
        index = codpod + '_' + str(x)
        try:
            ae = GetType(capitolo)
            ree = Reattiva_Extracter(capitolo)
            ree = [ree[i] for i in range(len(ree)) if not isinstance(ree[i], str)]
            pe = Potenza_Extractor(capitolo)
            al = [fatt_n, EM, codpod]
            if len(ae) <= 3:
                ae2 = []
                for t in range(len(ae)):
                    if t == 0:
                        ae2.extend([ae[t][0],ae[t][1],ae[t][2]])
                    else:
                        ae2.extend([ae[t][2]])
                al.extend(ae2)
                if len(ree) == 0:
                    ree = ['','','']
                al.extend(ree)
                al.extend([pe])
            else:
                #### CODICE PER TRATTARE + MESI
                print 'more months'
                mesi = []
                for t in range(len(ae)):
                    mesi.append(ae[t][0])
                mesi = list(set(mesi))

                for m in mesi:
                    if mesi.index(m) > 0:
                        ae3 = [ae[n][2] for n in range(len(ae)) if ae[n][0] == m]
                    else:
                        ae3 = [ae[0][0], ae[0][1]]
                        ae3.extend([ae[n][2] for m in mesi if ae[n][0] == m])
                    al.extend(ae3)
                    if ree != '':
                        al.extend(ree)
                    else:
                        al.extend([ree])
                    al.extend([pe])
            diz[index] = al
        except:
            not_proc[index] = codpod
            print '{} not processed'.format(codpod)
    
    if len(list_pod) > 0:        
        print 'percentage not processed: {}'.format(len(not_proc.keys())/len(list_pod))        
    else:
        print 'len(list_pod) is 0!'
    
    Diz = pd.DataFrame.from_dict(diz, orient = 'index')
    writer = pd.ExcelWriter('C:/Users/d_floriello/fatture/IRETI_'+fatt_n.replace('/','-')+'.xlsx', engine='xlsxwriter')
    if Diz.shape[0] > 0:
        Diz.columns = [['numero fattura', 'emessa il', 'POD', 'data inizio', 'data fine', 'attiva F1', 'attiva F2', 'attiva F3',
                    'reattiva F1', 'reattiva F2', 'reattiva F3', 'potenza']] 

        Diz.to_excel(writer, sheet_name = 'Sheet1')
        writer.save()
        
    else:
        print 'No POD processed'
    
    NP = pd.DataFrame.from_dict(not_proc, orient = 'index')
    
    if NP.shape[0] > 0:
        NP.columns = [['POD']]             
        writer2 = pd.ExcelWriter('C:/Users/d_floriello/fatture/' + fatt_n.replace('/','-')+'_IRETI_manuale.xlsx', engine='xlsxwriter')
        NP.to_excel(writer2)
        writer2.save()
    else:
        print 'Tutto finito'
    return 1
####################################################################################################

import os    
from os import listdir
from os.path import isfile, join

mypath = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

dirs = os.walk(mypath)
dirs = [x[0] for x in os.walk(mypath)]

dirs = [os.path.join(mypath,o) for o in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,o))]
dirs = dirs[1]


mypath2 = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI/'

for d in dirs:
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]
    ff = [y for y in onlyfiles if 'Unica' not in y]
    ff2 = [y for y in ff if 'Thumbs' not in y]
    for f in ff2: 
        print d+'/'+f
        IRETI_Executer(d+'/'+f)    
    
f = 'Aem_STAMPABOLLETTE.2017040550_61150650_Allegato_39_1.001 (1).pdf'    
prodotto = dirs + '/' + f
IRETI_Executer(prodotto)