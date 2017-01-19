# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:17:37 2017

@author: d_floriello

IRETI funzionante ma senza split dei mesi
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
#from textract import process
#from tika import parser
#import pdfrw #import PdfReader, PdfWriter, PageMerge
#
#ipdf = pdfrw.PdfReader('C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf')
#text = parser.from_file('C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf')
#text = process('C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf')
#

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

#prodotto = 'C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf'
#prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI/2727EM/Aem_STAMPABOLLETTE.2016120540_61150650_Allegato_24_1.001.pdf'
#prodotto = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI/2689EM/Aem_STAMPABOLLETTE.2016110560_61150650_Allegato_33_1.001.pdf'

def IRETI_Executer(prodotto):
    text = convert(prodotto)
    
    pods = [m.start() for m in re.finditer('Codice POD', text)]
    
    fatt_n = text[text.find('Fattura n.')+10:text.find('Fattura n.')+17]
    em = text.find('Emessa il')
    EM = text[em+9:em+25]
    
    
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
            if int(ae[0][3:5]) == int(ae[1][3:5]):
                raise ValueError
            ree = Reattiva_Extracter(capitolo)
            pe = Potenza_Extractor(capitolo)
            al = [fatt_n, EM, codpod]
            al.extend(ae)
            al.extend(ree)
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
    if Diz.shape[0] > 0:
        Diz.columns = [['numero fattura', 'emessa il', 'POD', 'data inizio', 'data fine', 'attiva F1', 'attiva F2', 'attiva F3',
                    'reattiva F1', 'reattiva F2', 'reattiva F3', 'potenza']] 

        Diz.to_excel('fatture/IRETI_'+fatt_n.replace('/','-')+'.xlsx')

    else:
        print 'No POD processed'
    
    NP = pd.DataFrame.from_dict(not_proc, orient = 'index')
    
    if NP.shape[0] > 0:
        NP.columns = [['POD']]             
        NP.to_excel('fatture/' + fatt_n.replace('/','-')+'_IRETI_manuale.xlsx')
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

mypath2 = 'Z:/AREA BO&B/00000.File Distribuzione/2. IRETI/'

for d in dirs:
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]
    ff = [y for y in onlyfiles if 'Unica' not in y]
    ff2 = [y for y in ff if 'Thumbs' not in y]
    for f in ff2: 
        print d+'/'+f
        IRETI_Executer(d+'/'+f)    
    
    
prodotto = d + '/' + f
IRETI_Executer(prodotto)