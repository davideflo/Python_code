# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:46:06 2016

@author: d_floriello

PDF-Extractor IRETI
"""

from __future__ import division
#import PyPDF2
import re
import os
from datetime import datetime
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
def Attiva_Extracter(tt):
    Eff = ['F1', 'F2', 'F3']
    res = []
    for F in Eff:
        da = [m.start() for m in re.finditer(F, tt)]
        for i in range(len(da)):
            if i < len(da)-1:
                tts = tt[da[i]:da[i+1]]
            else:
                tts = tt[da[i]:]
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
                if F == 'F1' or F == 'FA':
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
        if F == 'F1':
            da = re.findall('(F1((?!F1).)*?Fascia)',tt,re.S)[0][0]
            if len(da) == 0:
                da = re.findall('(F1((?!F1).)*?Quadro)',tt,re.S)[0][0]
        elif F == 'F2:':
            da = re.findall('(F2((?!F2).)*?Fascia)',tt,re.S)[0][0]
            if len(da) == 0:
                da = re.findall('(F2((?!F2).)*?Quadro)',tt,re.S)[0][0]
        elif F == 'F3':
            da = re.findall('(F3((?!F3).)*?Quadro)',tt,re.S)[0][0]
#        else:
#            da = re.findall('(FA((?!FA).)*?Fascia)',tt,re.S)[0][0]
#            if len(da) == 0:
#                da = re.findall('(FA((?!FA).)*?Quadro)',tt,re.S)[0][0]
            tts = da            
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
    Eff = [F1', 'F2', 'F3']
    pot = 0
    for F in Eff:
        da = [m.start() for m in re.finditer(F, tt)]
        for i in range(len(da)):
            if i < len(da)-1:
                tts = tt[da[i]:da[i+1]]
            else:
                tts = tt[da[i]:]
            if 'POTENZA ELETTRICA' in tts:
                en = ''
                for j in range(len(tts)-1, 0, -1):
                    en += tts[j]
                    if tts[j] == ',':
                        break
                en = en[::-1]                        
                en = en[4:]
                pot += float(en)
    return pot
####################################################################################################

prodotto = 'C:/Users/d_floriello/Documents/Aem_STAMPABOLLETTE.2016080540_61150650_Allegato_21_1.001.pdf'

text = convert(prodotto)

pods = [m.start() for m in re.finditer('Codice POD', text)]



for x in range(len(pods) - 1):    
    print x
    print pods[x]
    capitolo = text[pods[x]:pods[x+1]]