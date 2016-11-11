# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:23:45 2016

@author: d_floriello

PDF-Extractor
"""

import PyPDF2
import re

####################################################################################################
def Estrai_Linea_DATICONTATORE(tt, string, bVerbose = False):
    da = tt.find(string)
    inter = tt[da: da+56]
    en = ''
    end = 0
    for i in range(27, len(inter), 1):
        en += inter[i]
        end = i
        if inter[i] == ',':
            break
    en += inter[end+1:end+4]
    if bVerbose:
        data_inizio = inter[6:16]
        data_fine = inter[17:27]
        return data_inizio, data_fine, en
    else:
        return en
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
di = []
df = []
att0 = []
att1 = []
att2 = []
att3 = []
reatt0 = []
reatt1 = []
reatt2 = []
reatt3 = []
for x in range(len(pods) - 1):    
    capitolo = text[pods[x]:pods[x+1]]
    ixpod = capitolo.find('codice POD')
    POD = capitolo[ixpod+12:ixpod+26]
    list_pod.append(POD)
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f0', True)
    a1 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1', True)
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f2')
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f3')
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1')
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1')
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1')
    a0 = Estrai_Linea_DATICONTATORE(capitolo, 'Att-f1')
    