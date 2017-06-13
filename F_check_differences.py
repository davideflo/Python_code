# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:10:57 2017

@author: d_floriello

check differences Terna/PDOs
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime
import calendar
import re
import time
import os
from bs4 import BeautifulSoup as Soup

###############################################################################
def ReduceDF(x):
    X = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
    vec = np.repeat(0.0, 24)
    for h in range(8,104,4):
        vec[list(range(8,104,4)).index(h)] = (float(x[h].replace(',','.')) + float(x[h+1].replace(',','.')) + float(x[h+2].replace(',','.')) + float(x[h+3].replace(',','.')))
    X.extend(vec.tolist())
    return X
###############################################################################
def ToDF(pdos2):
    diz = OrderedDict()
    for i in range(pdos2.shape[0]):
        diz[i] = pdos2.ix[i]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['CodFlusso', 'Pod', 'MeseAnno', 'PuntoDispacciamento', 'Trattamento',
                    'Tensione', 'PotMax', 'Ea', '1','2','3','4','5','6','7','8','9','10','11','12','13',
                    '14','15','16','17','18','19',
                    '20','21','22','23','24']]
    return diz
###############################################################################
def GetData(pdos):
    diz = OrderedDict()
    pods = list(set(pdos['Pod'].values.ravel().tolist()))
    counter = 0
    for i in pods:
        print("avanzamento: {} %".format(pods.index(i)/len(pods)))
        ATP = pdos.ix[pdos['Pod'] == i]
        #ea = list(set(ATP['Ea'].values.ravel().tolist()))
        meseanno = list(set(ATP['MeseAnno'].values.ravel().tolist()))
        for ma in meseanno:
            ATP2 = ATP.ix[ATP['MeseAnno'] == ma]
            ea = range(1, calendar.monthrange(int(ma[3:]),  int(ma[:2]))[1])
            for e in ea:                
                ATP3 = ATP2.ix[ATP2['Ea'] == e]                
                if ATP3.shape[0] > 1:
                    ATPpdo = ATP3.ix[ATP3['CodFlusso'] == 'PDO']
                    ATPnpdo = ATP3.ix[ATP3['CodFlusso'] != 'PDO']
                    if ATPpdo.shape[0] > 0 and ATPnpdo.shape[0] > 0:
                        ATP4 = ATPnpdo.ix[max(ATPnpdo.index)]
                    elif ATPpdo.shape[0] > 0 and ATPnpdo.shape[0] == 0:
                        ATP4 = ATPpdo.ix[max(ATPpdo.index)]
                    elif ATPpdo.shape[0] == 0 and ATPnpdo.shape[0] > 0:
                        ATP4 = ATPnpdo.ix[max(ATPnpdo.index)]
                    else:
                        pass
                else:
                    ATP4 = ATP3
                if ATP4.shape[0] > 0:
                    ll = []
                    dt = datetime.date(int(ma[3:]), int(ma[:2]), int(e))
                    zona = ATP4['PuntoDispacciamento'] if isinstance(ATP4['PuntoDispacciamento'], str) else ATP4['PuntoDispacciamento'].values[0]
                    ll.append(i)
                    ll.append(dt)  
                    ll.append(zona)
                    vec = ATP4.values.ravel()[8:]
                    ll.extend(vec.tolist())
                    diz[counter] = ll
                    counter += 1
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['Pod', 'DATA','zona','1','2','3','4','5','6','7','8','9','10','11','12','13',
                    '14','15','16','17','18','19',
                    '20','21','22','23','24']]
    return diz
###############################################################################
def ToTS(Pdo,zona):
    Pdoz = Pdo.ix[Pdo['zona'] == zona]
    dr = pd.date_range('2017-01-01', '2017-04-01', freq = 'H')[:2160]
    diz = OrderedDict()
    for d in dr:
        dd = d.date()
        sum_at_day = Pdoz.ix[Pdoz['DATA'] == dd].sum()
        h = d.hour
        diz[d] = sum_at_day.ix[h]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    return diz
###############################################################################
def convertDates(vec):
   CD = vec.apply(lambda x: datetime.datetime(year = int(str(x)[6:10]), month = int(str(x)[3:5]), day = int(str(x)[:2]), hour = int(str(x)[11:13])))
   return CD
###############################################################################


terna = pd.read_excel("C:/users/d_floriello/Documents/aggregato_sbilanciamento2.xlsx")
pdos = pd.read_excel("C:/users/d_floriello/Desktop/XML_Importer.xlsm")

pdos.to_hdf("C:/users/d_floriello/Desktop/pdos.h5", "PDO")
pdos = pd.read_hdf("C:/users/d_floriello/Desktop/pdos.h5")

cl = [x for x in pdos.columns[:8].tolist()]
for h in range(24):
    cl.append(str(h) + '.A')
    cl.append(str(h) + '.B')
    cl.append(str(h) + '.C')
    cl.append(str(h) + '.D')
pdos.columns = cl

pdos = pdos.fillna(value = '0,0')

pdos2 = pdos.apply(ReduceDF, axis = 1)
pdos3 = ToDF(pdos2)
Pdo = GetData(pdos3)

Pdo = Pdo.ix[Pdo['DATA'] > datetime.date(2016,12,31)]



nord = ToTS(Pdo, "NORD")
nord = nord.apply(lambda x: x/1000)
nord.plot()


tnord = terna.ix[terna['CODICE RUC'] == 'UC_DP1608_NORD']
tnord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:tnord.shape[0]]
cd = convertDates(tnord['DATA RIFERIMENTO CORRISPETTIVO'])
tnord = tnord.set_index(cd.values)

tnord['MO [MWh]'].plot()

tnord = tnord.ix[tnord.index.date > datetime.date(2016,12,31)]
tnord = tnord.ix[tnord.index.date < datetime.date(2017,4,1)]

diz = OrderedDict()
dr = pd.date_range('2017-01-01', '2017-04-01', freq = 'H')[:2160]
for i in dr:
    if i in tnord.index:
        if tnord.ix[i].shape[0] > 1:
            y = tnord['MO [MWh]'].ix[i].sum()
        elif tnord.ix[i].shape[0] == 0:
            y = 0
        else:
            y = tnord['MO [MWh]'].ix[i]
    else:
        y = 0
    diz[i] = [y]

Tnord = pd.DataFrame.from_dict(diz, orient = 'index')

Tnord.plot(title = 'NORD - TERNA')

diff = nord - Tnord
diff.plot(color = 'green', title = 'differenza PDO-Terna')

diff.mean()
diff.std()
diff.median()



###############################################################################
##################### SOS EXTRACTION ##########################################
###############################################################################




###############################################################################
def Replacer(s):
    return s.replace(",",".")
###############################################################################    
def ReMeasureExtractor(s):
    E = [re.findall(r'"(.*?)"', s)][0]
    E = E[1:]
    mis = list(map(Replacer, E))
    mis2 = list(map(float, mis))
    return mis2
###############################################################################    
def SOSExtractor(infile):    
    count = 0
    dix = OrderedDict()
    pdox = Soup(open(infile).read(), "xml")
    bs = pdox.find_all('DatiPod')
    start_time = time.time()
    for b in bs:
        pod = b.find_all('Pod')
        M = b.find_all('MeseAnno')[:2]
        M = str(M)[11:18]
        y = int(M[3:])
        m = int(M[:2])
        Er = b.find_all('Ea')
        for er in Er:
            tbi = []
            day = str(er)[(str(er).find(">")+1):(str(er).find(">")+3)] 
            #print(day)
            mis = ReMeasureExtractor(str(er))
            if len(mis) < 96:
                pass
            else:
                tbi.append(str(pod[0])[5:19])
                tbi.append(day)
                tbi.append(datetime.date(y,m,int(day)))
                tbi.extend(mis)
                dix[count] = tbi
                count += 1
    print("--- %s seconds ---" % (time.time() - start_time))
    dix = pd.DataFrame.from_dict(dix, orient = 'index')
    return dix
###############################################################################
def ReduceSOS(x):
    X = [x[0],x[2]]
    vec = np.repeat(0.0, 24)
    for h in range(3,99,4):
        vec[list(range(3,99,4)).index(h)] = x[h] + x[h+1] + x[h+2] + x[h+3]
    X.extend(vec.tolist())
    return X
###############################################################################


mesi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno', 'Luglio', 
        'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
anni = [2016, 2017]

SOS = pd.DataFrame()
for a in anni:
    for mm in mesi:      
        im = mesi.index(mm) + 1
        sim = str(im) if len(str(im)) > 1 else "0" + str(im)
        path = 'Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/' + str(a) + '/' + sim + '. Invio di ' + mm + '/SOS/'
        print(path)
        if os.path.exists(path):
            files = os.listdir(path)
            for infile in files:
                sos = SOSExtractor(path + infile)
                SOS = SOS.append(sos, ignore_index = True)
            
SOS.to_excel("SOS_elaborati.xlsx")


SOS = pd.read_excel("C:/users/d_floriello/Documents/SOS_elaborati.xlsx")

sos = OrderedDict()
for i in range(SOS.shape[0]):
    sos[i] = ReduceSOS(SOS.ix[i])
sos = pd.DataFrame.from_dict(sos, orient = 'index')
sos.columns = [['Pod', 'DATA','1','2','3','4','5','6','7','8','9','10','11','12','13',
                    '14','15','16','17','18','19',
                    '20','21','22','23','24']]

sos.to_excel("sos_elaborati_finiti.xlsx")
sos.to_hdf("sos_elaborati_finiti.h5", "sos")

