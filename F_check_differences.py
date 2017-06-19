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
import zipfile
import shutil

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
def ListingExtractor(s):
    E = [re.findall(r'"(.*?)"', s)][0]
    Es = [re.findall(r' (.*?)=', s)][0]
    if len(E) >0:
        if "E" not in Es[0]:
            Es = Es[1:]
            E = E[1:]
        OEs = ["E" + str(r) for r in range(1,97,1)]
        mis = np.repeat(0.0, 96)
        for n in range(len(Es)):
            mis[OEs.index(Es[n])] = float(E[n].replace(",", "."))
        return mis.tolist()
    else:
        return np.repeat(0.0, 96).tolist()
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
            #mis = ReMeasureExtractor(str(er))
            mis = ListingExtractor(str(er))
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
def OrdinatingExtractor(s):
    E = [re.findall(r'"(.*?)"', s)][0]
    Es = [re.findall(r' (.*?)=', s)][0]
    Es = [e[1:] for e in Es]     
    nEs = np.array([int(e) for e in Es], dtype = int)
    permuted = np.argsort(nEs)
    E = [E[p] for p in permuted]
    if len(E) > 96:
        E = E[1:]
    mis = list(map(Replacer, E))
    mis2 = list(map(float, mis))
    return mis2
###############################################################################
def PDOMeasureExtractor(s):
    E = [re.findall(r'"(.*?)"', s)][0]
    if len(E) > 96:
        E = E[1:]
    mis = list(map(Replacer, E))
    mis2 = list(map(float, mis))
    return mis2
###############################################################################
def PDOExtractor(infile, flusso):    
    cl = ['POD', 'day', 'date', 'zona', 'flusso']
    for h in range(24):
        cl.append(str(h) + '.A')
        cl.append(str(h) + '.B')
        cl.append(str(h) + '.C')
        cl.append(str(h) + '.D')
    count = 0
    dix = OrderedDict()
    pdox = Soup(open(infile).read(), "xml")
    bs = pdox.find_all('DatiPod')
    start_time = time.time()
    for b in bs:
        pod = b.find_all('Pod')
        zona = re.findall(r'>(.*?)<',str(b.find_all('PuntoDispacciamento')[0]))[0]
        M = b.find_all('MeseAnno')[:2]
        M = str(M)[11:18]
        y = int(M[3:])
        m = int(M[:2])
        Er = b.find_all('Ea')
        for er in Er:
            tbi = []
            mis = ListingExtractor(str(er))
            day = str(er)[(str(er).find(">")+1):(str(er).find(">")+3)]
            if sum(mis) <= 0 or day == '':
                pass
            #print(day)
#            mis = PDOMeasureExtractor(str(er))
#            mis = OrdinatingExtractor(str(er))
            else:
                
                tbi.append(str(pod[0])[5:19])
                tbi.append(day)
                tbi.append(datetime.date(y,m,int(day)))
                tbi.append(zona)
                tbi.append(flusso)
#                if 'AEM Cremona' in infile:
#                    mis = reversed(mis)
                tbi.extend(mis)
                dix[count] = tbi
                count += 1
    print("--- %s seconds ---" % (time.time() - start_time))
    dix = pd.DataFrame.from_dict(dix, orient = 'index')
    if dix.shape[0] > 0:
        dix.columns = cl
        return dix
    else:
        print('empty dataframe')        
###############################################################################
def singlePDOReducer(x):
    ret = [x[0], x[2], x[3], x[4]]
    vec = np.repeat(0.0, 24)
    for h in range(5,101,4):
        vec[list(range(5,101,4)).index(h)] = x[h] + x[h+1] + x[h+2] + x[h+3]
    ret.extend(vec.tolist())
    return ret
###############################################################################
def PDOReducer(df):
    DF = OrderedDict()
    diffs = OrderedDict()
    for i in range(df.shape[0]):
        pod = df['POD'].ix[i]
        dt = df['date'].ix[i]
        #flux = df['flusso'].ix[i]
        dfp = df.ix[df['POD'] == pod]
        dfpd = dfp.ix[dfp['date'] == dt]
        if dfpd.shape[0] == 1:
            DF[i] = singlePDOReducer(dfpd.values.ravel().tolist())
        elif dfpd.shape[0] > 1:
            Xrfo = dfpd.ix[dfpd['flusso'] == 'RFO']
            Xpdo = dfpd.ix[dfpd['flusso'] == 'PDO']
            if Xrfo.shape[0] > 0:
                ll = [pod, dt]
                ll.append(Xrfo[Xrfo.columns[5:]].diff().mean().mean())
                Xrfo = Xrfo.drop_duplicates(subset = ['POD', 'date', 'zona', 'flusso'], keep = 'last')
                ll.append(Xrfo['flusso'].values[0])
                DF[i] = singlePDOReducer(Xrfo.values.ravel().tolist())
                diffs[i] = ll
            elif Xrfo.shape[0] == 0 and Xpdo.shape[0] > 0:
                ll = [pod, dt]
                ll.append(Xrfo[Xrfo.columns[5:]].diff().mean().mean())
                Xpdo = Xpdo.drop_duplicates(subset = ['POD', 'date', 'zona', 'flusso'],keep = 'last')
                ll.append(Xrfo['flusso'].values[0])
                DF[i] = singlePDOReducer(Xpdo.values.ravel().tolist())
                diffs[i] = ll
            else:
                pass
        else:
            pass
    DF = pd.DataFrame.from_dict(DF, orient = 'index').reset_index()            
    diffs = pd.DataFrame.from_dict(diffs, orient = 'index').reset_index()            
    return DF, diffs
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

###############################################################################
######################## ALL PDO EXTRACTION ###################################
###############################################################################

DIR = "H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/"
subdir = os.listdir(DIR)
subdir = [x for x in subdir if not x.startswith("_")]
subdir = [x for x in subdir if not (x.endswith(".xlsx") or x.endswith(".zip") or x.endswith(".xls") or x.endswith(".docx") or x.endswith(".xml"))]

for s in subdir:
    if os.path.exists(DIR + s + "/2016"):
        subpath = os.listdir(DIR + s + "/2016")
        for sp in subpath:
            if 'storici' in sp.lower():
                print(DIR + s + "/2016/storici")
                pass
            elif 'rettifiche tardive' in sp.lower():
                print("RFO's found")
                ### is it a directory?
                if os.path.isdir(DIR + s + "/2016/" + sp):
                    files = os.listdir(DIR + s + "/2016/" + sp)
                    for f in files:
                        print(DIR + s + "/2016/" + sp + "/" + f)
                        if '.zip' in f and 'RFO' in f:
                            zf = zipfile.ZipFile(DIR + s + "/2016/" + sp + "/" + f)
                            lzf = zf.namelist()
                            zf.extractall(path = "H:/Energy Management/02. EDM/01. MISURE/ZIP")
                            for unz in lzf:
                                pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz)
                                shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/ZIP_DONE/" + unz)
                        elif '.xlm' in f and 'RFO' in f:
                            pdo = PDOExtractor(DIR + s + "/2016/" + sp + "/" + f, 'RFO')
                        elif "RNO" in f:
                            pass
            else:
                ### months
                print('MONTHS')
                if os.path.exists(DIR + s + "/2016/" + sp + "/PDO"):
                    print('PDO folder')
                    ### is it a directory?
                    if os.path.isdir(DIR + s + "/2016/" + sp + "/PDO"):
                        files = os.listdir(DIR + s + "/2016/" + sp + "/PDO")
                        print(DIR + s + "/2016/" + sp + "/PDO")
                        for f in files:
                            if '.zip' in f and 'PDO' in f:
                                zf = zipfile.ZipFile(DIR + s + "/2016/" + sp + "/PDO/" + f)
                                lzf = zf.namelist()
                                zf.extractall(path = "H:/Energy Management/02. EDM/01. MISURE/ZIP")
                                for unz in lzf:
                                    pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz)
                                    shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/ZIP_DONE/" + unz)
                            elif '.xlm' in f and 'PDO' in f:
                                pdo = PDOExtractor(DIR + s + "/2016/" + sp + "/PDO/" + f, 'PDO')
                else:
                    ### no PDO folder
                    print('no PDO folder')
                    ### is it a directory?
                    if os.path.isdir(DIR + s + "/2016/" + sp):
                        files = os.listdir(DIR + s + "/2016/" + sp)
                        print(DIR + s + "/2016/" + sp)
                        for f in files:
                            if '.zip' in f and 'PDO' in f:
                                zf = zipfile.ZipFile(DIR + s + "/2016/" + sp + "/" + f)
                                lzf = [x for x in zf.namelist() if ".zip" in x]
                                zf.extractall(path = "H:/Energy Management/02. EDM/01. MISURE/ZIP")
                                for unz in lzf:
                                    pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, 'PDO')
                                    shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/ZIP_DONE/" + unz)
                            elif '.xlm' in f and 'PDO' in f:
                                pdo = PDOExtractor(DIR + s + "/2016/" + sp + "/" + f)


#### os.walk returns a tuple --> look into the tuple
all_subdir = [x for x in os.walk(DIR)]
all_subdir = [x for x in all_subdir if ('2016' in x or '2017' in x[0])]
for sd in all_subdir:
    files = sd[2]
    for f in files:
        if 'PDO' in f or 'RFO' in f:
            shutil.copy2(sd[0] + "/" + f, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO")
            
            
###############################################################################
############################# EXTRACTION ######################################
###############################################################################
Mis = pd.DataFrame()

files = os.listdir("H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO")
for f in files:
    if ".zip" in f.lower():
        zf = zipfile.ZipFile("H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + f)
        zf.extractall(path = "H:/Energy Management/02. EDM/01. MISURE/ZIP")
        unzfiles = os.listdir("H:/Energy Management/02. EDM/01. MISURE/ZIP")
        for unz in unzfiles:
            if "PDO" in unz:
                pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, 'PDO')
                shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/ZIP_DONE/" + unz)
            elif "RFO" in unz:
                pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, 'RFO')
                shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/ZIP_DONE/" + unz)
            else:
                print("neither PDO, nor RFO")
    else:                                                    
        if "PDO" in f:
            pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + f, 'PDO')
        elif "RFO" in unz:
            pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + f, 'RFO')
        else:
            print("neither PDO, nor RFO")
    Mis = Mis.append(pdo, ignore_index = True)
    
Mis["date"] = pd.to_datetime(Mis["date"])    
Mis["date"].dt.year

Mis = Mis.ix[Mis["date"].dt.year > 2015]
    
Mis.to_hdf("C:/Users/d_floriello/Documents/PDO_RFO_estratti.h5", "PDO_RFO")
Mis.to_csv("PDO_RFO_estratti.csv")

infile = "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + f
infile = "H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz

Mis.apply(PDOReducer, axis = 1)

df = pd.read_hdf("C:/Users/d_floriello/Documents/PDO_RFO_estratti.h5")

df = df.drop_duplicates(keep = 'last')

df = df.reset_index()
df = df.drop('index', 1)

DF, diffs = PDOReducer(df)

DF.to_hdf("C:/Users/d_floriello/Documents/DB_misure.h5", "misure")