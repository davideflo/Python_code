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
        vec[list(range(5,101,4)).index(h)] = float(x[h]) + float(x[h+1]) + float(x[h+2]) + float(x[h+3])
    ret.extend(vec.tolist())
    return ret
###############################################################################
def PDOReducer(df):
    DF = OrderedDict()
    diffs = OrderedDict()
    for i in range(df.shape[0]):
        if i % 100 == 0:
            print('avanzamento: {}'.format(i/df.shape[0]))        
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
                SHAPE = Xrfo.shape[0]
                ll.append(Xrfo[Xrfo.columns[5:]].diff().dropna().mean().mean())
                ll.append(SHAPE)
                Xrfo = Xrfo.drop_duplicates(subset = ['POD', 'date', 'zona', 'flusso'], keep = 'last')
                ll.append(Xrfo['flusso'].values[0])
                DF[i] = singlePDOReducer(Xrfo.values.ravel().tolist())
                diffs[i] = ll
            elif Xrfo.shape[0] == 0 and Xpdo.shape[0] > 0:
                ll = [pod, dt]
                SHAPE = Xrfo.shape[0]
                ll.append(Xrfo[Xrfo.columns[5:]].diff().dropna().mean().mean())
                ll.append(SHAPE)
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
anni = [2015, 2016, 2017]

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
sos.columns = [['Pod', 'Giorno','1','2','3','4','5','6','7','8','9','10','11','12','13',
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
missing = []
DIR = "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2015/"
DIR = "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2016/"
DIR = "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2017/"
all_subdir = [x for x in os.walk(DIR)]
#all_subdir = [x for x in all_subdir if ('2015' in x or '2016' in x or '2017' in x[0])]
for sd in all_subdir:
    files = sd[2]
    for f in files:
        if 'PDO' in f or 'RFO' in f:
            try:
                shutil.copy2(sd[0] + "/" + f, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO")
            except:
                missing.append(sd[0] + "/" + f)
###############################################################################
############################# EXTRACTION ######################################
###############################################################################
Mis = pd.DataFrame()

files = os.listdir("H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO")
files = [x for x in files if not 'old' in x]
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
        elif "RFO" in f:
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

df = df.drop_duplicates(subset = ['POD','date', 'zona', 'flusso'], keep = 'last')

df = df.reset_index()
df.head()
df = df.drop('index', 1)

DF, diffs = PDOReducer(df)

DF.to_hdf("C:/Users/d_floriello/Documents/DB_misure.h5", "misure")
DF.to_csv("C:/Users/d_floriello/Documents/DB_misure.csv")


pdo = pd.read_hdf("C:/Users/d_floriello/Documents/DB_misure.h5")
sos = pd.read_hdf("C:/Users/d_floriello/Documents/sos_elaborati_finiti.h5")
crpp = pd.read_excel("C:/Users/d_floriello/Documents/CRPP2017.xlsx")

pdo = pdo.drop('index', 1)
pdo.columns = [['POD', 'DATA', 'zona', 'flusso', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']]

import pickle
lf = os.listdir('H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO')
with open('H:/Energy Management/02. EDM/01. MISURE/PDO_fatti.txt', 'wb') as fp:
    pickle.dump(lf, fp)

with open('H:/Energy Management/02. EDM/01. MISURE/PDO_fatti.txt', 'rb') as fp:
    lf2 = pickle.load(fp)
###############################################################################
def MakeMOCRPP(pdo, sos, crpp, m, mprec):
    ### @PARAM: crpp is the 2017 one ad m is the target month for the CRPP
    ### m is a string
    ### DIFFERENCES are always PDO - SOS
    pdo['DATA'] = pd.to_datetime(pdo['DATA'])
    sos['DATA'] = pd.to_datetime(sos['DATA'])
    diz = OrderedDict()
    crpp_pods = list(set(crpp['POD'].values.ravel().tolist()))
    for p in crpp_pods:
        ixp = crpp['POD'].values.ravel().tolist().index(p)
        if crpp['Trattamento_' + m].ix[ixp] == 1:
            pdo_p = pdo.ix[pdo['POD'] == p]
            sos_p = sos.ix[sos['Pod'] == p]
            zona = pdo_p['zona'].values.ravel().tolist()[0]
            cons_sos_p = sos_p[sos.columns[2:]].sum().sum()
            cons_pdo_p = 0
            if datetime.date(2016, int(m),1) in pdo_p['DATA'].values.ravel().tolist():
                cons_pdo_p = pdo_p[pdo_p.columns[4:]].ix[pdo_p['DATA'] > datetime.date(2016, int(mprec), calendar.monthrange(2016, int(mprec))[1])].sum().sum()
            pdo_ = pdo_p.ix[pdo_p['DATA'].dt.month == int(m)]
            sos_ = sos_p.ix[sos_p['DATA'].dt.month == int(m)]
            pdo_prec = pdo_p.ix[pdo_p['DATA'].dt.month == int(mprec)]
            sos_prec = sos_p.ix[sos_p['DATA'].dt.month == int(mprec)]
            if sos_.shape[0] > 0:
                cons_sos_m = sos_[sos_.columns[2:]].sum().sum()
                if sos_prec.shape[0] > 0:
                    cons_sos_mprec = sos_prec[sos_prec.columns[2:]].sum().sum()
            if pdo_.shape[0] > 0:
                cons_pdo_m = pdo_[pdo_.columns[4:]].sum().sum()
                if pdo_prec.shape[0] > 0:
                    cons_pdo_mprec = pdo_prec[pdo_prec.columns[4:]].sum().sum()
            if sos_.shape[0] > 0 and pdo_.shape[0] > 0:
                DIFF_m = cons_pdo_m - cons_sos_m
                if cons_pdo_p > 0:
                    DIFF_tot = cons_pdo_p - cons_sos_p
                else:
                    DIFF_tot = float('nan')
            diz[p] = [p, zona, 1, cons_pdo_m, cons_pdo_mprec, cons_pdo_p,
                      cons_sos_m, cons_sos_mprec, cons_sos_p, DIFF_m, DIFF_tot]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['POD', 'zona', 'Trattamento_' + m, 'consumo_pdo_' + m , 'consumo_pdo_' + mprec, 'consumo_pdo_tot',
                    'consumo_sos_' + m , 'consumo_sos_' + mprec, 'consumo_sos_tot', 'diff_m_pdo_sos', 'diff_tot']]
    return diz
###############################################################################
def BuildCRPP(ddir):
    diz = OrderedDict()
    sos = pd.read_hdf("C:/Users/d_floriello/Documents/sos_elaborati_finiti.h5")
    sos.ix[sos['DATA'] > datetime.date(2016,5,31)]
    crpp = pd.read_excel("C:/Users/d_floriello/Documents/CRPP2017.xlsx")
    bud = "Z:/AREA BO&B/13. BUDGET/Budget2017/" + ddir 
    all_subdir = [x for x in os.walk(bud)]
    for a in all_subdir:
        print(a)
        for file in a[2]:
            print(a[0] + '/' + file)
            sn = pd.ExcelFile(a[0] + '/' + file, on_demand = True).sheet_names
            for s in sn:
                if 'PIVOT' in s or 'EMO' in s:
                    pass
                else:
                    print('opening sheet {}'.format(s))
                    df = pd.read_excel(a[0] + '/' + file, sheetname = s)
                    if df.shape[0] == 0 or 'Codice POD' not in df.columns:
                        pass
                    else:
                        cp = list(set(df['Codice POD'].values.ravel().tolist()))
                        for pod in cp:
                            zona = ''
                            if pod in crpp['POD'].values.ravel().tolist():
                                ixp = crpp['POD'].values.ravel().tolist().index(pod)
                                zona = crpp['ZONA'].ix[ixp]
                            dfp = df.ix[df['Codice POD'] == pod]
                            dfp = dfp.reset_index()
                            eng17 = np.repeat(0,12)
                            tratt17 = np.repeat(0,12)
                            for r in range(dfp.shape[0]):
                                tipo_misura = 0
                                if dfp['Tipo misuratore'].ix[r] == 'MIS_CURVA':
                                    tipo_misura = 1
                                elif dfp['Tipo misuratore'].ix[r] == 'TOT_FASCIA':
                                    tipo_misura = 3
                                else:
                                    tipo_misura = 2
                                #strm = str(int(dfp['Mese rif.'].ix[r])) if len(str(int(dfp['Mese rif.'].ix[r]))) > 1 else '0' + str(int(dfp['Mese rif.'].ix[r]))
                                eng17[(int(dfp['Mese rif.'].ix[r]) - 1)] = dfp['Energia totale'].ix[r]
                                tratt17[(int(dfp['Mese rif.'].ix[r]) - 1)] = tipo_misura
                            tot = [pod, zona]
                            tot.extend(eng17.tolist())
                            tot.extend(tratt17.tolist())
                            diz[pod] = tot
    SP = list(set(sos['Pod'].values.ravel().tolist()))
    for sp in SP:
        zona = ''
        if sp in crpp['POD'].values.ravel().tolist():
            ixp = crpp['POD'].values.ravel().tolist().index(sp)
            zona = crpp['ZONA'].ix[ixp]
        sos_p = sos.ix[sos['Pod'] == sp]
        sos_p = sos_p.set_index(pd.to_datetime(sos_p['DATA']))
        cons_sos_p = sos_p.resample('M').sum()
        if cons_sos_p.shape[0] == 12:
            eng17 = np.repeat(0,12)
            for i in cons_sos_p.index.tolist():
                m = i.month
                eng17[m-1] = cons_sos_p.ix[i].sum()
                tot = [sp, zona]
                tot.extend(eng17.tolist())
                tot.extend(np.repeat(1,12).tolist())
            diz[sp] = tot
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['pod', 'zona','Consumo_01','Consumo_02',
                    'Consumo_03','Consumo_04','Consumo_05','Consumo_06','Consumo_07',
                    'Consumo_08','Consumo_09','Consumo_10','Consumo_11','Consumo_12',
                    'Trattamento_01','Trattamento_02',
                    'Trattamento_03','Trattamento_04','Trattamento_05','Trattamento_06','Trattamento_07',
                    'Trattamento_08','Trattamento_09','Trattamento_10','Trattamento_11','Trattamento_12']]
    return diz
###############################################################################
def BuildCRPP2016():
    diz = OrderedDict()
    sos = pd.read_hdf("C:/Users/d_floriello/Documents/sos_elaborati_finiti.h5")
    crpp = pd.read_excel("C:/Users/d_floriello/Documents/CRPP2016.xlsx")
    bud = "Z:/AREA BO&B/13. BUDGET/Budget2016/Final"
    all_subdir = [x for x in os.walk(bud)]
    for a in all_subdir:
        for file in a[2]:
            print(a[0] + '/' + file)
            sn = pd.ExcelFile(a[0] + '/' + file, on_demand = True).sheet_names
            for s in sn:
                if 'PIVOT' in s or 'EMO' in s:
                    pass
                else:
                    df = pd.read_excel(a[0] + '/' + file, sheetname = s)
                    if df.shape[0] == 0 or 'Codice POD' not in df.columns:
                        pass
                    else:
                        cp = list(set(df['Codice POD'].values.ravel().tolist()))
                        for pod in cp:
                            zona = ''
                            if pod in crpp['POD'].values.ravel().tolist():
                                ixp = crpp['POD'].values.ravel().tolist().index(pod)
                                zona = crpp['ZONA'].ix[ixp]
                            dfp = df.ix[df['Codice POD'] == pod]
                            dfp = dfp.reset_index()
                            eng16 = np.repeat(0,12)
                            tratt16 = np.repeat(0,12)
                            for r in range(dfp.shape[0]):
                                tipo_misura = 0
                                if dfp['Tipo misuratore'].ix[r] == 'MIS_CURVA':
                                    tipo_misura = 1
                                elif dfp['Tipo misuratore'].ix[r] == 'TOT_FASCIA':
                                    tipo_misura = 3
                                else:
                                    tipo_misura = 2
                                #strm = str(int(dfp['Mese rif.'].ix[r])) if len(str(int(dfp['Mese rif.'].ix[r]))) > 1 else '0' + str(int(dfp['Mese rif.'].ix[r]))
                                eng16[(int(dfp['Mese rif.'].ix[r]) - 1)] = dfp['Energia totale'].ix[r]
                                if crpp.ix[crpp['POD'] == pod].shape[0] > 0:
                                    tratt16[(int(dfp['Mese rif.'].ix[r]) - 1)] = tipo_misura
                            tot = [pod, zona]
                            tot.extend(eng16.tolist())
                            tot.extend(tratt16.tolist())
                            diz[pod] = tot
    SP = list(set(sos['Pod'].values.ravel().tolist()))
    for sp in SP:
        zona = ''
        if sp in crpp['POD'].values.ravel().tolist():
            ixp = crpp['POD'].values.ravel().tolist().index(sp)
            zona = crpp['ZONA'].ix[ixp]
        sos_p = sos.ix[sos['Pod'] == sp]
        sos_p = sos_p.set_index(pd.to_datetime(sos_p['DATA']))
        cons_sos_p = sos_p.resample('M').sum()
        if cons_sos_p.shape[0] == 12:
            eng16 = np.repeat(0,12)
            for i in cons_sos_p.index.tolist():
                m = i.month
                eng16[m-1] = cons_sos_p.ix[i].sum()
                tot = [sp, zona]
                tot.extend(eng16.tolist())
                tot.extend(np.repeat(1,12).tolist())
            diz[sp] = tot
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['pod', 'zona','Consumo_01','Consumo_02',
                    'Consumo_03','Consumo_04','Consumo_05','Consumo_06','Consumo_07',
                    'Consumo_08','Consumo_09','Consumo_10','Consumo_11','Consumo_12',
                    'Trattamento_01','Trattamento_02',
                    'Trattamento_03','Trattamento_04','Trattamento_05','Trattamento_06','Trattamento_07',
                    'Trattamento_08','Trattamento_09','Trattamento_10','Trattamento_11','Trattamento_12']]
    return diz
###############################################################################
def convertToTS(df):
    diz = []
    dates = pd.to_datetime(df.index)
    for d in df.index:
        diz.extend(df.ix[d].values.ravel().tolist())
    diz = pd.DataFrame({'x': diz}).set_index(pd.date_range(dates[0], dates[-1].date() + datetime.timedelta(days = 19), freq = 'H')[:len(diz)])                     
    return diz
###############################################################################
                
diz.to_excel("CRPP_artigianale.xlsx")                

crpp2016 = BuildCRPP2016()
crpp2016.to_excel("CRPP2016_artigianale.xlsx")

Jan2017 = BuildCRPP("01. Gennaio/11-01-2017")
Jan2017.to_excel("CRPP_Jan_2017_artigianale.xlsx")
Feb2017 = BuildCRPP("02. Febbraio")
Feb2017.to_excel("CRPP_Feb_2017_artigianale.xlsx")
Mar2017 = BuildCRPP("03. Marzo")
Mar2017.to_excel("CRPP_Mar_2017_artigianale.xlsx")
Apr2017 = BuildCRPP("04. Aprile")
Apr2017.to_excel("CRPP_Apr_2017_artigianale.xlsx")
May2017 = BuildCRPP("05. Maggio")
May2017.to_excel("CRPP_May_2017_artigianale.xlsx")
Jun2017 = BuildCRPP("06. Giugno")
Jun2017.to_excel("CRPP_Jun_2017_artigianale.xlsx")
Jul2017 = BuildCRPP("07. Luglio")
Jul2017.to_excel("CRPP_Jul_2017_artigianale.xlsx")


sos = pd.read_hdf("C:/Users/d_floriello/Documents/sos_elaborati_finiti.h5")

sos2 = sos.groupby(['Pod', 'DATA']).agg(OrderedDict({'1':sum,'2':sum,'3':sum,'4':sum,'5':sum,
                                                     '6':sum,'7':sum,'8':sum,'9':sum,
                                                     '10':sum,'11':sum,'12':sum,'13':sum,
                                                     '14':sum,'15':sum,'16':sum,'17':sum,
                                                     '18':sum,'19':sum,'20':sum,'21':sum,
                                                     '22':sum,'23':sum,'24':sum}))

sos2 = sos2[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']]
                
                
agg = agg.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')                
agg = agg.set_index(pd.to_datetime(agg['Giorno']))

sa2a = agg[agg.columns[3:]].resample('D').sum()/1000

mdf2 = pd.DataFrame()
for e in enel:
    mdf2 = mdf2.append(mdf.ix[mdf['POD'] == e])

mdf2 = mdf2.set_index(pd.to_datetime(mdf2['Giorno']))
senel = mdf2[mdf2.columns[2:]].resample('D').sum()/1000


sa2a.plot()
senel.plot()

espod = agg[agg.columns[2:]].ix[agg['POD'] == 'IT012E00502501']
espod2 = agg[agg.columns[2:]].ix[agg['POD'] == 'IT012E00314756']


ts_a2a = convertToTS(sa2a)
ts_enel = convertToTS(senel)
ts_pod = convertToTS()

ts_a2a.plot()
ts_enel.plot()