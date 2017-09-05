# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:45:09 2017

@author: d_floriello

Sbilanciamento 9 -- Module for models and datasets updates
"""


import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime
#import calendar
import re
import time
import os
from bs4 import BeautifulSoup as Soup
import zipfile
import shutil
import pickle

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
def SOSExtraction():
#### @BRIEF: function to extract all the SOS in the folders    
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
                
#    SOS.to_excel("SOS_elaborati.xlsx")
    
    
#    SOS = pd.read_excel("C:/users/d_floriello/Documents/SOS_elaborati.xlsx")
    
    sos = OrderedDict()
    for i in range(SOS.shape[0]):
        sos[i] = ReduceSOS(SOS.ix[i])
    sos = pd.DataFrame.from_dict(sos, orient = 'index')
    sos.columns = [['Pod', 'Giorno','1','2','3','4','5','6','7','8','9','10','11','12','13',
                        '14','15','16','17','18','19',
                        '20','21','22','23','24']]
    
    sos.to_excel("sos_elaborati_finiti.xlsx")
    sos.to_hdf("sos_elaborati_finiti.h5", "sos")
    
    return sos
###############################################################################
def PDOExtraction():
### @BRIEF: function to extract all the PDOs in the folder FOR THE FIRST TIME ONLY!!!
    missing = []
    all_subdir = []
    DIR = ["Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2015/",
           "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2016/",
           "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2017/"]
    for D in DIR:
        asd = [x for x in os.walk(D)]
        all_subdir.extend(asd) 
    
    for sd in all_subdir:
        files = sd[2]
        for f in files:
            if 'PDO' in f or 'RFO' in f:
                try:
                    shutil.copy2(sd[0] + "/" + f, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO")
                except:
                    missing.append(sd[0] + "/" + f)
    ## EXTRACTION ###
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
                    shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + unz)
                elif "RFO" in unz:
                    pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, 'RFO')
                    shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + unz)
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

    Mis["date"] = pd.to_datetime(Mis["date"]).to_pydatetime().date()
        
#    Mis.to_hdf("C:/Users/d_floriello/Documents/PDO_RFO_estratti.h5", "PDO_RFO")
#    Mis.to_csv("PDO_RFO_estratti.csv")
        
#    df = pd.read_hdf("C:/Users/d_floriello/Documents/PDO_RFO_estratti.h5")
    
    Mis = Mis.drop_duplicates(subset = ['POD','date', 'zona', 'flusso'], keep = 'last')
    
    Mis = Mis.reset_index()
    Mis.head()
    
    if Mis.columns[0] == 'index':
        Mis = Mis.drop('index', 1)
    
    DF, diffs = PDOReducer(Mis)

    if DF.columns[0] == 'index':        
        DF = DF.drop('index', 1)
    
    DF.columns = [['POD', 'Giorno', 'zona', 'flusso', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']]

    DF.to_hdf("C:/Users/d_floriello/Documents/DB_misure.h5", 'pdo')


    lf = os.listdir('H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO')
    with open('H:/Energy Management/02. EDM/01. MISURE/PDO_fatti.txt', 'wb') as fp:
        pickle.dump(lf, fp)                
        
    return DF
###############################################################################
def UpdatePDO():
### @BRIEF: function to update the PDO -- only the newly received PDOs will be processed    
    missing = []
    all_subdir = []
    DIR = ["Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2015/",
           "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2016/",
           "Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2017/"]

    with open('H:/Energy Management/02. EDM/01. MISURE/PDO_fatti.txt', 'rb') as fp:
        lf = pickle.load(fp)

    
    for D in DIR:
        asd = [x for x in os.walk(D)]
        all_subdir.extend(asd) 
    
    for sd in all_subdir:
        files = sd[2]
        for f in files:
            if 'PDO' in f or 'RFO' in f:
                if not f in lf:
                    try:
                        shutil.copy2(sd[0] + "/" + f, "H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO")
                    except:
                        missing.append(sd[0] + "/" + f)
            
        
    ## EXTRACTION ###
    Mis = pd.DataFrame()
    
    files = os.listdir("H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO")
    files = [x for x in files if not 'old' in x]
    for f in files:
        if ".zip" in f.lower():
            zf = zipfile.ZipFile("H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO/" + f)
            zf.extractall(path = "H:/Energy Management/02. EDM/01. MISURE/ZIP")
            unzfiles = os.listdir("H:/Energy Management/02. EDM/01. MISURE/ZIP")
            for unz in unzfiles:
                if "PDO" in unz:
                    pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, 'PDO')
                    shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + unz)
                elif "RFO" in unz:
                    pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, 'RFO')
                    shutil.move("H:/Energy Management/02. EDM/01. MISURE/ZIP/" + unz, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + unz)
                else:
                    print("neither PDO, nor RFO")
        else:                                                    
            if "PDO" in f:
                pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO/" + f, 'PDO')
                shutil.move("H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO/" + f, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + f)
            elif "RFO" in f:
                pdo = PDOExtractor("H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO/" + f, 'RFO')
                shutil.move("H:/Energy Management/02. EDM/01. MISURE/new_PDO_RFO/" + f, "H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO/" + f)
            else:
                print("neither PDO, nor RFO")
        Mis = Mis.append(pdo, ignore_index = True)
        
    Mis["date"] = pd.to_datetime(Mis["date"])#.dt.date()
        
#    Mis.to_hdf("C:/Users/d_floriello/Documents/PDO_RFO_estratti.h5", "PDO_RFO")
#    Mis.to_csv("PDO_RFO_estratti.csv")
        
#    df = pd.read_hdf("C:/Users/d_floriello/Documents/PDO_RFO_estratti.h5")
    
    Mis = Mis.drop_duplicates(subset = ['POD','date', 'zona', 'flusso'], keep = 'last')
    
    Mis = Mis.reset_index()
    Mis.head()
    
    if Mis.columns[0] == 'index':
        Mis = Mis.drop('index', 1)
    
    DF, diffs = PDOReducer(Mis)

    if DF.columns[0] == 'index':        
        DF = DF.drop('index', 1)
    
    DF.columns = [['POD', 'Giorno', 'zona', 'flusso', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']]

    PDO = pd.read_hdf("C:/Users/d_floriello/Documents/DB_misure.h5")
    PDO = PDO.append(DF, ignore_index = True)
    
    today = datetime.datetime.now().date()

    PDO.to_hdf('C:/Users/d_floriello/Documents/DB_misure' + str(today) + '.h5', 'pdo')
    
    lf = os.listdir('H:/Energy Management/02. EDM/01. MISURE/All_PDO_RFO')
    with open('H:/Energy Management/02. EDM/01. MISURE/PDO_fatti.txt', 'wb') as fp:
        pickle.dump(lf, fp)      

    return PDO
###############################################################################