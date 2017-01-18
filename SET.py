# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:03:26 2017

@author: d_floriello

SET Extractor
"""

from __future__ import division
import pandas as pd
from collections import OrderedDict

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
        val = df2[df2.columns[7]].tolist()[0]
        if isinstance(val, str):
            val = float(val.replace(',','.'))
        if F == 'F1':
            res.append(str(df2[df2.columns[4]].tolist()[0]))
            res.append(str(df2[df2.columns[5]].tolist()[0]))
            res.append(round(val,0))
        else:
            res.append(round(val,0))
    return res
####################################################################################################
def ExtractReattiva_Set(df):
    Eff = ['F1', 'F2', 'F3']
    res = []
    for F in Eff:
        string = 'cosfi 1^ fascia - ' + F
        place = -1
        for i in range(int(df[df.columns[2]].size)):
            if string in str(df[df.columns[2]].tolist()[i]):
                place = i
        if place == -1:
            res.append('')
        else:
            df2 = df.ix[place]
            val = df2.ix[7]
            if isinstance(val, str):
                val = float(val.replace(',','.'))
            if df2.size > 0:
                res.append(round(val,0))
#            else:
#                res = ['','','']
    return res
####################################################################################################
def ExtractPotenza_Set(df):
    df2 = df.ix[df[df.columns[2]] == 'quota potenza']
    res = []
    if df2.size > 0:
        if isinstance(df2[df2.columns[7]].tolist()[0], str):
            res.append(round(float(df2[df2.columns[7]].tolist()[0].replace(',','.')),0))
        else:
            res.append(round(float(df2[df2.columns[7]].tolist()[0]),0))
    else:
        res = ['']
    return res
####################################################################################################

#set1 = pd.read_excel('C:/Users/d_floriello/Documents/set.xlsx')
#set1 = pd.read_table('Z:/AREA BO&B/00000.File Distribuzione/3. SET DISTRIBUZIONE/E1D05I_E1V171E-AXOPOWER SRL (SET) - DP1608-CL-01932800228_03728900964 (8).csv', sep = ';')

def SET_Extractor(set1):
    if '.xls' in set1:
        set1 = pd.read_excel(set1)
    else:
        set1 = pd.read_table(set1, sep = ';')
    
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
            print 'Errore nel pod {}'.format(pod)
            missing.append(pod)
    
    print 'pod non processati {}'.format(len(missing))
    
    
    
    DF = pd.DataFrame.from_dict(diz, orient = 'index')
    DF.columns = [['Num allegato', 'data inizio', 'data fine', 'En Attiva F1', 'En Attiva F2', 'En Attiva F3',
                   'En Reattiva F1','En Reattiva F2','En Reattiva F3', 'Potenza']]
    
    DF.to_excel('fattura_SET.xlsx')
    return 1
####################################################################################################
    
    
from os import listdir
from os.path import isfile, join

mypath = 'Z:/AREA BO&B/00000.File Distribuzione/3. SET DISTRIBUZIONE'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#dirs = os.walk(mypath)
#dirs = [x[0] for x in os.walk(mypath)]
#
#dirs = [os.path.join(mypath,o) for o in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,o))]

mypath2 = 'Z:/AREA BO&B/00000.File Distribuzione/3. SET DISTRIBUZIONE/'


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#ff = [y for y in  if 'Unica' not in y]
ff2 = [y for y in onlyfiles if 'Thumbs' not in y]
for f in ff2: 
    print mypath2+'/'+f
    SET_Extractor(mypath2+'/'+f)    
    
set1 = mypath2+'/'+f   