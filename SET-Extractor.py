# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:17:02 2016

@author: d_floriello

SET - extractor
"""

from __future__ import division
import pandas as pd
from collections import OrderedDict

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

set1 = pd.read_excel('C:/Users/d_floriello/Documents/set.xlsx')
set1 = pd.read_table('Z:/AREA BO&B/00000.File Distribuzione/3. SET DISTRIBUZIONE/E1D05I_E1V171E-AXOPOWER SRL (SET) - DP1608-CL-01932800228_03728900964 (8).csv', sep = ';')

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

