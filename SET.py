# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:03:26 2017

@author: d_floriello

SET Extractor
"""

from __future__ import division
import pandas as pd
from collections import OrderedDict
import xlwt
import unidecode
#from tempfile import TemporaryFile

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
        df2 = df.ix[df[df.columns[2]].values.ravel() == string]
        if F == 'F1':
            res.append(str(df[df.columns[4]].dropna().tolist()[1]))
            res.append(str(df[df.columns[5]].dropna().tolist()[1]))
        val = 0
        if df2.shape[0] > 0:        
            val = df2[df2.columns[7]].tolist()[0]
            if isinstance(val, str):
                val = float(val.replace(',','.'))
            
                res.append(round(val,0))
            else:
                res.append(round(val,0))
        else:
            res.append('')
    string = 'quota variabile'
    df2 = df.ix[df[df.columns[2]].values.ravel() == string]
    #if res[2] == '':
    if df2.shape[0] > 0:
        res.append(round(df2[df2.columns[7]].sum(),0))   
    else:
        res.append('')
    return res
####################################################################################################
def ExtractReattiva_Set(df):
    Eff = ['F1', 'F2', 'F3']
    res = []
    for F in Eff:
        ai = Eff.index(F)
        string = 'cosfi 1^ fascia - ' + F
        string2 = 'cosfi 2^ fascia - ' + F
        place = -1
        place2 = -1
        for i in range(int(df[df.columns[2]].size)):
            if isinstance(df[df.columns[2]].values.tolist()[i], unicode):
                x = unidecode.unidecode(df[df.columns[2]].values.tolist()[i])
                if string in x:
                    place = i
                if string2 in x:
                    place2 = i
        if place == -1:
            res.append('')
        else:
            val = [df[df.columns[7]].values.tolist()[place]]
            if len(val) > 0:            
                if isinstance(val, str):
                    val = float(val[0].replace(',','.'))
                else:
                    val = val[0]
                if place2 != -1:
                    val2 = [df[df.columns[7]].values.tolist()[place2]]
                    if len(val2) > 0:            
                        if isinstance(val2, str):
                            val2 = float(val2[0].replace(',','.'))
                        else:
                            val2 = val2[0]
                    
                    attiva = ExtractAttiva_Set(df)
                    val = attiva[2+ai]*(0.33) + val + val2
                    
            res.append(round(float(val),0))
#            else:
#               res.append('')
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

def SET_Extractor(set1, name):
    if '.xls' in set1:
        set1 = pd.read_excel(set1)
    else:
        set1 = pd.read_table(set1, sep = ';')
    
    ix_pod = set1.ix[set1[set1.columns[0]] == 'POD'].index
    
    DE = str(set1[set1.columns[1]].ix[set1[set1.columns[0]] == 'Data allegato'].tolist()[0])[:10]
    
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
        al.append([DE])
        al.append([pod])
        try:
            al.append(ExtractAttiva_Set(capitolo))
            al.append(ExtractReattiva_Set(capitolo))
            al.append(ExtractPotenza_Set(capitolo))
            diz[pod] = [item for sublist in al for item in sublist]
        except:
            print 'Errore nel pod {}'.format(pod)
            missing.append(pod)
    
    print 'pod non processati {}'.format(len(missing))
    
    book = pd.DataFrame(missing)
    
    if len(missing) > 0:
        book.to_excel('C:/Users/d_floriello/fatture/SET_manuale_' + name + '.xlsx')
    
    DF = pd.DataFrame.from_dict(diz, orient = 'index')
    if DF.shape[0] > 0:    
        DF.columns = [['Numero fattura', 'data emissione', 'POD', 'data inizio', 'data fine', 'En Attiva F1', 'En Attiva F2', 'En Attiva F3',
                       'En Attiva F0', 'En Reattiva F1','En Reattiva F2','En Reattiva F3', 'Potenza']]
        
        DF = DF.reset_index(drop = True)
        
        DF.to_excel('C:/Users/d_floriello/fatture/fattura_SET_' + name + '.xlsx')
    
    return 1
####################################################################################################
    
    
from os import listdir
from os.path import isfile, join


#dirs = os.walk(mypath)
#dirs = [x[0] for x in os.walk(mypath)]
#
#dirs = [os.path.join(mypath,o) for o in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,o))]

#mypath2 = 'Z:/AREA BO&B/00000.File Distribuzione/3. SET DISTRIBUZIONE/Dettagli originali'
mypath2 = 'Z:/AREA BO&B/00000.File Distribuzione/3. SET DISTRIBUZIONE'


onlyfiles = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
#ff = [y for y in  if 'Unica' not in y]
ff2 = [y for y in onlyfiles if 'Thumbs' not in y]
ff2 = onlyfiles[2:4]
for f in ff2: 
    print mypath2+'/'+f
    SET_Extractor(mypath2+'/'+f, f[:-4])    
    
set1 = mypath2+'/'+ff2[0]   
name = ff2[0]