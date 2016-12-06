# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 10:22:48 2016

@author: utente

Construction of GAS historical Consumption
"""
from __future__ import division
import pandas as pd
import cx_Oracle 
import numpy as np
from collections import OrderedDict
import datetime

####################################################################################################
def FilterMisura(df, pdr, year, month):
    df2 = df.ix[df['pdr'] == pdr]
    df3 = df2.ix[df2['year'] == year]
    df4 = df3.ix[df3['month'] == month]
    if df4.shape[0] == 1:
        return df4['lettura']
    else:
        return 'unknown'
####################################################################################################


dns = cx_Oracle.makedsn('192.168.0.60', '1521', 'argon')
connection = cx_Oracle.connect('read_only', 'read_only', dns) 
cursor = connection.cursor()

SQLmisura = """SELECT D_LETTURA,
               S_PDR,
               CD_TP_LETTURA
               FROM MNH_LOGISTICA.T_LETTURE_GAS
               """

cursor.execute(SQLmisura)
data2 = cursor.fetchall()

ii = 0

dfm = pd.DataFrame()

for j in range(len(data2)):
    dfm = dfm.append(pd.DataFrame({'pdr':str(data2[j][1]), 'year':data2[j][0].year, 'month':data2[j][0].month, 'lettura':data2[j][2]}, index = [ii]))
    ii += 1

SQL181 = """SELECT ANNO_COMPETENZA, 
            CLIENTE,
            COD_CLIENTE,
            PDR,
            MATRICOLA_MISURATORE,
            MATRICOLA_CORRETTORE,
            DISTRIBUTORE,
            DATA_FATTURA,
            MESE_EMISS,
            MESE_COMP,
            CONSUMO_SMC,
            VENDITA_GAS_CLIENTI,
            CD_TP_MISURATORE,
            CD_PRODOTTO,
            REMI
            FROM MNH_COMMON.V_AUT_FATTURATO_GAS
            """
                       
cursor.execute(SQL181)
data = cursor.fetchall()


pdr = []
for i in range(len(data)):
    pdr.append(str(data[i][3]))
    
pdr = list(set(pdr))

df = pd.DataFrame()

si = 0
p = pdr[0]
df2 = pd.DataFrame()

for p in pdr:
    if p != 'None':
        loc = []
        anno = []
        for d in data:
            if p in d:
                loc.append(d)
                anno.append(d[0])
        cliente = loc[0][1]
        cd_cli = loc[0][2]
        ppdr = loc[0][3]
        mis = loc[0][4]
        corre = loc[0][5]
        distr = loc[0][6]
        for a in list(set(anno)):
            dia = OrderedDict()
            loc_anno = []
            for l in loc:
                if l[0] == a:                    
                    for m in range(1,13,1):                
                        if l[9] == m:
                            if m in dia.keys():
                                dia[m].append(l[10])
                            else:
                                dia[m] = [l[10]]
            for m in range(1,13,1):
                misura = FilterMisura(dfm, ppdr, a, m)
                if m in dia.keys():
                    cons = [x for x in dia[m] if x is not None]
                    ldf = pd.DataFrame({'cliente': cliente,'cod_cliente': cd_cli,'pdr': ppdr,'anno_competenza': a,
                                                'matricola_misuratore': mis,'matricola_correttore': corre,'distributore': distr,
                                                'mese_competenza': m, 'consumo_smc': np.sum(cons), 'tipo_misura': misura}, index = [si])
                else:
                    ldf = pd.DataFrame({'cliente': cliente,'cod_cliente': cd_cli,'pdr': ppdr,'anno_competenza': a,
                                                'matricola_misuratore': mis,'matricola_correttore': corre,'distributore': distr,
                                                'mese_competenza': m, 'consumo_smc': 0, 'tipo_misura': misura}, index = [si])
                si += 1
                df = df.append(ldf)

#####################################################################################################                
#def changeencode(data, cols):
#    import string
#    printable = set(string.printable)   
#    for col in cols:
#        for i in range(data.shape[0]):            
#            data[col].ix[i] = filter(lambda x: x in printable, data[col].ix[i])
#    return data   
#####################################################################################################

#df2 = changeencode(df, ['cliente', 'distributore'])

df.to_csv('storico_GAS.csv', sep = '|')
#df2.to_excel('storico_GAS.xlsx', encoding='utf-8')

