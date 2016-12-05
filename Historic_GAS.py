# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 10:22:48 2016

@author: utente

Construction of GAS historical Consumption
"""
from __future__ import division
import pandas as pd
import cx_Oracle 
import datetime


dns = cx_Oracle.makedsn('192.168.0.60', '1521', 'argon')
connection = cx_Oracle.connect('read_only', 'read_only', dns) 
cursor = connection.cursor()

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
    pdr.append(data[i][3])
    
pdr = list(set(pdr))

df = pd.DataFrame()

si = 0
for p in pdr:
    if p is not None:
        loc = []
        anno = []
        for d in data:
            if p in d:
                loc.append(d)
                anno.append(d[0])
        for a in list(set(anno)):
            loc_anno = []
            for l in loc:
                if l[0] == a:
                    loc_anno.append(l)
            for la in loc_anno:
                loc_anno_mese = []
                for m in range(1,13,1):                
                    if la[9] == m:
                        loc_anno_mese.append(la)
                    consumo = 0
                    for lam in loc_anno_mese:
                        if lam[10] is not None:
                            consumo += lam[10]
                    ldf = pd.DataFrame({'cliente': la[1],'cod_cliente': la[2],'pdr': la[3],'anno_competenza': a,
                                        'matricola_misuratore': la[4],'matricola_correttore': la[5],'distributore': la[6],
                                        'mese_competenza': m, 'consumo_smc': consumo}, index = [si])
                    si += 1
                    df = df.append(ldf)
                