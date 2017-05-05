# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:27:06 2017

@author: d_floriello
"""

import pandas as pd
from collections import OrderedDict

#df = pd.read_excel("H:/Energy Management/12. Aggregatore/Aggregatore consumi orari/Mensili/DB_2016.xlsm", sheetname = "DB_SI_perd")
df = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_2016.xlsm", sheetname = "DB_SI_perd")


diz = OrderedDict()
for i in range(df.shape[0]):
    ll = []
    ll.append(df["Pod"].ix[i])
    ll.append(df["Area"].ix[i])
    ll.append(df["Giorno"].ix[i].to_pydatetime())
    sp = df[df.columns[10:35]].ix[i].values/df["PERDITA applicata"].ix[i]
    ll.extend(sp.tolist())
    diz[i] = ll
