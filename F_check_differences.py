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


###############################################################################
def GetData(pdos):
    diz = OrderedDict()
    for i in range(pdos.shape[0]):
        ll = []
        pod = pdos['Pod'].ix[i]
        dt = datetime.date(int(pdos['MeseAnno'].ix[i][3:]), int(pdos['MeseAnno'].ix[i][:2]), int(pdos['Ea'].ix[i]))
        zona = pdos['PuntoDispacciamento'].ix[i]
        ll.append(pod)
        ll.append(dt)  
        ll.append(zona)
        vec = np.repeat(0.0, 24)
        for h in range(24):
            vec[h] = float(pdos[str(h) + '.A'].ix[i].replace(',','.')) + float(pdos[str(h) + '.B'].ix[i].replace(',','.')) + float(pdos[str(h) + '.C'].ix[i].replace(',','.')) + float(pdos[str(h) + '.D'].ix[i].replace(',','.')) 
        ll.extend(vec.tolist())
        diz[i] = ll
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

cl = [x for x in pdos.columns[:8].tolist()]
for h in range(24):
    cl.append(str(h) + '.A')
    cl.append(str(h) + '.B')
    cl.append(str(h) + '.C')
    cl.append(str(h) + '.D')
pdos.columns = cl

pdos = pdos.fillna(value = '0,0')
Pdo = GetData(pdos)

Pdo = Pdo.ix[Pdo['DATA'] > datetime.date(2016,12,31)]
Pdo = Pdo.ix[Pdo['CodFlusso'] == 'PDO']

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

pd.tools.plotting.autocorrelation_plot(diff)


import xml.etree.ElementTree as ET
import os
import glob
path = 'Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2017/01. Invio di Gennaio/SOS/'

for infile in glob.glob( os.path.join(path, '*.xml') ):
        tree = ET.parse(infile)
        root = tree.getroot()
        with open(infile+'new.csv','w') as outfile:
            for elem in root.findall('.//event[@type="MEDIA"]'):
                    mediaidelem = elem.find('./mediaid')
                    if mediaidelem is not None:
                            outfile.write("{}\n".format(mediaidelem.text))

for infile in glob.glob( os.path.join(path, '*.xml') ):
    try:
        tree = ET.parse(infile)
        root = tree.getroot()
        print(root)
    except ET.ParseError as e:
        print(infile, str(e))
        continue
