"""
Created on Thu Sep 22 16:37:29 2016

@author: utente

Flow Analysis
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

flow = pd.read_excel('C:/Users/d_floriello/Documents/Flows.xlsx', sheetname = 'flow')
pun = pd.read_excel('C:/Users/d_floriello/Documents/Flows.xlsx', sheetname = 'pun', header = None)
flow = flow.set_index(flow['Date'])

rng = pd.date_range('2016-09-01', '2016-09-23', freq = 'D')

diz = OrderedDict()

CF = []
CN = []
F = []
#D = []
for d in rng:
    CF.append(flow[flow.columns[2]].ix[flow.index == d].mean())
    CN.append(flow[flow.columns[3]].ix[flow.index == d].mean())
    F.append(flow[flow.columns[4]].ix[flow.index == d].mean())
#    D.append(np.mean(flow[flow.columns[2]].ix[flow.index == d] - flow[flow.columns[4]].ix[flow.index == d]))
    
diz['CF'] = np.array(CF)
diz['CN'] = np.array(CN)
diz['F'] = np.array(F)
#diz['diff'] = np.array(D)
diz['anom'] = np.array(list([1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1]))
diz['pun'] = pun[pun.columns[0]].ix[0:22]

df= pd.DataFrame.from_dict(diz).set_index(rng)

df[df.columns[0:3]].plot()
plt.figure()
df[df.columns[-1]].plot()

################################################################
dem = pd.read_excel('C:/Users/d_floriello/Documents/Demand.xlsx')

rng = pd.date_range('2016-09-01', '2016-09-24', freq = 'D')

diz = OrderedDict()

sard = []
sici = []
sud = []
csud = []
cnor = []
nord = []
for d in rng:
    sard.append(dem['SARD'].ix[dem.index == d].mean())
    sici.append(dem['SICI'].ix[dem.index == d].mean())
    sud.append(dem['SUD'].ix[dem.index == d].mean())
    csud.append(dem['CSUD'].ix[dem.index == d].mean())
    cnor.append(dem['CNOR'].ix[dem.index == d].mean())
    nord.append(dem['NORD'].ix[dem.index == d].mean())

    
diz['SARD'] = np.array(sard)
diz['SICI'] = np.array(sici)
diz['SUD'] = np.array(sud)
diz['CSUD'] = np.array(csud)
diz['CNOR'] = np.array(cnor)
diz['NORD'] = np.array(nord)

de = pd.DataFrame.from_dict(diz).set_index(rng)
de.plot()