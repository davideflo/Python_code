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
for d in rng:
    CF.append(flow[flow.columns[2]].ix[flow.index == d].mean())
    CN.append(flow[flow.columns[3]].ix[flow.index == d].mean())
    F.append(flow[flow.columns[4]].ix[flow.index == d].mean())
    
diz['CF'] = np.array(CF)
diz['CN'] = np.array(CN)
diz['F'] = np.array(F)
diz['anom'] = np.array(list([1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1]))
diz['pun'] = pun[pun.columns[0]].ix[0:22]

df= pd.DataFrame.from_dict(diz).set_index(rng)

df[df.columns[0:3]].plot()
plt.figure()
df[df.columns[-1]].plot()
