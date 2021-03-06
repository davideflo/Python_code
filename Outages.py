# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:29:07 2016

@author: d_floriello

Outages analysis
"""

import pandas as pd
import datetime
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

out = pd.read_excel('C:/Users/d_floriello/Documents/ourages_fran_2016-10-12.xlsx')

today = datetime.datetime(2016, 10, 24)

out = out.ix[out['Fin Indispo'] >= today]

###############################################################################
def update_outages(out, today):
    out = out.ix[out['Fin Indispo'] >= today]
    ids = np.unique(out['ID Indisponibilité de production'])
    ixl = []
    for ii in ids:
        given = out.ix[out['ID Indisponibilité de production'] == ii]
        ixl.append(max(given.index.tolist()))
    out = out[['ID Indisponibilité de production', "Type d'indisponibilité", 'Filière',
       "Type de l'unité de production",  'Nom du producteur',
       'Puissance nominale', 'Puissance disponible restante', 'Cause',
       'Statut']]
    return out.ix[ixl]
###############################################################################
def get_statistics(out, by, varname):
    diz1 = OrderedDict() 
    diz2 = OrderedDict()    
#    categorical = ["Type d'indisponibilité", 'Filière',
#       "Type de l'unité de production",  'Nom du producteur','Cause',
#       'Statut']
    grouped = out.groupby(by = by)
    names = list(set(out[by].tolist()))
#    if varname in categorical:
    for i in names:
        il1 = [] 
        il2 = []           
        for vn in list(set(out[varname].tolist())):
            il1.append(np.where(grouped.get_group(i)[varname] == vn)[0].size)
        il2 = [i/sum(il1) if sum(il1) > 0 else 0 for i in il1]
        diz1[i] = il1
        diz2[i] = il2
    df1 = pd.DataFrame.from_dict(diz1, orient = 'index')
    df2 = pd.DataFrame.from_dict(diz2, orient = 'index')
    df1.columns = [list(set(out[varname].tolist()))]
    df2.columns = [list(set(out[varname].tolist()))]
    
    return df1, df2
###############################################################################
def get_statistics_all(out):
    varnames = ["Type d'indisponibilité", 'Filière',
       "Type de l'unité de production",  'Nom du producteur',
       'Puissance nominale', 'Puissance disponible restante', 'Cause']
#    categorical = ["Type d'indisponibilité", 'Filière',
#       "Type de l'unité de production",  'Nom du producteur','Cause',
#       'Statut'] 
    for vn in varnames:
        remains = set(varnames).difference(vn)
        for rmn in remains:
            S, P = get_statistics(out, vn, rmn)
            print('Statistics of {} vs {}:'.format(vn, rmn))
            print(S)
            print(P)
            print('##################################')
###############################################################################
def get_simple_count(out, month, var):
    today = datetime.datetime(2016, month, 1)
    out2 = update_outages(out, today)
    print('num outages totali nel mese {}= {}'.format(month, out2.shape[0]))
    vn = set(list(out2[var]))
    cts = []
    for v in vn:
        print('percentage of {} outages = {}'.format(v, out2[var].ix[out2[var] == v].count()/out2.shape[0]))
        cts.append(out2[var].ix[out2[var] == v].count()/out2.shape[0])
    return cts
###############################################################################
def get_remaining_power(out, month):
    today = datetime.datetime(2016, month, 1)
    out2 = update_outages(out, today)
    out2 = out2.reset_index(drop = True)
    print('num outages totali nel mese {}= {}'.format(month, out2.shape[0]))
    remaining = perc = 0
    for r in range(out2.shape[0]):
        remaining += out2['Puissance nominale'].ix[r] - out2['Puissance disponible restante'].ix[r]
    perc = remaining/out2['Puissance nominale'].sum()
    return remaining, perc
###############################################################################
out2 = update_outages(out, today)    
 
varnames = ['ID Indisponibilité de production', "Type d'indisponibilité", 'Filière',
       "Type de l'unité de production",  'Nom du producteur',
       'Puissance nominale', 'Puissance disponible restante', 'Cause',
       'Statut'] 
 
S, P = get_statistics(out2, varnames[2], varnames[6])   
P.sum(axis = 1)

###############################################################################
##################################### 2015 ####################################
###############################################################################

out = pd.read_excel('C:/Users/d_floriello/Documents/out2015.xlsx')

nmn = ['num outages tot', 'tot indisp', '%tot indisp', 'altro_filiera', 'Hydraulique lacs',
 "Hydraulique fil de l'eau / éclusée",
 'Nucléaire',
 'Charbon',
 'Autre',
 'Gaz',
 'Marin',
 'Fioul',
 'Hydraulique STEP',
 'altro',
 'Indisponibilité planifiée', 'Indisponibilité fortuite']#,
# 'altro_prod',
# 'DIRECT ENERGIE',
# 'TOTAL',
# 'GDF-SUEZ',
# 'UNIPER',
# 'EDF',
# 'PSS POWER',
# 'ALPIQ']


diz = OrderedDict()
# il DataFrame risultante verra riempito da diz per righe.
# ogni entrata di diz sarà:
# 0: num outages totali;
# 1: indisponibilita totale in MWh(?)
# 2: indisponibilita inpercentuale
# 3: per ogni tipo di centrale la percentuale fuori uso di quel tipo sul totale outages
# 4: percentuale cause
# 5: percentuale produttori
mon = [1,2,3,4,5,6,7,8,9,10,11,12]
filiera = list(set(out['Filière']))
tipo = list(set(out["Type d'indisponibilité"]))
prod = list(set(out['Nom du producteur']))
cause = list(set(out['Cause']))
filiera = [i for i in filiera if isinstance(i, str)]
tipo = [i for i in tipo if isinstance(i, str)]
prod = [i for i in prod if isinstance(i, str)]
cause = [i for i in cause if isinstance(i, str)]
for m in mon:
    vec = []
    out2 = out.set_index(out['Fin Indispo'])
    atm = out2.ix[out2.index.month > m]
    ids = list(set(atm['ID Indisponibilité de production']))
    ind = tot = 0
    nfil = np.repeat(0, len(filiera)).tolist()
    ntip = np.repeat(0, len(tipo)).tolist()
    npro = np.repeat(0, len(prod)).tolist()
    ncau = np.repeat(0, len(cause)).tolist()
    if len(ids) > 0:
        vec.append([len(ids)]) ### num outages totali
        for i in ids:
#            print(ind)
#            print(type(ind))
#            print(tot)
#            print(type(tot))
            I = atm.ix[atm['ID Indisponibilité de production'] == i]
            M = I.ix[I['Version'] == np.max(I['Version'])]
            ind += M['Puissance nominale'].values[0] - M['Puissance disponible restante'].values[0]
            tot += M['Puissance nominale'].values[0]
#            print(ind)
#            print(type(ind))
#            print(tot)
#            print(type(tot))
            for fil in filiera:
                if fil == M['Filière'].values[0]:
                    indx = filiera.index(fil)
                    nfil[indx] += 1
            for tip in tipo:
                if tip == M["Type d'indisponibilité"].values[0]:
                   #print(M["Type d'indisponibilité"].values[0]) 
                   indx = tipo.index(tip)
                   ntip[indx] += 1
                   #print(ntip)
            for pro in prod:
                if pro == M['Nom du producteur'].values[0]:
                    indx = prod.index(pro)
                    npro[indx] += 1
            for cu in cause:
                if cu == M['Cause'].values[0]:
                    indx = cause.index(cu)
                    ncau[indx] += 1
        perc = ind/tot
        print(perc)
        vec.append([ind])
        vec.append([perc])
        print((np.array(ntip)/len(ids)).tolist())
        vec.append((np.array(nfil)/len(ids)).tolist())
        #vec.append((np.array(ntip)/len(ids)).tolist())
        vec.append([ntip[0]/len(ids)])
        vec.append([ntip[1]/len(ids)])
        #vec.append((np.array(npro)/len(ids)).tolist())
        #vec.append(ncau/ids.size)
        vec= [item for sublist in vec for item in sublist]
        #vec = np.array(vec).ravel().tolist()
        diz[str(m)] = vec
    else:
        diz[str(m)] = np.repeat(0,len(nmn)).tolist()        
        
        
DF = pd.DataFrame.from_dict(diz, orient = 'index')
DF.columns = [nmn]
DF.shape

fs = pd.read_excel('C:/Users/d_floriello/Documents/Prezzi Francia e Svizzera (2015 -2016).xlsx', sheetname = '2015')
fs = fs[fs.columns[[2,3]]].set_index(fs['Data'])
plt.figure()
plt.plot(fs[fs.columns[0]].resample('D').mean())
plt.figure()
DF[DF.columns[3:13]].plot()

from pandas.tools import plotting

plt.figure()
plotting.parallel_coordinates(DF[DF.columns[3:13]])

for i in range(1,13,1):
    print(sorted(DF[DF.columns[3:13]].ix[DF.index == str(i)]))

nuc = []
hydro = []
gas = []
coal = []
rim = []
perc = []
for i in range(1,13,1):
    nuc.append(get_simple_count(out, i, 'Filière')[3])
    hydro.append(get_simple_count(out, i, 'Filière')[2] + get_simple_count(out, i, 'Filière')[5])
    gas.append(get_simple_count(out, i, 'Filière')[1])
    coal.append(get_simple_count(out, i, 'Filière')[0])
    rim.append(get_remaining_power(out, i)[0])
    perc.append(get_remaining_power(out, i)[1])

diz2 = OrderedDict()
diz2['nucleare'] = nuc    
diz2['hydro'] = hydro    
diz2['gas'] = gas    
diz2['carbone'] = coal    

diz3 = OrderedDict()
diz3['remaining power'] = rim
diz3['%remaining power'] = perc

D2 = pd.DataFrame.from_dict(diz2)
D3 = pd.DataFrame.from_dict(diz3)
    
plt.figure()
plt.plot(np.array(nuc))
plt.figure()
plt.plot(np.array(fs[fs.columns[0]].resample('M').mean()))
plt.figure()
D3['%remaining power'].plot()

plt.figure()
plt.scatter(np.array(D3['remaining power']), np.array(fs[fs.columns[0]].resample('M').mean()))