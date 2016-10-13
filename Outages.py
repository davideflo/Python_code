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

out = pd.read_excel('C:/Users/d_floriello/Documents/ourages_fran_2016-10-12.xlsx')

today = datetime.datetime(2016, 10, 13)

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
out2 = update_outages(out, today)    
 
varnames = ['ID Indisponibilité de production', "Type d'indisponibilité", 'Filière',
       "Type de l'unité de production",  'Nom du producteur',
       'Puissance nominale', 'Puissance disponible restante', 'Cause',
       'Statut'] 
 
S, P = get_statistics(out2, varnames[2], varnames[5])   