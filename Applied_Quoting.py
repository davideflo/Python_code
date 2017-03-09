# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 09:05:42 2017

@author: d_floriello

Application of Quoting script
"""

import pandas as pd
import Quoting
import datetime


lpun = pd.read_excel('C:/Users/d_floriello/Desktop/longterm_pun.xlsx')
lpun = lpun.set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:8760])

Quoting.SimpleRedimensioniser(lpun, 40.50, datetime.datetime(2017, 04, 01), datetime.datetime(2017, 04, 30))
Quoting.SimpleRedimensioniser(lpun, 40.50, datetime.datetime(2017, 05, 01), datetime.datetime(2017, 05, 31))

Quoting.ConstrainedRedimensioniser(lpun, 41.80, {'Apr': 42, 'Mag':0, 'Giu':0})

lpun = Quoting.ConstrainedRedimensioniser(lpun, 40.50, {'Apr': 0})
lpun = Quoting.ConstrainedRedimensioniser(lpun, 40.30, {'Mag':0})
lpun = Quoting.ConstrainedRedimensioniser(lpun, 40.50, {'Apr': 40.50, 'Mag':40.30, 'Giu':0})


#lpun = ConstrainedRedimensioniser(lpun, 40.50, {'Apr': 0})
#lpun = ConstrainedRedimensioniser(lpun, 40.30, {'Mag':0})
#lpun = ConstrainedRedimensioniser(lpun, 40.50, {'Apr': 40.50, 'Mag':40.30, 'Giu':0})
#
#lpun = ConstrainedRedimensioniser(lpun, 45.85, {'Lug': 0, 'Ago':0, 'Set':0})
#lpun = ConstrainedRedimensioniser(lpun, 46.65, {'Ott': 0, 'Nov':0, 'Dic':0})


np = pd.read_excel('newpun.xlsx')
np = np.set_index(lpun.index)


np.plot()
np.resample('D').mean().plot()
np.resample('D').std().plot()
np.std()
np.mean()


old = pd.read_excel('dati_2014-2017.xlsx')
old.plot()
old.resample('D').std().plot()

old4 = old.ix[old.index.year == 2014]
old5 = old.ix[old.index.year == 2015]
old6 = old.ix[old.index.year == 2016]
feb6 = old6.ix[old6.index.month == 2]
leap = feb6.ix[feb6.index.day == 29].index
old6 = old6.drop(leap)


mold = (1/3)*(old4.values.ravel() + old5.values.ravel() + old6.values.ravel())
mold = pd.DataFrame(mold)
mold = mold.set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:8760])

mold.resample('D').std().plot()