# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:28:19 2017

@author: utente

Updating of meteo's trend

"""

import pandas as pd



mi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Milano.xlsx')
mi2017 = mi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
fi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Firenze.xlsx')
fi2017 = fi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ro2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Roma.xlsx')
ro2017 = ro2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ba2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Bari.xlsx')
ba2017 = ba2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
pa2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Palermo.xlsx')
pa2017 = pa2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ca2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Cagliari.xlsx')
ca2017 = ca2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]

mi2017['Tmedia'].plot()
fi2017['Tmedia'].plot()
ro2017['Tmedia'].plot()
ba2017['Tmedia'].plot()
pa2017['Tmedia'].plot()
ca2017['Tmedia'].plot()

TM = pd.DataFrame.from_dict({'NORD': mi2017['Tmedia'].values.ravel().tolist(),
                             'CNOR': fi2017['Tmedia'].values.ravel().tolist(),
                             'CSUD': ro2017['Tmedia'].values.ravel().tolist(),
                             'SUD': ba2017['Tmedia'].values.ravel().tolist(),
                             'SICI': pa2017['Tmedia'].values.ravel().tolist(),
                             'SARD': ca2017['Tmedia'].values.ravel().tolist()}, orient = 'columns')
                             
TM = TM.set_index(mi2017.index)

TM.plot()

TM.to_excel('Tmedie_per_zona.xlsx')