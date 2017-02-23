# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:32:40 2017

@author: utente

New Meteo Extractor Executer
"""

import schedule
import NewMeteoExtractor
import datetime
import EE_interp


schedule.every().day.at("11:35").do(NewMeteoExtractor.Extractor)

NewMeteoExtractor.Extractor()

print 'Extraction Forward Meteo Done'

today = datetime.datetime.today()

if today.day == 1:
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print ' '
    print ' '
    print 'UPDATE THE METEO FILES OF THE CITIES'
    print ' '
    print ' '
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'
    print '#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@'

path = 'C:/Users/utente/Documents/PUN/curve_domanda'

schedule.every().day.at("12:00").do(EE_interp.Extract_Dataset, path)

EE_interp.Extract_Dataset(path)