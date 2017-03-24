# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:32:40 2017

@author: utente

New Meteo Extractor Executer
"""

#import schedule

if __name__ == '__main__':
    
    import NewMeteoExtractor
    import datetime
    import EE_interp
    
    
    
    today = datetime.datetime.today()
    
    print "Today's {}".format(today)
    
        
    NewMeteoExtractor.Extractor()
        
    print 'Extraction Forward Meteo Done'
        
        
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
        
        
    EE_interp.Extract_Dataset(path)