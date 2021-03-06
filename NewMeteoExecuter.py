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
    from pyper import *
    
    count_fails = 0    
    done = False
    
    r = R("C://Program Files//R//R-3.3.3//bin//R")
    r.run("library(webshot)")
    r.run("today <- Sys.Date()")
    r.run("path_to_save <- paste0('C:/Users/utente/Documents/prova/', as.character(today),'.png')")
    r.run("webshot('http://www.terna.it/DesktopModules/GraficoTerna/GraficoTernaEsteso/ctlGraficoTerna.aspx?sLang=it-IT', path_to_save)")
    r.run("print('FATTO')")    
    
    today = datetime.datetime.today()
    
    print "Today's {}".format(today)
    
    try:    
        NewMeteoExtractor.Extractor()
        NewMeteoExtractor.ExtractorMoreDays()
    except:
        count_fails += 1
        while count_fails <= 3:
            NewMeteoExtractor.Extractor()
            NewMeteoExtractor.ExtractorMoreDays()
            done = True
            if done:
                break
            
    print 'Extraction Forward Meteo Done'
    
    if today.day == 28:
        print '                                    '
        print '####################################'
        print 'Update perdite and CRPP ARTIGIANALE'
        print '####################################'
        print '                                    '
        
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
    
    