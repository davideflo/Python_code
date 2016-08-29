# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:43:32 2016

@author: utente

test for one single complete ROLLING dataset
"""

import pandas as pd
import numpy as np
import rolling
import time

####################################################################################
def Launcher():
    data = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2010.xlsx", sheetname=1)
    data2 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2011.xlsx", sheetname=0)
    data3 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2012.xlsx", sheetname=0)
    data4 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2013.xlsx", sheetname=0)
    data5 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2014.xlsx", sheetname=0)
    data6 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2015.xlsx", sheetname=0)
    
    meteo = pd.read_table("C:/Users/utente/Documents/PUN/storico_roma.txt", sep="\t")
    
#    varn = "PUN"
    
#    pun1 = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
#                                   np.array(data3[varn]),
#                             np.array(data4[varn]), np.array(data5[varn]), 
#                                np.array(data6[varn])])
    
    cols = set(data.columns).intersection(set(data2.columns))
    cols = set(cols).intersection(set(data3.columns))
    cols = set(cols).intersection(set(data4.columns))
    cols = set(cols).intersection(set(data5.columns))   
    cols = set(cols).intersection(set(data6.columns))
    
    cols = list(cols)
    
    hour = cols.index('Ora\nHour')
    dt = cols.index('   Data/Date\n(YYYYMMDD)')

    ncol = []
    ncol.append(cols[dt])    
    ncol.append(cols[hour])

    ncol2 = [el for i,el in enumerate(cols) if i not in ncol]    
    
    data = data.append(data2[ncol2], ignore_index=True)
    data = data.append(data3[ncol2], ignore_index=True)
    data = data.append(data4[ncol2], ignore_index=True)
    data = data.append(data5[ncol2], ignore_index=True)
    data = data.append(data6[ncol2], ignore_index=True)
    
    start = time.time()
    test, y = rolling.create_rolling_dataset(data,"ven", "CSUD",meteo,1,0,24)
    end = time.time()
    print end-start
    return test, y
###################################################################################

if __name__ == '__main__':
    t,y = Launcher()
    
