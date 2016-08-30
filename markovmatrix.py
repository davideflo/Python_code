# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:10:22 2016

@author: utente
"""

import os
os.chdir('C:/Users/utente/Documents/Python Scripts')
import PUN_distribution
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    data = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2010.xlsx", sheetname=1)
    data2 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2011.xlsx", sheetname=0)
    data3 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2012.xlsx", sheetname=0)
    data4 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2013.xlsx", sheetname=0)
    data5 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2014.xlsx", sheetname=0)
    data6 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2015.xlsx", sheetname=0)
    data7 = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2016_07.xlsx", sheetname=1)
    
    varn = "PUN"
    pun1 = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
                                   np.array(data3[varn]),
                             np.array(data4[varn]), np.array(data5[varn]), 
                                np.array(data6[varn]), np.array(data7[varn])])
                                
    pun = {"PUN": pun1}
    rng2 = pd.date_range(start="01-01-2010", periods = pun1.size,freq = 'H')
    ixx = np.arange(56951)
    rng = pd.DataFrame([rng2,ixx])
    rng.columns = [["rng", "ixx"]]
    rng.columns = ["rng"]
    dd = {"ixx": ixx, "pun": pun1}
    ap = pd.DataFrame(dd).set_index(rng2)
    
    ap2 = pd.DataFrame(pun1)
    ap2.columns = ["pun"]
    
    months = np.unique(rng.month)
    
    M = PUN_distribution.FindMarkovMatrix(ap,ap2,8,varn)
    print np.linalg.det(M)