# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:59:20 2017

@author: d_floriello

Launcher
"""

if __name__ == '__main__':
    
    runfile('C:/Users/d_floriello/Documents/Python Scripts/FFDE.py', wdir='C:/Users/d_floriello/Documents/Python Scripts')    
    today = datetime.datetime.now()
    ZIPExtractor()

    mdf = Aggregator(today)

    mdf.to_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
