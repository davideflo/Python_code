# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:53:02 2017

@author: utente

Meteo Scheduler
"""

import datetime
import threading

####################################################################################################
def Executer():
    execfile('NewMeteoExecuter.py')
####################################################################################################
def CheckTimeDifference(today, now):
    if now.day - today.day >= 1:
        return True
    else:
        return False
####################################################################################################
def Job(today, now):
    b = CheckTimeDifference(today, now)
    if b:
        Executer()
    else:
        print 'same day'
####################################################################################################
Executer()
today = datetime.datetime.today()

secs = 3600*24    
secs2 = 3600*48
secs3 = 3600*72    
secs4 = 3600*96


T1 = threading.Timer(secs, Executer)
T2 = threading.Timer(secs2, Executer)
T3 = threading.Timer(secs3, Executer)
T4 = threading.Timer(secs4, Executer)

T1.start()
T2.start()
T3.start()
T4.start()