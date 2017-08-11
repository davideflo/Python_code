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

secs = 3600*24    # 12
secs2 = 3600*24*2    # 13
secs3 = 3600*24*3    # 14
secs4 = 3600*24*4    # 15
secs5 = 3600*24*5    # 16
secs6 = 3600*24*6    # 17
secs7 = 3600*24*7    # 18
secs8 = 3600*24*8    # 19
secs9 = 3600*24*9    # 20
secs10 = 3600*24*10    # 21
secs11 = 3600*24*11    # 22
secs12 = 3600*24*12    # 23
secs13 = 3600*24*12    # 24
secs14 = 3600*24*14    # 25
secs15 = 3600*24*15    # 26
secs16 = 3600*24*16    # 27
secs17 = 3600*24*17    # 28
secs18 = 3600*24*18    # 29
secs19 = 3600*24*19    # 30
secs20 = 3600*24*20    # 31

secs21 = 3600*24*21
secs22 = 3600*24*22    
secs23 = 3600*24*23
secs24 = 3600*24*24
secs25 = 3600*24*25


T1 = threading.Timer(secs, Executer)
T2 = threading.Timer(secs2, Executer)
T3 = threading.Timer(secs3, Executer)
T4 = threading.Timer(secs4, Executer)
T5 = threading.Timer(secs5, Executer)
T6 = threading.Timer(secs6, Executer)
T7 = threading.Timer(secs7, Executer)
T8 = threading.Timer(secs8, Executer)
T9 = threading.Timer(secs9, Executer)
T10 = threading.Timer(secs10, Executer)
T11 = threading.Timer(secs11, Executer)
T12 = threading.Timer(secs12, Executer)
T13 = threading.Timer(secs13, Executer)
T14 = threading.Timer(secs14, Executer)
T15 = threading.Timer(secs15, Executer)
T16 = threading.Timer(secs16, Executer)
T17 = threading.Timer(secs17, Executer)
T18 = threading.Timer(secs18, Executer)
T19 = threading.Timer(secs19, Executer)
T20 = threading.Timer(secs20, Executer)
T21 = threading.Timer(secs21, Executer)
T22 = threading.Timer(secs22, Executer)
T23 = threading.Timer(secs23, Executer)
T24 = threading.Timer(secs24, Executer)
T25 = threading.Timer(secs25, Executer)

T1.start()
T2.start()
T3.start()
T4.start()
T5.start()
T6.start()
T7.start()
T8.start()
T9.start()
T10.start()
T11.start()
T12.start()
T13.start()
T14.start()
T15.start()
T16.start()
T17.start()
T18.start()
T19.start()
T20.start()
T21.start()
T22.start()
T23.start()
T24.start()
T25.start()