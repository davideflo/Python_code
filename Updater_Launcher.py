# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:39:18 2017

@author: d_floriello

Updater Launcher
"""

import threading
import datetime

###############################################################################
def Executer():
    exec(open('BLauncher.py').read())        
###############################################################################

secs = 3600*18    
secs2 = secs + 3600*24
secs3 = secs2 + 3600*24
secs6 = secs3 + 3600*24
secs7 = secs6 + 3600*24*3
secs8 = secs7 + 3600*24



T1 = threading.Timer(secs, Executer)
T2 = threading.Timer(secs2, Executer)
T3 = threading.Timer(secs3, Executer)
T6 = threading.Timer(secs6, Executer)
T7 = threading.Timer(secs7, Executer)
T8 = threading.Timer(secs8, Executer)


T1.start()
T2.start()
T3.start()
T6.start()
T7.start()
T8.start()


