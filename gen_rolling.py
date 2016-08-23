# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:52:30 2016

@author: utente

Generation of all ROLLING datasets
"""

import pandas as pd
import numpy as np
import time
import sys, os
sys.path.append('C:/Users/utente/Python Scripts')
import rolling

data = pd.read_excel("C:/Users/utente/Documents/PUN/Anno 2011.xlsx")
meteo = pd.read_table("C:/Users/utente/Documents/PUN/storico_roma.txt", sep="\t")

start = time.time()
test, y = create_rolling_dataset(data,"sab", "CSUD",meteo,1,0,24)
end = time.time()
print end-start

