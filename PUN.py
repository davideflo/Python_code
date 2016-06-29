# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:46:50 2016

@author: d_floriello
"""

## activate snakes
## to deactivate: deactivate


import temp
import pandas as pd
import numpy as np
import h2o

data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")

data = data.ix[0:8759]

D,Y = temp.create_dataset(data, "ven")

h2o.init()

Ddict = D.to_dict()
 
df = h2o.H2OFrame.from_python(Ddict)
y = h2o.H2OFrame.from_python(np.array(Y))

