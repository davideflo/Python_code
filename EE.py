# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 11:34:46 2016

@author: utente

script to use EE_interp.py from command line
"""

import sys
import EE_interp

#path = 'prediction_PUN_2016-09-02.xlsx'
### when put path as an argument in command line, remove "'" (hyphen)
path = sys.argv[1]

df = EE_interp.Extract_Dataset(path)

print 'dimensions of the updated dataset: {}'.format(df.shape)

EE_interp.get_InterpolatedCurves(path)