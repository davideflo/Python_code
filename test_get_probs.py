# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:58:54 2016

@author: d_floriello

get_probabilities from pun
"""

import pandas as pd
import sys
import pun_prob

#path = 'prediction_PUN_2016-09-02.xlsx'

path = sys.argv[1]
dataset = sys.argv[2]

df = pun_prob.get_Probabilities2(path, dataset)

df = pun_prob.get_Probabilities(path)
#df = df.set_index(['upwards_prob', 'error'])

for i in df.columns:
    print("probability in {} of going up = {}".format(i, df[i].ix[0]))