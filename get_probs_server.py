# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 12:30:44 2016

@author: utente

get_probabilities server
"""
import sys
import pun_prob_server

# path = 'prediction_PUN_2016-09-08.xlsx'
# dataset =  'data7'

path = sys.argv[1]
dataset = sys.argv[2]

df = pun_prob_server.get_Probabilities2(path, dataset)

df2 = pun_prob_server.get_Probabilities(path)

for i in df.columns:
    print("probability in {} of going up = {}".format(i, df[i].ix[0]))

for i in df.columns:
    print("probability in past years in {} of going up = {}".format(i, df2[i].ix[0]))
    
pun_prob_server.Find_Anomalies(df)

EL = pun_prob_server.get_ExpectedLoss(path, dataset)

for i in EL.columns:
    print("Expected Loss in {} = {}".format(i, EL[i].ix[0]))
