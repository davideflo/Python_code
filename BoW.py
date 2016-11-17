# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:28:41 2016

@author: d_floriello

Bag of Words
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer


cols = pd.read_excel('vectorized2.xlsx', sheetname = 'Sheet2')


macros = cols.index.tolist()

n_cols = cols.shape[1]
el_cols = cols.columns.tolist()
diz = OrderedDict()

for m in macros:
    c = m.find('.')
    cat = m[5:c]
    for col in el_cols:
        if cols[col].ix[m] > 0:
            if cat in diz.keys():
                diz[cat].append(col.replace('_', ' '))
            else:
                diz[cat] = [col.replace('_', ' ')]
                

            
vectorizer = CountVectorizer(min_df = 1)
X_pagam = vectorizer.fit_transform(diz['agam'])
            
features = vectorizer.get_feature_names()

#vectorizer.transform([string]).toarray().size
