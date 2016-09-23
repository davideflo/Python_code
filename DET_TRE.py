# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:42:15 2016

@author: d_floriello

method for automatically detecting trends in time series
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

###############################################################################
def find_trends(ts, list_x, size):
    err = 0
    beta_start = []
    
    for i in range(len(list_x)-1):
        X_ = np.array(list(range(list_x[i], list_x[i+1], 1))).reshape(-1, 1)
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        print('size X_ = {}; size ts[] = {}'.format(X_.size, ts[list_x[i]:list_x[i+1]].size))
        model_ransac.fit(X_, ts[list_x[i]:list_x[i+1]])
        beta_start.append(model_ransac.estimator_.coef_)
        line_y_ransac = model_ransac.predict(X_)
        err += np.mean((ts[list_x[i]:list_x[i+1]] - line_y_ransac)**2)  
        
    if size - list_x[-1] > 3:
        X_ = np.array(list(range(list_x[-1], size, 1))).reshape(-1, 1)
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(X_, ts[list_x[-1]:])
        beta_start.append(model_ransac.estimator_.coef_)
        line_y_ransac = model_ransac.predict(X_)
        err += np.mean((ts[list_x[-1]:] - line_y_ransac)**2)  
        
    return err, beta_start
###############################################################################
def detect_trends(ts, lam = 1):

    changes = 0       
    loss = 0
    num_int_start = 0
    size = 0

    if isinstance(ts.index, pd.DatetimeIndex):
        num_int_start  = np.max(ts.index.month)
        size = ts.index.month.size
    else:
        num_int_start = np.ceil(ts.size/30)
        size = ts.size
    
    list_x = list(range(0, size, np.int(num_int_start)))
        
    err, beta_start = find_trends(ts, list_x, size)        
        
    ic = []
    for i in range(len(beta_start)-1):
        if beta_start[i] * beta_start[i+1] < 0:
            changes += 1
            ic.append(i+1)            
    
    print(ic)
    loss = err + lam * (num_int_start + changes)
    print(list_x)
     
    if changes == 0:
        X_ = np.array(list(range(size))).reshape(-1, 1)
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(X_, ts)
        line_y_ransac = model_ransac.predict(X_)
        best_error = np.mean((ts - line_y_ransac)**2)  
        return best_error, model_ransac.estimator_.coef_, list_x        
        
    else:
        for j in range(len(ic)):
            print('popping {}'.format(ic[j]))            
            list_x.pop((ic[j]-j))
            
        xnew = list_x    
        err2, beta2 = find_trends(ts, xnew, size)
        changes2 = 0        
        for i in range(len(beta2)-1):
            if beta2[i] * beta2[i+1] < 0:
                changes2 += 1
        
        if loss > err2 + lam * (changes2 + len(xnew)):
            return err2, beta2, xnew
        else: 
            return err, beta_start, list_x
            