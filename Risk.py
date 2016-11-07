# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:35:02 2016

@author: d_floriello

Example Risk calculation
"""

import numpy as np
import sdeint
import matplotlib.pyplot as plt

# Weiner Process Simulation



# Time steps
n = 1000 
T = 1.  

###############################################################################
def Wiener(mu, sig, n, T):
    Delta = T/n
#    t = np.arange(1, step=Delta)
    S = np.zeros(n, np.dtype(float))
    x = range(1, len(S))
    for i in x:
        dW = np.random.normal(mu, sig)
        dt = np.sqrt(Delta)
        dS = dW*dt
        S[i] = S[i-1] + dS
    return S
###############################################################################

plt.figure()
plt.plot(Wiener(0,1,n,T))

def mu(x,t):
    return 0

def SIG(x, t):
    return x

tspan = np.linspace(0.0, 1.0, 10001)
res = sdeint.itoint(mu, SIG, 0, tspan)

plt.figure()
plt.plot(res.ravel())